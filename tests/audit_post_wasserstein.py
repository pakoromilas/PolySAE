#!/usr/bin/env python3
"""Re-audit after the SAEBench-side Wasserstein scoping change.

Runs Checks 8, 9, 10, 11 (reused from audit_against_dev.py) plus four new
checks (12, 13, 14, 15, 16) specific to the Wasserstein change set.

Run inside the audit venv:
    /tmp/polysae-audit/bin/python tests/audit_post_wasserstein.py
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import textwrap
import traceback
from pathlib import Path

# Reuse helpers + the four numerical checks from the prior audit.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_against_dev as base  # noqa: E402


DEV = base.DEV
PUB = base.PUB
PASS, FAIL, WARN, SKIP = base.PASS, base.FAIL, base.WARN, base.SKIP


# ---------------------------------------------------------------------------
# CHECK 12: Wasserstein scoping (grep + programmatic contract test)
# ---------------------------------------------------------------------------

def check_12_wasserstein_scoping() -> None:
    detail: list[str] = []
    problems: list[str] = []

    # --- 12a. Grep every wasserstein hit under PolySAE/ ---
    hits: list[str] = []
    for p in base.walk_files(PUB):
        if p.suffix not in {".py", ".md", ".txt", ".json"}:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if "wasserstein" in line.lower():
                hits.append(f"{base.rel(p, PUB)}:{i}: {line.strip()[:160]}")

    detail.append(f"--- grep -rn -i wasserstein ({len(hits)} hits) ---")
    detail.extend("  " + h for h in hits)

    # --- 12b. Programmatic contract test on train_probe_on_activations ---
    detail.append("--- programmatic contract test ---")
    try:
        import torch
        from sae_bench.evals.sparse_probing import probe_training

        torch.manual_seed(0)
        # Two "professions" so the per-class loop fires twice.
        train = {"p1": torch.randn(40, 8), "p2": torch.randn(40, 8)}
        test = {"p1": torch.randn(20, 8), "p2": torch.randn(20, 8)}

        _, _, _, w_true = probe_training.train_probe_on_activations(
            train, test, select_top_k=1, compute_wasserstein=True,
        )
        _, _, _, w_false = probe_training.train_probe_on_activations(
            train, test, select_top_k=5, compute_wasserstein=False,
        )

        detail.append(f"  compute_wasserstein=True  → wasserstein={w_true}")
        detail.append(f"  compute_wasserstein=False → wasserstein={w_false}")

        if not (isinstance(w_true, dict) and len(w_true) == 2 and
                all(isinstance(v, float) for v in w_true.values())):
            problems.append(
                f"compute_wasserstein=True returned {w_true!r}; expected non-empty dict[str, float]"
            )
        if w_false != {}:
            problems.append(
                f"compute_wasserstein=False returned {w_false!r}; expected empty dict"
            )
    except Exception as e:
        problems.append(f"programmatic call raised: {type(e).__name__}: {e}")
        detail.append("  " + traceback.format_exc().splitlines()[-1])

    # --- 12c. Verify main.py k-loop scoping by reading source AST-style ---
    main_src = (PUB / "SAEBench/sae_bench/evals/sparse_probing/main.py").read_text()
    cw_count = len(re.findall(r"compute_wasserstein\s*=\s*\(\s*k\s*==\s*1\s*\)", main_src))
    if_k1_writes = re.findall(
        r"if\s+k\s*==\s*1\s*:\s*\n\s+\S*_results\S*\[f[\"']\S*_top_\{k\}_wasserstein[\"']]",
        main_src,
    )
    detail.append(
        f"--- main.py k-loop scoping: "
        f"compute_wasserstein=(k==1) → {cw_count}×; "
        f"if k==1 → wasserstein dict-write blocks → {len(if_k1_writes)}× ---"
    )
    if cw_count != 2:
        problems.append(f"expected 2 occurrences of compute_wasserstein=(k == 1) in main.py, found {cw_count}")
    if len(if_k1_writes) != 2:
        problems.append(
            f"expected 2 `if k == 1:` blocks guarding *_wasserstein writes in main.py, found {len(if_k1_writes)}"
        )

    if problems:
        base.banner("12.wasserstein-scoping", FAIL,
                    f"{len(problems)} scoping problem(s)",
                    "\n".join(detail + ["", "PROBLEMS:"] + ["  " + p for p in problems]))
    else:
        base.banner("12.wasserstein-scoping", PASS,
                    f"contract holds: True→non-empty dict, False→empty dict; "
                    f"main.py has 2× compute_wasserstein=(k==1) + 2× `if k == 1:` write-guard",
                    "\n".join(detail))


# ---------------------------------------------------------------------------
# CHECK 13: eval_output.py byte-identical to dev
# ---------------------------------------------------------------------------

EVAL_OUTPUT = "SAEBench/sae_bench/evals/sparse_probing/eval_output.py"


def check_13_eval_output_unchanged() -> None:
    dev_p = DEV / EVAL_OUTPUT
    pub_p = PUB / EVAL_OUTPUT
    if not dev_p.exists() or not pub_p.exists():
        base.banner("13.eval-output-schema", FAIL,
                    "eval_output.py missing in one of the repos",
                    f"dev exists={dev_p.exists()}; pub exists={pub_p.exists()}")
        return
    h_dev = base.sha256(dev_p)
    h_pub = base.sha256(pub_p)
    if h_dev != h_pub:
        diff = "\n".join(difflib.unified_diff(
            dev_p.read_text().splitlines(),
            pub_p.read_text().splitlines(),
            fromfile=f"dev/{EVAL_OUTPUT}", tofile=f"pub/{EVAL_OUTPUT}", lineterm="",
        )[:80])
        base.banner("13.eval-output-schema", FAIL,
                    "eval_output.py was modified (agent claimed it wasn't)",
                    f"sha256 dev={h_dev[:16]} pub={h_pub[:16]}\n{diff}")
    else:
        base.banner("13.eval-output-schema", PASS,
                    f"byte-identical (sha256={h_dev[:16]}…)")


# ---------------------------------------------------------------------------
# CHECK 14: probe_training.py + main.py changes are scoped
# ---------------------------------------------------------------------------

def _function_line_range(src: str, target_name: str) -> tuple[int, int] | None:
    """Return (start_line, end_line) (1-indexed, inclusive) of `target_name`."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target_name:
            start = node.lineno
            # end_lineno needs Python 3.8+; we're on 3.11
            end = node.end_lineno or start
            return (start, end)
    return None


def _classify_diff(dev_src: str, pub_src: str, dev_range: tuple[int, int],
                   pub_range: tuple[int, int]) -> tuple[list[int], list[int]]:
    """Return (dev_lines_changed_outside, pub_lines_changed_outside).

    A line is "changed" if it appears in the diff as removed (dev side) or
    added (pub side). "Outside" = outside the supplied function line range.
    """
    dev_lines = dev_src.splitlines()
    pub_lines = pub_src.splitlines()
    sm = difflib.SequenceMatcher(a=dev_lines, b=pub_lines, autojunk=False)

    out_of_range_dev: list[int] = []
    out_of_range_pub: list[int] = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        # 1-indexed line numbers of the change region in each file
        for ln in range(i1 + 1, i2 + 1):
            if not (dev_range[0] <= ln <= dev_range[1]):
                out_of_range_dev.append(ln)
        for ln in range(j1 + 1, j2 + 1):
            if not (pub_range[0] <= ln <= pub_range[1]):
                out_of_range_pub.append(ln)
    return out_of_range_dev, out_of_range_pub


def _print_function_diff(dev_p: Path, pub_p: Path, max_lines: int = 80) -> str:
    diff = list(difflib.unified_diff(
        dev_p.read_text().splitlines(),
        pub_p.read_text().splitlines(),
        fromfile=f"dev/{dev_p.relative_to(DEV)}",
        tofile=f"pub/{pub_p.relative_to(PUB)}",
        lineterm="", n=3,
    ))
    if len(diff) > max_lines:
        return "\n".join(diff[:max_lines] + [f"... ({len(diff)-max_lines} more diff lines)"])
    return "\n".join(diff)


def check_14_minimal_edits() -> None:
    """Both edited files: changes must be contained to the target function,
    and the target function must contain the expected new constructs."""
    problems: list[str] = []
    details: list[str] = []

    # --- probe_training.py : changes in train_probe_on_activations only ---
    pt_dev = DEV / "SAEBench/sae_bench/evals/sparse_probing/probe_training.py"
    pt_pub = PUB / "SAEBench/sae_bench/evals/sparse_probing/probe_training.py"
    dev_src = pt_dev.read_text()
    pub_src = pt_pub.read_text()
    dev_range = _function_line_range(dev_src, "train_probe_on_activations")
    pub_range = _function_line_range(pub_src, "train_probe_on_activations")
    if dev_range is None or pub_range is None:
        problems.append("probe_training.py: train_probe_on_activations not found in one of the repos")
    else:
        oor_dev, oor_pub = _classify_diff(dev_src, pub_src, dev_range, pub_range)
        if oor_dev or oor_pub:
            problems.append(
                f"probe_training.py: changes leak OUTSIDE train_probe_on_activations "
                f"(dev lines {oor_dev[:10]}…, pub lines {oor_pub[:10]}…)"
            )

        # Required structural additions
        if "compute_wasserstein: bool" not in pub_src:
            problems.append("probe_training.py: missing `compute_wasserstein: bool` kwarg")
        if re.search(r"if\s+compute_wasserstein\s*:", pub_src) is None:
            problems.append("probe_training.py: missing `if compute_wasserstein:` conditional")
        # Required structural absences in dev (sanity)
        if "compute_wasserstein" in dev_src:
            problems.append("probe_training.py: dev unexpectedly already references compute_wasserstein")

        details.append(
            f"probe_training.py: function lines dev[{dev_range[0]}-{dev_range[1]}] "
            f"pub[{pub_range[0]}-{pub_range[1]}]; diff fully inside function = "
            f"{not (oor_dev or oor_pub)}"
        )

    # --- main.py : changes in run_eval_single_dataset only ---
    m_dev = DEV / "SAEBench/sae_bench/evals/sparse_probing/main.py"
    m_pub = PUB / "SAEBench/sae_bench/evals/sparse_probing/main.py"
    dev_src = m_dev.read_text()
    pub_src = m_pub.read_text()
    dev_range = _function_line_range(dev_src, "run_eval_single_dataset")
    pub_range = _function_line_range(pub_src, "run_eval_single_dataset")
    if dev_range is None or pub_range is None:
        problems.append("main.py: run_eval_single_dataset not found in one of the repos")
    else:
        oor_dev, oor_pub = _classify_diff(dev_src, pub_src, dev_range, pub_range)
        if oor_dev or oor_pub:
            problems.append(
                f"main.py: changes leak OUTSIDE run_eval_single_dataset "
                f"(dev lines {oor_dev[:10]}…, pub lines {oor_pub[:10]}…)"
            )

        cw_count = len(re.findall(r"compute_wasserstein\s*=\s*\(\s*k\s*==\s*1\s*\)", pub_src))
        if cw_count != 2:
            problems.append(f"main.py: expected 2× compute_wasserstein=(k == 1), got {cw_count}")
        if_k1 = len(re.findall(r"if\s+k\s*==\s*1\s*:\s*\n\s+\S*_results\S*\[f[\"']\S*_top_\{k\}_wasserstein[\"']]", pub_src))
        if if_k1 != 2:
            problems.append(f"main.py: expected 2× `if k == 1:` guards on wasserstein writes, got {if_k1}")
        if "compute_wasserstein" in dev_src:
            problems.append("main.py: dev unexpectedly already references compute_wasserstein")

        details.append(
            f"main.py: function lines dev[{dev_range[0]}-{dev_range[1]}] "
            f"pub[{pub_range[0]}-{pub_range[1]}]; diff fully inside function = "
            f"{not (oor_dev or oor_pub)}"
        )

    if problems:
        diff_pt = _print_function_diff(pt_dev, pt_pub)
        diff_m = _print_function_diff(m_dev, m_pub)
        base.banner("14.scoped-edits", FAIL,
                    f"{len(problems)} scoping/structure problem(s)",
                    "\n".join(details + ["", "PROBLEMS:"] +
                              ["  " + p for p in problems] +
                              ["", "--- probe_training.py diff ---", diff_pt,
                               "", "--- main.py diff ---", diff_m]))
    else:
        base.banner("14.scoped-edits", PASS,
                    "both files: all changes contained to target function + expected constructs present",
                    "\n".join(details))


# ---------------------------------------------------------------------------
# CHECK 15: no unrelated SAEBench file edits
# ---------------------------------------------------------------------------

SAEBENCH_ALLOWED_EDITS = {
    "SAEBench/pyproject.toml",
    "SAEBench/sae_bench/evals/sparse_probing/probe_training.py",
    "SAEBench/sae_bench/evals/sparse_probing/main.py",
}


def check_15_no_other_saebench_edits() -> None:
    EXCLUDE = base.EXCLUDE_DIRS

    def _saebench_files(root: Path) -> set[str]:
        out: set[str] = set()
        sb = root / "SAEBench"
        for dirpath, dirnames, filenames in os.walk(sb):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE]
            for f in filenames:
                p = Path(dirpath) / f
                out.add(str(p.relative_to(root)))
        return out

    dev_set = _saebench_files(DEV)
    pub_set = _saebench_files(PUB)

    only_in_dev = sorted(dev_set - pub_set)
    only_in_pub = sorted(pub_set - dev_set)
    shared = sorted(dev_set & pub_set)

    diffs: list[str] = []
    for relpath in shared:
        h_dev = base.sha256(DEV / relpath)
        h_pub = base.sha256(PUB / relpath)
        if h_dev != h_pub and relpath not in SAEBENCH_ALLOWED_EDITS:
            diffs.append(f"  ~ {relpath}  dev={h_dev[:12]}  pub={h_pub[:12]}")

    problems: list[str] = []
    if only_in_dev:
        problems.append(f"files removed from SAEBench/ ({len(only_in_dev)}):")
        problems += [f"  - {f}" for f in only_in_dev]
    if only_in_pub:
        problems.append(f"files added to SAEBench/ ({len(only_in_pub)}):")
        problems += [f"  + {f}" for f in only_in_pub]
    if diffs:
        problems.append(f"unexpected SAEBench file edits ({len(diffs)}):")
        problems += diffs

    if problems:
        base.banner("15.no-other-saebench-edits", FAIL,
                    f"unexpected SAEBench tree changes",
                    "\n".join(problems))
    else:
        base.banner("15.no-other-saebench-edits", PASS,
                    f"only the 3 allow-listed SAEBench files differ; "
                    f"{len(shared)} shared SAEBench files compared")


# ---------------------------------------------------------------------------
# CHECK 16: sparse-probing pipeline smoke
# ---------------------------------------------------------------------------

def check_16_pipeline_smoke() -> None:
    """The minimum acceptable version per the spec: call
    train_probe_on_activations directly with synthetic inputs in both modes.

    Additionally, import sparse_probing.main (and walk both k-loop call
    patterns by hand) to confirm the gated dict-write semantics behave as
    main.py would.
    """
    try:
        import torch
        from sae_bench.evals.sparse_probing import probe_training
        # Confirm main module is importable (catches refactor breakage)
        from sae_bench.evals.sparse_probing import main as sp_main  # noqa: F401
    except Exception as e:
        base.banner("16.pipeline-smoke", FAIL,
                    "sparse_probing modules not importable",
                    f"{type(e).__name__}: {e}")
        return

    torch.manual_seed(0)
    # 8 features, 32 samples, 2 classes (= 2 "professions")
    train = {"p1": torch.randn(32, 8), "p2": torch.randn(32, 8)}
    test = {"p1": torch.randn(16, 8), "p2": torch.randn(16, 8)}

    # Simulate main.py's k-loop with k_values = [1, 2, 5]; only k=1 should
    # contribute a *_wasserstein entry.
    results_dict: dict = {}
    try:
        for k in [1, 2, 5]:
            _, accs, f1s, wassersteins = probe_training.train_probe_on_activations(
                train, test, select_top_k=k, compute_wasserstein=(k == 1),
            )
            results_dict[f"sae_top_{k}_test_accuracy"] = accs
            results_dict[f"sae_top_{k}_test_f1"] = f1s
            if k == 1:
                results_dict[f"sae_top_{k}_wasserstein"] = wassersteins
    except Exception as e:
        base.banner("16.pipeline-smoke", FAIL,
                    "k-loop simulation raised",
                    f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        return

    problems: list[str] = []
    # Required keys
    if "sae_top_1_wasserstein" not in results_dict:
        problems.append("results dict is missing sae_top_1_wasserstein")
    else:
        ws = results_dict["sae_top_1_wasserstein"]
        if not (isinstance(ws, dict) and len(ws) == 2):
            problems.append(f"sae_top_1_wasserstein dict shape unexpected: {ws}")
    # Forbidden keys
    for k in (2, 5):
        key = f"sae_top_{k}_wasserstein"
        if key in results_dict:
            problems.append(f"results dict unexpectedly contains {key} (should be omitted at k>1)")
    # F1 keys must still exist for every k
    for k in (1, 2, 5):
        for stat in ("test_f1", "test_accuracy"):
            key = f"sae_top_{k}_{stat}"
            if key not in results_dict:
                problems.append(f"results dict is missing {key} (F1/accuracy must remain for all k)")

    summary = sorted(results_dict.keys())
    if problems:
        base.banner("16.pipeline-smoke", FAIL,
                    f"{len(problems)} contract violation(s) in simulated k-loop",
                    "\n".join(["  " + p for p in problems] +
                              ["", "result keys: " + ", ".join(summary)]))
    else:
        base.banner("16.pipeline-smoke", PASS,
                    f"sparse_probing.main importable; simulated k=[1,2,5] loop emits "
                    f"sae_top_1_wasserstein (only) and F1/accuracy at every k",
                    "  result keys: " + ", ".join(summary))


# ---------------------------------------------------------------------------
# SKIP placeholders (documented for GPU box)
# ---------------------------------------------------------------------------

def skip_gpu_only() -> None:
    base.banner("S1.gpu-training-smoke", SKIP,
                "Full training smoke on the 8 paper commands needs a GPU; GPU-box only.")
    base.banner("S2.large-LM-forward", SKIP,
                "Pythia-1.4B / Gemma-2-2B forward passes need to download LM checkpoints; GPU-box only.")
    base.banner("S3.saebench-end-to-end", SKIP,
                "End-to-end SAEBench eval on a trained checkpoint needs trained weights; GPU-box only.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 76)
    print("PolySAE post-Wasserstein re-audit")
    print(f"  dev: {DEV}")
    print(f"  pub: {PUB}")
    print(f"  python: {sys.executable}")
    print("=" * 76)

    base.results.clear()  # fresh tally

    checks = [
        base.check_8_forward_pass,
        base.check_9_one_optimizer_step,
        base.check_10_param_count,
        base.check_11_state_dict_keys,
        check_12_wasserstein_scoping,
        check_13_eval_output_unchanged,
        check_14_minimal_edits,
        check_15_no_other_saebench_edits,
        check_16_pipeline_smoke,
    ]
    for fn in checks:
        try:
            fn()
        except Exception:
            base.banner(fn.__name__, FAIL,
                        "uncaught exception in audit check",
                        traceback.format_exc())

    skip_gpu_only()

    counts = {PASS: 0, FAIL: 0, WARN: 0, SKIP: 0}
    for _, status, _, _ in base.results:
        counts[status] += 1

    print("\n" + "=" * 76)
    print(f"SUMMARY: pass={counts[PASS]} fail={counts[FAIL]} "
          f"warn={counts[WARN]} skip={counts[SKIP]}")
    print("=" * 76)

    if counts[FAIL]:
        print("\nFAILURES:")
        for check, status, msg, details in base.results:
            if status == FAIL:
                print(f"  [{check}] {msg}")
                for line in details.splitlines()[:10]:
                    print(f"        {line}")

    return 1 if counts[FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
