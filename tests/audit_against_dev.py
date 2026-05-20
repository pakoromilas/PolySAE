#!/usr/bin/env python3
"""Release-readiness audit: PolySAE/ vs PolySAE_dev/.

Runs 14 independent checks; one failure does not skip the rest. Exits
non-zero if any check fails. Does NOT auto-fix anything — reports only.

Run inside the audit venv:
    /tmp/polysae-audit/bin/python tests/audit_against_dev.py
"""
from __future__ import annotations

import dataclasses
import difflib
import hashlib
import importlib
import inspect
import json
import os
import re
import subprocess
import sys
import textwrap
import traceback
from pathlib import Path

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DEV = Path("/Users/koro/repos/tmp/PolySAE_dev").resolve()
PUB = Path("/Users/koro/repos/tmp/PolySAE").resolve()

# Files intentionally dropped from dev → public.
DROP_LIST = {
    "sweep_runner.py",
    "sweep_runner_linear_topk.py",
    "sweep_runner_polysae.py",
    "debug_folding.py",
    "test_polysae_folding.py",
    "test_saebench_polysae.py",
    "run_calibration.sh",
    "calibration_results.log",
    "run_commands.txt",
    "__init__.py",  # the EMPTY root __init__.py only
}

# Files intentionally added in public.
ADD_LIST = {
    ".gitignore",
    "tests/test_polysae_imports.py",
    "tests/test_polysae_cli.py",
    "tests/saes/test_polysae_smoke.py",
    "tests/audit_against_dev.py",
}

# Files allowed to differ in content between dev and public.
EDIT_LIST = {
    "train_and_saebench.py",
    "train_sae.py",
    "scripts/replication_how_train_saes.py",
    "scripts/replication_how_train_saes_control.py",
    "SAEBench/pyproject.toml",
    "requirements.freeze.txt",
    "requirements.hpc.txt",
    "README.md",
}

EXCLUDE_DIRS = {".git", "__pycache__", ".venv", ".pytest_cache", ".ruff_cache",
                ".mypy_cache", "wandb", "outputs", "checkpoints", "eval_results",
                ".claude", ".pytest_cache"}

# --------------------------------------------------------------------------
# Result tracking
# --------------------------------------------------------------------------

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"
results: list[tuple[str, str, str, str]] = []  # (check, status, message, details)


def banner(check: str, status: str, message: str, details: str = "") -> None:
    color = {"PASS": "32", "FAIL": "31", "WARN": "33", "SKIP": "36"}[status]
    print(f"\033[{color}m[{status}]\033[0m {check}: {message}")
    if details:
        for line in details.splitlines():
            print(f"        {line}")
    results.append((check, status, message, details))


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def walk_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for f in filenames:
            p = Path(dirpath) / f
            out.append(p)
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p: Path, root: Path) -> str:
    return str(p.relative_to(root))


def run_subprocess(args: list[str], env: dict | None = None,
                   cwd: Path | None = None, timeout: int = 300) -> subprocess.CompletedProcess:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    return subprocess.run(
        args, capture_output=True, text=True, env=full_env,
        cwd=str(cwd) if cwd else None, timeout=timeout,
    )


def py_with_pythonpath(repo: Path) -> dict[str, str]:
    """Return env-var overlay that resolves sae_lens / sae_bench to `repo`.

    PYTHONPATH entries take precedence over site-packages, so the editable
    install (which points at PUB) is shadowed for the duration of the call.
    """
    extra = f"{repo}{os.pathsep}{repo / 'SAEBench'}"
    return {"PYTHONPATH": extra + os.pathsep + os.environ.get("PYTHONPATH", "")}


# --------------------------------------------------------------------------
# CHECK 1: File-tree diff
# --------------------------------------------------------------------------

def check_1_file_tree() -> None:
    dev_files = {rel(p, DEV) for p in walk_files(DEV)}
    pub_files = {rel(p, PUB) for p in walk_files(PUB)}

    # in dev not in pub
    missing_in_pub = sorted(dev_files - pub_files)
    unexpected_drops = [f for f in missing_in_pub if f not in DROP_LIST]

    # in pub not in dev
    extra_in_pub = sorted(pub_files - dev_files)
    unexpected_adds = [f for f in extra_in_pub if f not in ADD_LIST]

    # hash compare for files present in both
    hash_diffs: list[tuple[str, str, str]] = []
    for f in sorted(dev_files & pub_files):
        h_dev = sha256(DEV / f)
        h_pub = sha256(PUB / f)
        if h_dev != h_pub and f not in EDIT_LIST:
            hash_diffs.append((f, h_dev[:12], h_pub[:12]))

    problems: list[str] = []
    if unexpected_drops:
        problems.append(f"Dropped files NOT in allow-list ({len(unexpected_drops)}):")
        for f in unexpected_drops:
            problems.append(f"  - {f}")
    if unexpected_adds:
        problems.append(f"Added files NOT in allow-list ({len(unexpected_adds)}):")
        for f in unexpected_adds:
            problems.append(f"  + {f}")
    if hash_diffs:
        problems.append(f"Content differs but file NOT in edit-list ({len(hash_diffs)}):")
        for f, hd, hp in hash_diffs:
            problems.append(f"  ~ {f}  dev={hd}  pub={hp}")

    if problems:
        banner("1.file-tree", FAIL,
               f"unexpected tree differences: drops={len(unexpected_drops)} adds={len(unexpected_adds)} edits={len(hash_diffs)}",
               "\n".join(problems))
    else:
        # Also confirm DROP_LIST files truly are absent (sanity).
        confirmed_drops = [f for f in DROP_LIST if not (PUB / f).exists()]
        banner("1.file-tree", PASS,
               f"{len(dev_files & pub_files)} shared files identical; "
               f"{len(confirmed_drops)} drops + {len(ADD_LIST)} adds + "
               f"{len(EDIT_LIST)} edits as allow-listed")


# --------------------------------------------------------------------------
# CHECK 2: Scrubs
# --------------------------------------------------------------------------

def check_2_scrubs() -> None:
    patterns = [
        ("/home/paperspace", r"/home/paperspace"),
        ("/home/koromilas|suser", r"/home/(koromilas|suser)"),
        ("personal identifiers", r"(koromilas|@di\.uoa\.gr)"),
        ('default="icml', r'default="icml'),
        ('hardcoded entity="..."', r'entity="[^"$]+"'),
        ("/scratch/ or /mnt/", r"(^|[^a-zA-Z0-9_])(/scratch/|/mnt/)"),
    ]

    text_exts = {".py", ".sh", ".yaml", ".yml", ".toml", ".md", ".txt", ".cfg",
                 ".ini", ".json"}
    # Exclude the audit script itself (self-reference: it contains the regex
    # patterns it greps for) and upstream-SAELens ansible docs where /mnt/s3
    # appears as a documented AWS S3 mount example, not author personal info.
    SELF_EXCLUDE = {
        "tests/audit_against_dev.py",
        "scripts/ansible/README.md",
        "scripts/ansible/configs_example/shared.yml",
        # Upstream SAELens test fixture, not the author's wandb entity:
        "tests/helpers.py",
        # Upstream SAELens test fixtures live under tests/_comparison:
    }
    files = [p for p in walk_files(PUB)
             if p.suffix in text_exts
             and rel(p, PUB) not in SELF_EXCLUDE
             and not rel(p, PUB).startswith("tests/_comparison/")]

    hits_by_pattern: dict[str, list[str]] = {}
    for label, pat in patterns:
        regex = re.compile(pat)
        hits: list[str] = []
        for p in files:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    # Exclude false-positive: WANDB_ENTITY env-var-based entity=
                    if label == 'hardcoded entity="..."' and "WANDB_ENTITY" in line:
                        continue
                    # Exclude false-positive: site-packages references
                    if "site-packages" in line:
                        continue
                    hits.append(f"{rel(p, PUB)}:{lineno}: {line.strip()}")
        hits_by_pattern[label] = hits

    any_fail = False
    summary_lines = []
    for label, hits in hits_by_pattern.items():
        summary_lines.append(f"{label}: {len(hits)} hits")
        if hits:
            any_fail = True
            for h in hits[:10]:
                summary_lines.append(f"    {h}")
            if len(hits) > 10:
                summary_lines.append(f"    ... and {len(hits)-10} more")

    if any_fail:
        banner("2.scrubs", FAIL, "forbidden patterns still present",
               "\n".join(summary_lines))
    else:
        banner("2.scrubs", PASS, "all 6 forbidden patterns returned 0 hits",
               "\n".join(summary_lines))


# --------------------------------------------------------------------------
# CHECK 3: pip install verification
# --------------------------------------------------------------------------

def check_3_install() -> None:
    try:
        import sae_lens  # noqa: F401
        import sae_bench  # noqa: F401
    except ImportError as e:
        banner("3.install", FAIL,
               "sae_lens / sae_bench not importable in audit venv",
               f"{type(e).__name__}: {e}")
        return

    sae_lens_file = Path(sae_lens.__file__).resolve()
    sae_bench_file = Path(sae_bench.__file__).resolve()
    problems = []
    if PUB not in sae_lens_file.parents:
        problems.append(f"sae_lens leaked away from PUB: {sae_lens_file}")
    if PUB not in sae_bench_file.parents:
        problems.append(f"sae_bench leaked away from PUB: {sae_bench_file}")
    if problems:
        banner("3.install", FAIL, "bundled forks not resolving inside the repo",
               "\n".join(problems))
    else:
        banner("3.install", PASS,
               f"sae_lens at {rel(sae_lens_file, PUB)}, sae_bench at {rel(sae_bench_file, PUB)}")


# --------------------------------------------------------------------------
# CHECK 4: compileall + every-module-imports
# --------------------------------------------------------------------------

def check_4_compileall() -> None:
    cp = run_subprocess([sys.executable, "-m", "compileall", "-q", str(PUB)],
                        timeout=120)
    if cp.returncode != 0:
        banner("4a.compileall", FAIL,
               "compileall reported syntax errors",
               cp.stdout + cp.stderr)
    else:
        banner("4a.compileall", PASS, "all .py files compile")

    # Now import every module under sae_lens.* (skip tests, scripts, bundled
    # forks' own tests, the SAEBench Python).
    importable_roots = [
        ("sae_lens", PUB / "sae_lens"),
        ("sae_bench", PUB / "SAEBench" / "sae_bench"),
    ]
    import_errors: list[str] = []
    imported: list[str] = []

    for pkg, root in importable_roots:
        for py in root.rglob("*.py"):
            if any(part in EXCLUDE_DIRS for part in py.parts):
                continue
            # Convert path to dotted module name
            rel_path = py.relative_to(root.parent)
            mod = ".".join(rel_path.with_suffix("").parts)
            if mod.endswith(".__init__"):
                mod = mod[: -len(".__init__")]
            # Skip bundled forks' own test dirs
            if "/tests/" in str(py).replace(os.sep, "/"):
                continue
            # Skip upstream-SAEBench eval scripts: they have side-effect imports
            # (reading openai_api_key.txt, missing optional deps like loguru,
            # demo entry-point scripts with arg-parsing at import-time, etc.)
            # These are scripts, not API surface — not PolySAE regressions.
            if mod.startswith("sae_bench.evals."):
                continue
            # Skip notebook-companion modules
            if "testing_notebooks" in mod:
                continue
            try:
                importlib.import_module(mod)
                imported.append(mod)
            except Exception as e:
                import_errors.append(f"{mod}: {type(e).__name__}: {e}")

    if import_errors:
        banner("4b.imports", FAIL,
               f"{len(import_errors)} module(s) failed to import "
               f"(of {len(imported)+len(import_errors)})",
               "\n".join(import_errors[:25]) +
               (f"\n... and {len(import_errors)-25} more" if len(import_errors) > 25 else ""))
    else:
        banner("4b.imports", PASS, f"all {len(imported)} modules import cleanly")


# --------------------------------------------------------------------------
# CHECK 5: CLI parity
# --------------------------------------------------------------------------

def _parse_args_dump_script(target_dir: Path) -> str:
    """Return a Python source that, when run, prints JSON of train_and_saebench
    parser's flag → (default, type-name).

    Loading via importlib.util requires `sys.modules[name] = mod` BEFORE
    exec_module, otherwise `@dataclass` decorators that consult
    `sys.modules.get(cls.__module__)` get None.
    """
    return textwrap.dedent(f"""
        import json, sys, importlib.util, argparse
        sys.path.insert(0, {str(target_dir)!r})
        sys.path.insert(0, {str(target_dir / 'SAEBench')!r})

        # Intercept ArgumentParser construction to capture the parser instance.
        orig_init = argparse.ArgumentParser.__init__
        parsers = []
        def capture_init(self, *a, **kw):
            orig_init(self, *a, **kw)
            parsers.append(self)
        argparse.ArgumentParser.__init__ = capture_init

        spec = importlib.util.spec_from_file_location(
            "train_and_saebench",
            {str(target_dir / 'train_and_saebench.py')!r},
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["train_and_saebench"] = mod  # needed for @dataclass under importlib.util
        spec.loader.exec_module(mod)

        # Invoke parse_args() so all add_argument calls fire on the parser.
        sys.argv = ["train_and_saebench.py", "--help"]
        try:
            mod.parse_args()
        except SystemExit:
            pass

        parser = parsers[0]
        flags = {{}}
        for action in parser._actions:
            if not action.option_strings:
                continue
            for opt in action.option_strings:
                if opt.startswith("--"):
                    type_name = (action.type.__name__ if callable(action.type) else
                                 type(action.default).__name__)
                    flags[opt] = {{"default": repr(action.default),
                                   "type": type_name,
                                   "nargs": repr(action.nargs)}}
        print(json.dumps(flags, sort_keys=True))
    """)


def _get_train_flags(repo: Path) -> dict:
    script = _parse_args_dump_script(repo)
    cp = run_subprocess([sys.executable, "-c", script],
                        env=py_with_pythonpath(repo), timeout=60)
    if cp.returncode != 0:
        raise RuntimeError(f"flag dump for {repo} failed: {cp.stderr}")
    return json.loads(cp.stdout.strip().splitlines()[-1])


def check_5_cli_parity() -> None:
    try:
        dev_flags = _get_train_flags(DEV)
        pub_flags = _get_train_flags(PUB)
    except Exception as e:
        banner("5.cli-parity", FAIL,
               "could not extract argparse flags from one of the repos",
               f"{type(e).__name__}: {e}")
        return

    dev_set = set(dev_flags)
    pub_set = set(pub_flags)
    missing = sorted(dev_set - pub_set)
    extra = sorted(pub_set - dev_set)

    # Allowed-extra flags (per spec): --wandb, --wandb_entity
    # (--wandb_entity is part of the WandB opt-in design — wasn't in dev.)
    ALLOWED_EXTRA = {"--wandb", "--wandb_entity"}

    unexpected_extra = [f for f in extra if f not in ALLOWED_EXTRA]

    # For shared flags, compare default value and type name
    shared = sorted(dev_set & pub_set)
    diff_lines: list[str] = []
    # The --wandb_run_name help string changed (says safe-to-pass-without-wandb)
    # — argparse "default" / "type" / "nargs" are what we compare; help isn't.
    # Allow-listed wandb default-flips: --wandb_project default went from "icml_sae" → None
    ALLOWED_DEFAULT_FLIPS = {"--wandb_project"}
    for flag in shared:
        d, p = dev_flags[flag], pub_flags[flag]
        if d != p:
            if flag in ALLOWED_DEFAULT_FLIPS:
                continue
            diff_lines.append(f"  {flag}: dev={d} pub={p}")

    problems: list[str] = []
    if missing:
        problems.append(f"flags missing in public ({len(missing)}):")
        problems += [f"  - {f}" for f in missing]
    if unexpected_extra:
        problems.append(f"unexpected new flags in public ({len(unexpected_extra)}):")
        problems += [f"  + {f}" for f in unexpected_extra]
    if diff_lines:
        problems.append(f"shared-flag default/type changes ({len(diff_lines)}):")
        problems += diff_lines

    if problems:
        banner("5.cli-parity", FAIL,
               f"missing={len(missing)} unexpected-extra={len(unexpected_extra)} "
               f"shared-diffs={len(diff_lines)}",
               "\n".join(problems))
    else:
        banner("5.cli-parity", PASS,
               f"{len(shared)} shared flags equivalent; "
               f"extras=[{', '.join(extra)}] (all allow-listed)")


# --------------------------------------------------------------------------
# CHECK 6: Public API parity (dir() snapshot)
# --------------------------------------------------------------------------

API_MODULES = [
    "sae_lens",
    "sae_lens.saes.polysae",
    "sae_lens.saes.sae",
    "sae_lens.saes.topk_sae",
    "sae_lens.saes.batchtopk_sae",
    "sae_lens.saes.jumprelu_sae",
    "sae_lens.saes.matryoshka_batchtopk_sae",
    "sae_lens.saes.standard_sae",
    "sae_lens.config",
    "sae_lens.registry",
    "sae_lens.llm_sae_training_runner",
]


def _dump_api(repo: Path, modules: list[str]) -> dict:
    code = textwrap.dedent(f"""
        import json, importlib
        out = {{}}
        for m in {modules!r}:
            try:
                mod = importlib.import_module(m)
            except Exception as e:
                out[m] = {{"error": f"{{type(e).__name__}}: {{e}}"}}
                continue
            out[m] = {{
                "names": sorted(n for n in dir(mod) if not n.startswith("_")),
            }}
        print(json.dumps(out))
    """)
    cp = run_subprocess([sys.executable, "-c", code],
                        env=py_with_pythonpath(repo), timeout=120)
    if cp.returncode != 0:
        raise RuntimeError(f"API dump for {repo} failed:\n{cp.stderr}")
    return json.loads(cp.stdout.strip().splitlines()[-1])


def check_6_api_parity() -> None:
    try:
        dev = _dump_api(DEV, API_MODULES)
        pub = _dump_api(PUB, API_MODULES)
    except Exception as e:
        banner("6.api-parity", FAIL,
               "could not snapshot public API from one of the repos",
               f"{type(e).__name__}: {e}")
        return

    fails: list[str] = []
    warns: list[str] = []
    for m in API_MODULES:
        d, p = dev.get(m, {}), pub.get(m, {})
        if "error" in d or "error" in p:
            fails.append(f"  {m}: dev={d.get('error')}  pub={p.get('error')}")
            continue
        d_names = set(d["names"])
        p_names = set(p["names"])
        missing = d_names - p_names
        added = p_names - d_names
        if missing:
            fails.append(f"  {m}: names missing in public: {sorted(missing)}")
        if added:
            warns.append(f"  {m}: names new in public: {sorted(added)}")

    status = FAIL if fails else (WARN if warns else PASS)
    msg_parts = []
    if fails:
        msg_parts.append(f"{len(fails)} module(s) with missing names")
    if warns:
        msg_parts.append(f"{len(warns)} module(s) with new-name warnings")
    if not msg_parts:
        msg_parts.append(f"{len(API_MODULES)} modules identical")
    banner("6.api-parity", status, "; ".join(msg_parts),
           "\n".join(["FAIL:"] + fails + (["WARN:"] + warns if warns else [])) if (fails or warns) else "")


# --------------------------------------------------------------------------
# CHECK 7: Signature parity
# --------------------------------------------------------------------------

def _dump_signatures(repo: Path, modules: list[str]) -> dict:
    code = textwrap.dedent(f"""
        import json, importlib, inspect
        out = {{}}
        for m in {modules!r}:
            try:
                mod = importlib.import_module(m)
            except Exception as e:
                out[m] = {{"error": f"{{type(e).__name__}}: {{e}}"}}
                continue
            entries = {{}}
            for name in dir(mod):
                if name.startswith("_"):
                    continue
                obj = getattr(mod, name)
                if not (inspect.isclass(obj) or inspect.isfunction(obj)):
                    continue
                # Skip imported-from-elsewhere objects: only audit own-module defs
                try:
                    own = inspect.getmodule(obj) is mod
                except Exception:
                    own = False
                if not own:
                    continue
                try:
                    sig = str(inspect.signature(obj))
                except (TypeError, ValueError) as e:
                    sig = f"<no signature: {{e}}>"
                entries[name] = sig
            out[m] = entries
        print(json.dumps(out))
    """)
    cp = run_subprocess([sys.executable, "-c", code],
                        env=py_with_pythonpath(repo), timeout=120)
    if cp.returncode != 0:
        raise RuntimeError(f"sig dump for {repo} failed:\n{cp.stderr}")
    return json.loads(cp.stdout.strip().splitlines()[-1])


def check_7_signatures() -> None:
    try:
        dev = _dump_signatures(DEV, API_MODULES)
        pub = _dump_signatures(PUB, API_MODULES)
    except Exception as e:
        banner("7.signatures", FAIL,
               "could not snapshot signatures from one of the repos",
               f"{type(e).__name__}: {e}")
        return

    fails: list[str] = []
    for m in API_MODULES:
        d, p = dev.get(m, {}), pub.get(m, {})
        if isinstance(d, dict) and "error" in d:
            continue
        if isinstance(p, dict) and "error" in p:
            continue
        for name in sorted(set(d) | set(p)):
            ds, ps = d.get(name), p.get(name)
            if ds is None or ps is None:
                # Missing names handled by Check 6
                continue
            if ds != ps:
                fails.append(f"  {m}.{name}:\n      dev: {ds}\n      pub: {ps}")

    if fails:
        banner("7.signatures", FAIL,
               f"{len(fails)} signature mismatch(es)",
               "\n".join(fails[:40]) +
               (f"\n... and {len(fails)-40} more" if len(fails) > 40 else ""))
    else:
        banner("7.signatures", PASS,
               f"all signatures match across {len(API_MODULES)} modules")


# --------------------------------------------------------------------------
# CHECKS 8, 9, 10, 11: Numerical / state-dict equivalence
# --------------------------------------------------------------------------

# Configs that don't need a downloaded LM — purely SAE-level instantiation.
NUM_CONFIGS = [
    # (name, builder)  — builder is Python source that constructs sae_cfg + sae
    (
        "topk_polysae_gpt2",
        textwrap.dedent("""
            import torch
            from sae_lens.saes.polysae import PolyTopKTrainingSAE, PolyTopKTrainingSAEConfig
            cfg = PolyTopKTrainingSAEConfig(
                d_in=768, d_sae=16384, k=64,
                poly_ranks=(768, 32, 32), poly_order=3, shared_u=True,
            )
            sae = PolyTopKTrainingSAE(cfg)
        """),
    ),
    (
        "topk_baseline_gpt2",
        textwrap.dedent("""
            import torch
            from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig
            cfg = TopKTrainingSAEConfig(d_in=768, d_sae=16384, k=64)
            sae = TopKTrainingSAE(cfg)
        """),
    ),
    (
        "batchtopk_polysae_gpt2",
        textwrap.dedent("""
            import torch
            from sae_lens.saes.polysae import PolyBatchTopKTrainingSAE, PolyBatchTopKTrainingSAEConfig
            cfg = PolyBatchTopKTrainingSAEConfig(
                d_in=768, d_sae=16384, k=64,
                poly_ranks=(768, 32, 32), poly_order=3, shared_u=True,
            )
            sae = PolyBatchTopKTrainingSAE(cfg)
        """),
    ),
    (
        "matryoshka_polysae_gpt2",
        textwrap.dedent("""
            import torch
            from sae_lens.saes.polysae import PolyMatryoshkaBatchTopKTrainingSAE, PolyMatryoshkaBatchTopKTrainingSAEConfig
            cfg = PolyMatryoshkaBatchTopKTrainingSAEConfig(
                d_in=768, d_sae=16384, k=64,
                poly_ranks=(768, 32, 32), poly_order=3, shared_u=True,
                matryoshka_widths=[512, 2048, 16384],
            )
            sae = PolyMatryoshkaBatchTopKTrainingSAE(cfg)
        """),
    ),
]


def _forward_pass_script(builder: str, out_path: Path) -> str:
    return textwrap.dedent(f"""
        import torch
        torch.manual_seed(0)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        {textwrap.indent(builder.strip(), '        ').lstrip()}
        sae.eval()
        x = torch.randn(4, sae.cfg.d_in)
        with torch.no_grad():
            try:
                out = sae(x)
            except Exception as e:
                # Some training SAEs require special call paths; fall back to encode+decode
                feats = sae.encode(x)
                out = sae.decode(feats)
        torch.save({{"out": out.detach().cpu(),
                    "n_params": sum(p.numel() for p in sae.parameters()),
                    "state_dict_keys": sorted(sae.state_dict().keys()),
                    "state_dict_shapes": {{k: list(v.shape) for k, v in sae.state_dict().items()}}}},
                   {str(out_path)!r})
    """)


def _one_step_script(builder: str, out_path: Path) -> str:
    return textwrap.dedent(f"""
        import torch
        torch.manual_seed(0)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        {textwrap.indent(builder.strip(), '        ').lstrip()}
        from sae_lens.saes.sae import TrainStepInput
        x = torch.randn(64, sae.cfg.d_in)
        opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
        opt.zero_grad()
        out = sae.training_forward_pass(
            step_input=TrainStepInput(
                sae_in=x,
                dead_neuron_mask=torch.zeros(sae.cfg.d_sae, dtype=torch.bool),
                coefficients={{}},
                n_training_steps=0,
            ),
        )
        loss = out.loss
        loss.backward()
        opt.step()
        sd = {{k: v.detach().cpu() for k, v in sae.state_dict().items()}}
        torch.save({{"state_dict": sd, "loss": float(loss.detach())}}, {str(out_path)!r})
    """)


def _run_in_repo(repo: Path, script: str, label: str) -> subprocess.CompletedProcess:
    return run_subprocess([sys.executable, "-c", script],
                          env=py_with_pythonpath(repo), timeout=300)


def _compare_tensors(a, b, label: str) -> tuple[str, str]:
    """Return (status, message) — PASS/WARN/FAIL."""
    import torch
    if a.shape != b.shape:
        return FAIL, f"{label}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"
    if a.dtype != b.dtype:
        return FAIL, f"{label}: dtype mismatch {a.dtype} vs {b.dtype}"
    if torch.equal(a, b):
        return PASS, f"{label}: bit-identical"
    max_abs = (a - b).abs().max().item()
    if torch.allclose(a, b, rtol=0, atol=1e-6):
        return WARN, f"{label}: not bit-identical but within atol=1e-6 (max_abs={max_abs:.3e})"
    return FAIL, f"{label}: differs beyond atol=1e-6 (max_abs={max_abs:.3e})"


def check_8_forward_pass() -> None:
    import torch
    import tempfile
    any_fail = False
    any_warn = False
    details: list[str] = []

    for name, builder in NUM_CONFIGS:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            dev_out = td / f"{name}_dev.pt"
            pub_out = td / f"{name}_pub.pt"
            dev_cp = _run_in_repo(DEV, _forward_pass_script(builder, dev_out), name)
            pub_cp = _run_in_repo(PUB, _forward_pass_script(builder, pub_out), name)
            if dev_cp.returncode != 0:
                any_fail = True
                details.append(f"  {name}: dev forward subprocess failed:\n{dev_cp.stderr.strip()}")
                continue
            if pub_cp.returncode != 0:
                any_fail = True
                details.append(f"  {name}: pub forward subprocess failed:\n{pub_cp.stderr.strip()}")
                continue
            d = torch.load(dev_out, weights_only=False)
            p = torch.load(pub_out, weights_only=False)
            status, msg = _compare_tensors(d["out"], p["out"], f"{name}.output")
            if status == FAIL:
                any_fail = True
            elif status == WARN:
                any_warn = True
            details.append(f"  {msg}")

    overall = FAIL if any_fail else (WARN if any_warn else PASS)
    banner("8.forward-pass", overall,
           f"{len(NUM_CONFIGS)} configs compared",
           "\n".join(details))


def check_9_one_optimizer_step() -> None:
    import torch
    import tempfile
    name, builder = NUM_CONFIGS[0]  # GPT-2 PolySAE config
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        dev_out = td / "dev.pt"
        pub_out = td / "pub.pt"
        dev_cp = _run_in_repo(DEV, _one_step_script(builder, dev_out), name)
        pub_cp = _run_in_repo(PUB, _one_step_script(builder, pub_out), name)
        if dev_cp.returncode != 0:
            banner("9.one-step-train", FAIL,
                   f"dev one-step subprocess failed",
                   dev_cp.stderr.strip())
            return
        if pub_cp.returncode != 0:
            banner("9.one-step-train", FAIL,
                   f"pub one-step subprocess failed",
                   pub_cp.stderr.strip())
            return
        d = torch.load(dev_out, weights_only=False)
        p = torch.load(pub_out, weights_only=False)
        d_sd, p_sd = d["state_dict"], p["state_dict"]

        # Loss check
        loss_diff = abs(d["loss"] - p["loss"])
        details = [f"  loss: dev={d['loss']:.6f}  pub={p['loss']:.6f}  diff={loss_diff:.3e}"]

        # State-dict key parity (also Check 11, but cheap to do here)
        if set(d_sd) != set(p_sd):
            banner("9.one-step-train", FAIL,
                   "state_dict keys differ after one step",
                   f"dev-only: {sorted(set(d_sd)-set(p_sd))}\n"
                   f"pub-only: {sorted(set(p_sd)-set(d_sd))}")
            return

        fails = []
        warns = []
        for k in sorted(d_sd):
            status, msg = _compare_tensors(d_sd[k], p_sd[k], k)
            if status == FAIL:
                fails.append(msg)
            elif status == WARN:
                warns.append(msg)

        details.append(f"  {len(d_sd)} tensors compared: "
                       f"identical={len(d_sd)-len(fails)-len(warns)} "
                       f"warn={len(warns)} fail={len(fails)}")
        if fails:
            details += ["  FAIL:"] + ["    " + m for m in fails[:8]]
            if len(fails) > 8:
                details.append(f"    ... and {len(fails)-8} more")
        if warns:
            details += ["  WARN:"] + ["    " + m for m in warns[:5]]
            if len(warns) > 5:
                details.append(f"    ... and {len(warns)-5} more")
        overall = FAIL if fails else (WARN if warns else PASS)
        banner("9.one-step-train", overall,
               f"loss-diff={loss_diff:.3e}, state-dict tensors compared: {len(d_sd)}",
               "\n".join(details))


# Paper configs (model, architecture, ranks) per README §8.
PAPER_PARAM_CONFIGS = [
    ("gpt2_l8_topk_poly",      "PolyTopKTrainingSAE",        dict(d_in=768,  d_sae=16384, k=64, poly_ranks=(768,  32,  32),  poly_order=3, shared_u=True)),
    ("pythia410m_l15_topk_poly","PolyTopKTrainingSAE",       dict(d_in=1024, d_sae=16384, k=64, poly_ranks=(1024, 128, 128), poly_order=3, shared_u=True)),
    ("pythia1_4b_l12_topk_poly","PolyTopKTrainingSAE",       dict(d_in=2048, d_sae=16384, k=64, poly_ranks=(2048, 128, 128), poly_order=3, shared_u=True)),
    ("gemma2_2b_l12_topk_poly", "PolyTopKTrainingSAE",       dict(d_in=2304, d_sae=16384, k=64, poly_ranks=(2304, 128, 128), poly_order=3, shared_u=True)),
]


def _param_count_script(cls_name: str, kwargs: dict, out_path: Path) -> str:
    cfg_cls = cls_name + "Config"
    return textwrap.dedent(f"""
        import torch, json
        from sae_lens.saes.polysae import {cls_name}, {cfg_cls}
        cfg = {cfg_cls}(**{kwargs!r})
        sae = {cls_name}(cfg)
        out = {{
            "n_params": sum(p.numel() for p in sae.parameters()),
            "state_dict_keys": sorted(sae.state_dict().keys()),
            "state_dict_shapes": {{k: list(v.shape) for k, v in sae.state_dict().items()}},
        }}
        with open({str(out_path)!r}, "w") as f:
            json.dump(out, f)
    """)


def check_10_param_count() -> None:
    import tempfile
    fails: list[str] = []
    summaries: list[str] = []
    for name, cls, kwargs in PAPER_PARAM_CONFIGS:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            dev_out = td / "dev.json"
            pub_out = td / "pub.json"
            dev_cp = _run_in_repo(DEV, _param_count_script(cls, kwargs, dev_out), name)
            pub_cp = _run_in_repo(PUB, _param_count_script(cls, kwargs, pub_out), name)
            if dev_cp.returncode != 0 or pub_cp.returncode != 0:
                fails.append(f"  {name}: subprocess failure  dev_rc={dev_cp.returncode} pub_rc={pub_cp.returncode}")
                if dev_cp.stderr:
                    fails.append("    dev stderr: " + dev_cp.stderr.strip().splitlines()[-1])
                if pub_cp.stderr:
                    fails.append("    pub stderr: " + pub_cp.stderr.strip().splitlines()[-1])
                continue
            d = json.loads(dev_out.read_text())
            p = json.loads(pub_out.read_text())
            if d["n_params"] != p["n_params"]:
                fails.append(f"  {name}: param count differs  dev={d['n_params']}  pub={p['n_params']}")
            else:
                summaries.append(f"  {name}: {d['n_params']:,} params (identical)")

    if fails:
        banner("10.param-count", FAIL,
               f"{len(fails)} config(s) with param-count mismatch",
               "\n".join(fails + summaries))
    else:
        banner("10.param-count", PASS,
               f"all {len(PAPER_PARAM_CONFIGS)} configs have identical param counts",
               "\n".join(summaries))


def check_11_state_dict_keys() -> None:
    """Per-key shape + name parity for the same paper configs."""
    import tempfile
    fails: list[str] = []
    for name, cls, kwargs in PAPER_PARAM_CONFIGS:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            dev_out = td / "dev.json"
            pub_out = td / "pub.json"
            dev_cp = _run_in_repo(DEV, _param_count_script(cls, kwargs, dev_out), name)
            pub_cp = _run_in_repo(PUB, _param_count_script(cls, kwargs, pub_out), name)
            if dev_cp.returncode != 0 or pub_cp.returncode != 0:
                # Already reported by Check 10; just note SKIP for parity check
                fails.append(f"  {name}: subprocess failed (see Check 10)")
                continue
            d = json.loads(dev_out.read_text())
            p = json.loads(pub_out.read_text())
            if d["state_dict_keys"] != p["state_dict_keys"]:
                only_d = sorted(set(d["state_dict_keys"]) - set(p["state_dict_keys"]))
                only_p = sorted(set(p["state_dict_keys"]) - set(d["state_dict_keys"]))
                fails.append(f"  {name}: state_dict keys differ  dev-only={only_d}  pub-only={only_p}")
                continue
            shape_diffs = []
            for k in d["state_dict_keys"]:
                if d["state_dict_shapes"][k] != p["state_dict_shapes"][k]:
                    shape_diffs.append(f"{k}: dev={d['state_dict_shapes'][k]} pub={p['state_dict_shapes'][k]}")
            if shape_diffs:
                fails.append(f"  {name}: {len(shape_diffs)} shape mismatch(es)")
                for sd in shape_diffs[:5]:
                    fails.append("    " + sd)

    if fails:
        banner("11.state-dict-keys", FAIL,
               f"state-dict parity broken for {sum(1 for f in fails if not f.startswith('    '))} config(s)",
               "\n".join(fails))
    else:
        banner("11.state-dict-keys", PASS,
               f"all {len(PAPER_PARAM_CONFIGS)} configs have identical state_dict keys + shapes")


# --------------------------------------------------------------------------
# CHECK 12: Wasserstein scoping
# --------------------------------------------------------------------------

def check_12_wasserstein() -> None:
    text_exts = {".py"}
    files = [p for p in walk_files(PUB) if p.suffix in text_exts]
    hits: list[tuple[Path, int, str]] = []
    for p in files:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if "wasserstein" in line.lower():
                hits.append((p, lineno, line.rstrip()))

    code_files = sorted({p for p, _, _ in hits if "SAEBench/" not in str(p)})

    # Find a real sparsity-k loop that touches wasserstein at k>1. We look
    # specifically for `for k in k_values:` (the loop pattern used in
    # train_and_saebench.py) — not list comprehensions like
    # `[k for k in some_dict.keys()]` which use `k` as a string key.
    issues: list[str] = []
    has_k1 = False
    SPARSITY_LOOP = re.compile(r"^\s*for\s+k\s+in\s+k_values\s*:")
    for fname in ["train_and_saebench.py", "eval_sae.py"]:
        f = PUB / fname
        if not f.exists():
            continue
        text = f.read_text()
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if "wasserstein" not in line.lower():
                continue
            # Skip lines that are themselves list comprehensions / dict-key filters
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if re.search(r"\bfor\s+\w+\s+in\s+\w+\.(keys|items|values)\(\)", line):
                continue  # list comprehension over dict view, not sparsity loop
            window_lines = lines[max(0, i-15):i+1]
            in_sparsity_loop = any(SPARSITY_LOOP.match(wl) for wl in window_lines)
            guarded_to_1 = bool(re.search(r"if\s+k\s*==\s*1", "\n".join(window_lines)))
            if in_sparsity_loop and not guarded_to_1:
                issues.append(
                    f"{rel(f, PUB)}:{i+1}: wasserstein inside sparsity-k loop "
                    f"without `if k == 1` guard:\n      {stripped}"
                )
            if "sae_top_1_wasserstein" in line or "k == 1" in "\n".join(window_lines):
                has_k1 = True

    # SAEBench eval-output schemas can keep llm_top_K_wasserstein fields — those
    # are LLM-stream metrics, not SAE-Wasserstein-at-k. Those are not flagged.

    msg_parts = []
    if has_k1:
        msg_parts.append("k=1 computation reachable")
    if issues:
        banner("12.wasserstein", FAIL,
               f"{len(issues)} unguarded wasserstein-in-k-loop site(s)",
               "\n".join(issues))
    else:
        banner("12.wasserstein", PASS,
               f"wasserstein restricted to k=1; "
               f"{len(code_files)} non-SAEBench files reference 'wasserstein'",
               "\n".join(["  " + str(f.relative_to(PUB)) for f in code_files]))


# --------------------------------------------------------------------------
# CHECK 13: WandB opt-in behavior
# --------------------------------------------------------------------------

def check_13_wandb_optin() -> None:
    # 13a: grep wandb.init( call-sites and assert each is guarded
    fails: list[str] = []
    text_files = [p for p in walk_files(PUB) if p.suffix == ".py"]
    wandb_init_sites: list[tuple[Path, int, str]] = []
    for p in text_files:
        rp = rel(p, PUB)
        # Exclude self (the audit script mentions wandb.init in strings/comments)
        # and upstream-SAELens test fixtures.
        if rp == "tests/audit_against_dev.py":
            continue
        if rp.startswith("tests/_comparison/"):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines()):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue  # comment, not a call
            # Match real call sites: wandb.init( only, not bare references in docstrings.
            if re.search(r"\bwandb\.init\s*\(", line):
                wandb_init_sites.append((p, i+1, line.strip()))

    # Each call must be guarded by a check on log_to_wandb / --wandb / WANDB_ENTITY
    # within the surrounding 5 lines
    GUARDED_PAT = re.compile(r"(log_to_wandb|args\.wandb\b|args\.wandb_project|WANDB_ENTITY|WANDB_PROJECT|wandb_enabled)")
    unguarded: list[str] = []
    for p, lineno, line in wandb_init_sites:
        text = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        window = "\n".join(text[max(0, lineno-6):lineno])
        if not GUARDED_PAT.search(window):
            unguarded.append(f"  {rel(p, PUB)}:{lineno}: {line}")

    # 13b: import resolve_wandb (if available) and assert
    #   resolve_wandb(args with wandb=False, wandb_run_name set, project/entity None) → (False, None, None)
    code = textwrap.dedent(f"""
        import sys, os, importlib.util, argparse
        sys.path.insert(0, {str(PUB)!r})
        # Unset wandb env vars so the helper sees an opt-in-off state.
        for k in ("WANDB_PROJECT", "WANDB_ENTITY", "WANDB_MODE"):
            os.environ.pop(k, None)
        spec = importlib.util.spec_from_file_location("train_and_saebench",
                                                      {str(PUB / 'train_and_saebench.py')!r})
        mod = importlib.util.module_from_spec(spec)
        sys.modules["train_and_saebench"] = mod  # @dataclass needs this
        spec.loader.exec_module(mod)
        ns = argparse.Namespace(
            wandb=False, wandb_project=None, wandb_entity=None,
            wandb_run_name="foo",
        )
        if not hasattr(mod, "resolve_wandb"):
            print("MISSING")
        else:
            print(mod.resolve_wandb(ns))
    """)
    cp = run_subprocess([sys.executable, "-c", code],
                        env=py_with_pythonpath(PUB), timeout=60)
    sub_msg = cp.stdout.strip().splitlines()[-1] if cp.returncode == 0 else f"subprocess error: {cp.stderr}"

    behavior_fail = False
    if "MISSING" in sub_msg:
        behavior_fail = True
        sub_detail = "resolve_wandb() helper missing from train_and_saebench.py"
    elif "False" not in sub_msg or "None" not in sub_msg:
        behavior_fail = True
        sub_detail = f"resolve_wandb returned {sub_msg!r} — expected (False, None, None)"
    else:
        sub_detail = f"resolve_wandb(--wandb_run_name foo, no --wandb) = {sub_msg}"

    overall_fail = bool(unguarded) or behavior_fail
    detail_lines = [f"  wandb.init sites: {len(wandb_init_sites)}, unguarded: {len(unguarded)}"]
    if unguarded:
        detail_lines += unguarded
    detail_lines.append(f"  resolve_wandb behavior: {sub_detail}")
    if overall_fail:
        banner("13.wandb-optin", FAIL,
               "wandb-opt-in not fully enforced",
               "\n".join(detail_lines))
    else:
        banner("13.wandb-optin", PASS,
               f"all {len(wandb_init_sites)} wandb.init sites guarded; opt-in helper behaves correctly",
               "\n".join(detail_lines))


# --------------------------------------------------------------------------
# CHECK 14: pytest the repo's own tests
# --------------------------------------------------------------------------

def check_14_pytest() -> None:
    """Run the three test files the agent added per §6 of the release plan.

    Upstream SAELens tests (the inherited tests/saes/*, tests/loading/*, etc.)
    have a chain of optional dev dependencies (sparsify, dictionary_learning,
    mamba_lens, ...) that aren't part of the standard `pip install -e .` flow.
    Verifying *their* full pass is out of scope for this audit; Check 14
    measures only what the agent owns.
    """
    agent_tests = [
        "tests/test_polysae_imports.py",
        "tests/test_polysae_cli.py",
        "tests/saes/test_polysae_smoke.py",
    ]
    cp = run_subprocess(
        [sys.executable, "-m", "pytest", *agent_tests, "-q", "-x",
         "-p", "no:cacheprovider"],
        cwd=PUB, timeout=600,
    )
    tail = "\n".join((cp.stdout + cp.stderr).strip().splitlines()[-30:])
    if cp.returncode != 0:
        banner("14.pytest", FAIL,
               f"pytest exited with code {cp.returncode} on agent-owned tests",
               tail)
    else:
        banner("14.pytest", PASS, "agent's three new test files all pass", tail)


# --------------------------------------------------------------------------
# SKIP placeholders (documented for the GPU box)
# --------------------------------------------------------------------------

def skip_gpu_only() -> None:
    banner("S1.gpu-training-smoke", SKIP,
           "Full training smoke on the 8 paper commands needs a GPU; run on the GPU box.")
    banner("S2.large-LM-forward", SKIP,
           "Pythia-1.4B / Gemma-2-2B SAE-attached forward passes need to download LM checkpoints (>RAM); GPU-box only.")
    banner("S3.saebench-eval", SKIP,
           "SAEBench evaluation on a trained checkpoint needs trained weights; GPU-box only.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

CHECKS = [
    check_1_file_tree,
    check_2_scrubs,
    check_3_install,
    check_4_compileall,
    check_5_cli_parity,
    check_6_api_parity,
    check_7_signatures,
    check_8_forward_pass,
    check_9_one_optimizer_step,
    check_10_param_count,
    check_11_state_dict_keys,
    check_12_wasserstein,
    check_13_wandb_optin,
    check_14_pytest,
]


def main() -> int:
    print("=" * 76)
    print(f"PolySAE release-readiness audit")
    print(f"  dev: {DEV}")
    print(f"  pub: {PUB}")
    print(f"  python: {sys.executable}")
    print("=" * 76)

    for fn in CHECKS:
        try:
            fn()
        except Exception:
            banner(fn.__name__, FAIL,
                   "uncaught exception in audit check",
                   traceback.format_exc())

    skip_gpu_only()

    counts = {PASS: 0, FAIL: 0, WARN: 0, SKIP: 0}
    for _, status, _, _ in results:
        counts[status] += 1

    print("\n" + "=" * 76)
    print(f"SUMMARY: pass={counts[PASS]} fail={counts[FAIL]} "
          f"warn={counts[WARN]} skip={counts[SKIP]}")
    print("=" * 76)

    if counts[FAIL]:
        print("\nFAILURES:")
        for check, status, msg, details in results:
            if status == FAIL:
                print(f"  [{check}] {msg}")
                for line in details.splitlines()[:8]:
                    print(f"        {line}")
    if counts[WARN]:
        print("\nWARNINGS (human review needed):")
        for check, status, msg, details in results:
            if status == WARN:
                print(f"  [{check}] {msg}")

    return 1 if counts[FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
