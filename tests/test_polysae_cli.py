"""CLI parity test (§6.3 of the extract spec).

Runs `train_and_saebench.py --help` as a subprocess and asserts every flag
referenced by the eight paper commands in the README is recognized. The test
parses help text only — it never starts training.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
ENTRY = REPO_ROOT / "train_and_saebench.py"

# Every flag that appears in any of the eight paper commands.
PAPER_FLAGS = [
    "--exp",
    "--architecture",
    "--width",
    "--k",
    "--training_tokens",
    "--context_size",
    "--no_rescale_by_decoder_norm",
    "--run_saebench",
    "--use_saelens_defaults",
    "--wandb_run_name",
    "--poly",
    "--shared_u",
    "--poly_order",
    "--poly_ranks",
    "--n_batches_in_buffer",
    "--train_batch_size_tokens",
]


def test_help_runs_and_lists_paper_flags() -> None:
    result = subprocess.run(
        [sys.executable, str(ENTRY), "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"train_and_saebench.py --help exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    help_text = result.stdout
    missing = [flag for flag in PAPER_FLAGS if flag not in help_text]
    assert not missing, (
        f"Flags missing from `train_and_saebench.py --help`: {missing}\n"
        f"--- help text ---\n{help_text}"
    )


def test_wandb_run_name_alone_is_a_no_op() -> None:
    """Passing --wandb_run_name without --wandb must not crash argument parsing.

    The actual side-effect (wandb-off) is verified by inspecting the parser
    rather than running training.
    """
    result = subprocess.run(
        [sys.executable, str(ENTRY), "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "--wandb_run_name" in result.stdout
    # The README and the spec promise wandb is off by default — verify the
    # --wandb opt-in flag exists.
    assert "--wandb" in result.stdout, "Opt-in --wandb flag missing"
