"""Verification per §6.2 and §6.6 of the public-repo extract spec.

(2) imports every package module to catch missing deps / circular imports.
(6) asserts the bundled SAELens and SAEBench resolve *inside* this repo,
    not under site-packages from PyPI.
"""

from __future__ import annotations

import importlib
import pathlib
import pkgutil

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _module_path(module) -> pathlib.Path:
    """Return the on-disk path of an imported module."""
    file = getattr(module, "__file__", None)
    if file is None:
        # Namespace package — use first __path__ entry
        return pathlib.Path(next(iter(module.__path__))).resolve()
    return pathlib.Path(file).resolve()


def test_sae_lens_resolves_inside_repo() -> None:
    import sae_lens

    p = _module_path(sae_lens)
    assert REPO_ROOT in p.parents, (
        f"sae_lens resolved to {p}, expected to live under {REPO_ROOT}. "
        "Make sure you ran `pip install -e .` from the PolySAE root, not from PyPI."
    )


def test_sae_bench_resolves_inside_repo() -> None:
    import sae_bench

    p = _module_path(sae_bench)
    assert REPO_ROOT in p.parents, (
        f"sae_bench resolved to {p}, expected to live under {REPO_ROOT}/SAEBench. "
        "Make sure you ran `pip install -e ./SAEBench` from the PolySAE root, not from PyPI."
    )


def test_polysae_module_imports() -> None:
    """Importing sae_lens.saes.polysae must succeed; all Poly* names must export."""
    module = importlib.import_module("sae_lens.saes.polysae")
    for name in [
        "PolyTopKSAE",
        "PolyTopKSAEConfig",
        "PolyTopKTrainingSAE",
        "PolyTopKTrainingSAEConfig",
        "PolyBatchTopKTrainingSAE",
        "PolyBatchTopKTrainingSAEConfig",
        "PolyJumpReLUTrainingSAE",
        "PolyJumpReLUTrainingSAEConfig",
        "PolyMatryoshkaBatchTopKTrainingSAE",
        "PolyMatryoshkaBatchTopKTrainingSAEConfig",
    ]:
        assert hasattr(module, name), f"sae_lens.saes.polysae missing export: {name}"


def test_polysae_classes_exported_from_top_level() -> None:
    import sae_lens

    for name in [
        "PolyTopKSAE",
        "PolyTopKTrainingSAE",
        "PolyTopKTrainingSAEConfig",
        "PolyBatchTopKTrainingSAE",
        "PolyJumpReLUTrainingSAE",
        "PolyMatryoshkaBatchTopKTrainingSAE",
    ]:
        assert hasattr(sae_lens, name), f"sae_lens top-level missing re-export: {name}"


@pytest.mark.parametrize(
    "modname",
    [
        "sae_lens",
        "sae_lens.config",
        "sae_lens.registry",
        "sae_lens.llm_sae_training_runner",
        "sae_lens.saes.sae",
        "sae_lens.saes.topk_sae",
        "sae_lens.saes.batchtopk_sae",
        "sae_lens.saes.jumprelu_sae",
        "sae_lens.saes.matryoshka_batchtopk_sae",
        "sae_lens.saes.polysae",
        "sae_lens.training.sae_trainer",
        "sae_lens.training.activations_store",
    ],
)
def test_smoke_import(modname: str) -> None:
    importlib.import_module(modname)
