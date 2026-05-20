"""Smoke tests for PolySAE per §6.4 and §6.5 of the extract spec.

(4) Instantiates a PolySAE for each paper (model, architecture, rank) config
    and asserts overhead is in a sane band over the matched baseline SAE.
(5) Runs one forward pass on random tensors, asserting shape and finiteness.

These run on CPU and finish in a few seconds. A heavier 100-step loss-decrease
check is gated behind RUN_HEAVY=1 (off by default).
"""

from __future__ import annotations

import os

import pytest
import torch

from sae_lens.saes.polysae import (
    PolyBatchTopKTrainingSAE,
    PolyBatchTopKTrainingSAEConfig,
    PolyJumpReLUTrainingSAE,
    PolyJumpReLUTrainingSAEConfig,
    PolyMatryoshkaBatchTopKTrainingSAE,
    PolyMatryoshkaBatchTopKTrainingSAEConfig,
    PolyTopKTrainingSAE,
    PolyTopKTrainingSAEConfig,
)
from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig


# (d_in, d_sae, poly_ranks) tuples — paper configurations from README §8.
PAPER_CONFIGS = [
    pytest.param(768, 16384, (768, 32, 32), id="gpt2_l8"),
    pytest.param(1024, 16384, (1024, 128, 128), id="pythia410m_l15"),
    pytest.param(2048, 16384, (2048, 128, 128), id="pythia1_4b_l12"),
    pytest.param(2304, 16384, (2304, 128, 128), id="gemma2_2b_l12"),
]


def _count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


@pytest.mark.parametrize("d_in,d_sae,poly_ranks", PAPER_CONFIGS)
def test_polysae_topk_param_count_in_band(
    d_in: int, d_sae: int, poly_ranks: tuple[int, int, int]
) -> None:
    """PolySAE must add some — but not implausibly many — params over base TopK."""
    base_cfg = TopKTrainingSAEConfig(d_in=d_in, d_sae=d_sae, k=64)
    base = TopKTrainingSAE(base_cfg)
    base_params = _count_params(base)

    poly_cfg = PolyTopKTrainingSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        k=64,
        poly_ranks=poly_ranks,
        poly_order=3,
        shared_u=True,
    )
    poly = PolyTopKTrainingSAE(poly_cfg)
    poly_params = _count_params(poly)

    assert poly_params > base_params, (
        f"PolySAE has fewer params than base ({poly_params} ≤ {base_params}) — "
        "polynomial decoder weights are not being registered."
    )
    # A loose upper bound: catches accidental rank explosions or duplicate
    # parameter registration. The actual paper-reported overhead is ~3% of
    # the matched-rank baseline, but the inherited W_enc/W_dec from the base
    # SAE class make the on-disk ratio larger — assert it isn't pathological.
    assert poly_params < 3 * base_params, (
        f"PolySAE has implausibly many params: {poly_params} vs base {base_params}"
    )

    # Polynomial-specific params must exist and be finite.
    assert hasattr(poly, "lambda2"), "PolySAE missing λ₂"
    assert hasattr(poly, "lambda3"), "PolySAE missing λ₃"
    assert torch.isfinite(poly.lambda2).all()
    assert torch.isfinite(poly.lambda3).all()


@pytest.mark.parametrize("d_in,d_sae,poly_ranks", PAPER_CONFIGS)
def test_polysae_topk_forward_pass_is_finite(
    d_in: int, d_sae: int, poly_ranks: tuple[int, int, int]
) -> None:
    poly_cfg = PolyTopKTrainingSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        k=64,
        poly_ranks=poly_ranks,
        poly_order=3,
        shared_u=True,
    )
    poly = PolyTopKTrainingSAE(poly_cfg)

    torch.manual_seed(0)
    x = torch.randn(8, d_in)
    poly.eval()
    with torch.no_grad():
        out = poly(x)

    assert out.shape == x.shape, f"Output shape {out.shape} != input shape {x.shape}"
    assert torch.isfinite(out).all(), "PolySAE forward produced non-finite values"


@pytest.mark.parametrize(
    "cfg_cls,sae_cls,extra_kwargs",
    [
        (PolyBatchTopKTrainingSAEConfig, PolyBatchTopKTrainingSAE, {"k": 64}),
        (PolyJumpReLUTrainingSAEConfig, PolyJumpReLUTrainingSAE, {}),
        (
            PolyMatryoshkaBatchTopKTrainingSAEConfig,
            PolyMatryoshkaBatchTopKTrainingSAE,
            # Matryoshka validator requires matryoshka_widths[-1] == d_sae;
            # a single-level matryoshka is the minimal valid configuration.
            {"k": 64, "matryoshka_widths": [512]},
        ),
    ],
)
def test_polysae_other_architectures_instantiate(
    cfg_cls, sae_cls, extra_kwargs
) -> None:
    """The polynomial mixin must compose with each base activation scheme."""
    d_in, d_sae = 128, 512
    cfg = cfg_cls(
        d_in=d_in,
        d_sae=d_sae,
        poly_ranks=(128, 16, 16),
        poly_order=3,
        shared_u=True,
        **extra_kwargs,
    )
    sae = sae_cls(cfg)
    assert _count_params(sae) > 0

    torch.manual_seed(0)
    x = torch.randn(4, d_in)
    sae.eval()
    with torch.no_grad():
        out = sae(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@pytest.mark.skipif(
    os.environ.get("RUN_HEAVY") != "1",
    reason="Heavy training-step check is gated behind RUN_HEAVY=1",
)
def test_polysae_loss_decreases_over_100_steps() -> None:
    """Optional: 100 SGD steps on random data should reduce reconstruction loss."""
    from sae_lens.saes.sae import TrainStepInput

    torch.manual_seed(0)
    d_in, d_sae = 64, 256
    cfg = PolyTopKTrainingSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        k=16,
        poly_ranks=(64, 16, 16),
        poly_order=3,
        shared_u=True,
    )
    sae = PolyTopKTrainingSAE(cfg)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    x = torch.randn(128, d_in)
    losses: list[float] = []
    for step in range(100):
        opt.zero_grad()
        out = sae.training_forward_pass(
            step_input=TrainStepInput(
                sae_in=x,
                dead_neuron_mask=torch.zeros(d_sae, dtype=torch.bool),
                coefficients={},
                n_training_steps=step,
            )
        )
        loss = out.loss
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0], (
        f"Loss did not decrease over 100 steps: {losses[0]:.3f} → {losses[-1]:.3f}"
    )
