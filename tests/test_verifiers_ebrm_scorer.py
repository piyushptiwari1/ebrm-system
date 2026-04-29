"""Tests for the EBRM-as-verifier scorer.

The full ``EBRMScorer.from_pretrained`` path requires a 4 GB Qwen3-4B
download and a CUDA box; we don't exercise it in CI. These tests cover the
encoder-free subsystem: vendored architecture matches the upstream training
state-dict layout, and ``score_batch`` returns a finite scalar per
candidate when given hand-built tiny weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from ebrm_system.verifiers._ebrm_arch import (  # noqa: E402
    CrossAttentionEnergy,
    GatedProjector,
    WeightedPooler,
)
from ebrm_system.verifiers.ebrm_scorer import (  # noqa: E402
    EBRMScorer,
    EBRMSelection,
    _load_subset,
)

HIDDEN_DIM = 16
LATENT_DIM = 8


def test_pooler_shape() -> None:
    pooler = WeightedPooler(HIDDEN_DIM, num_heads=4)
    hidden = torch.randn(2, 7, HIDDEN_DIM)
    mask = torch.ones(2, 7, dtype=torch.long)
    out = pooler(hidden, mask)
    assert out.shape == (2, HIDDEN_DIM)


def test_projector_shape() -> None:
    proj = GatedProjector(HIDDEN_DIM, LATENT_DIM)
    out = proj(torch.randn(3, HIDDEN_DIM))
    assert out.shape == (3, LATENT_DIM)


def test_energy_head_returns_scalar_per_pair() -> None:
    head = CrossAttentionEnergy(LATENT_DIM, num_heads=2, hidden_dim=16)
    sol = torch.randn(4, LATENT_DIM)
    prob = torch.randn(4, LATENT_DIM)
    e = head(sol, prob)
    assert e.shape == (4,)
    assert torch.isfinite(e).all()


def test_load_subset_strips_prefix_and_loads() -> None:
    pooler = WeightedPooler(HIDDEN_DIM, num_heads=4)
    state = {f"pooler.{k}": v for k, v in pooler.state_dict().items()}
    state["energy_head.foo"] = torch.zeros(1)  # noise; should be ignored
    fresh = WeightedPooler(HIDDEN_DIM, num_heads=4)
    _load_subset(fresh, state, prefix="pooler.")
    for k in pooler.state_dict():
        assert torch.equal(fresh.state_dict()[k], pooler.state_dict()[k])


def test_load_subset_raises_on_missing_prefix() -> None:
    pooler = WeightedPooler(HIDDEN_DIM, num_heads=4)
    with pytest.raises(RuntimeError, match="No keys with prefix"):
        _load_subset(pooler, {"other.weight": torch.zeros(1)}, prefix="pooler.")


# ---------------------------------------------------------------- end-to-end


@dataclass
class _FakeTokenized:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor

    def __getitem__(self, k: str) -> torch.Tensor:
        return getattr(self, k)


class _FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(
        self,
        texts: list[str] | str,
        max_length: int = 8,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> _FakeTokenized:
        if isinstance(texts, str):
            texts = [texts]
        # Deterministic faux-tokenization: hash chars into the vocab range.
        ids = torch.tensor(
            [[(ord(c) % 100) + 1 for c in (t.ljust(max_length))[:max_length]] for t in texts],
            dtype=torch.long,
        )
        mask = torch.ones_like(ids)
        return _FakeTokenized(input_ids=ids, attention_mask=mask)


class _FakeOutputs:
    def __init__(self, hidden: torch.Tensor) -> None:
        self.hidden_states = (hidden,)


class _FakeEncoder(torch.nn.Module):
    """Minimal stand-in for AutoModelForCausalLM.

    Returns a single hidden-state tensor of shape (batch, seq, hidden_dim).
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(200, hidden_dim)

        class _Cfg:
            pass

        cfg = _Cfg()
        cfg.hidden_size = hidden_dim  # type: ignore[attr-defined]
        self.config = cfg

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = True,
    ) -> _FakeOutputs:
        return _FakeOutputs(self.embed(input_ids))


def _build_scorer() -> EBRMScorer:
    torch.manual_seed(0)
    encoder = _FakeEncoder(HIDDEN_DIM)
    pooler = WeightedPooler(HIDDEN_DIM, num_heads=4)
    projector = GatedProjector(HIDDEN_DIM, LATENT_DIM)
    energy_head = CrossAttentionEnergy(LATENT_DIM, num_heads=2, hidden_dim=16)
    return EBRMScorer(
        base_encoder=encoder,
        tokenizer=_FakeTokenizer(),
        pooler=pooler,
        projector=projector,
        energy_head=energy_head,
        max_length=8,
        device="cpu",
    )


def test_score_batch_returns_one_finite_per_candidate() -> None:
    scorer = _build_scorer()
    energies = scorer.score_batch(question="solve x", candidates=["x = 1", "x = 2", "x = 3"])
    assert len(energies) == 3
    assert all(isinstance(e, float) for e in energies)
    assert all(e == e and abs(e) < 1e6 for e in energies)  # finite (no NaN/inf)


def test_score_single_matches_score_batch() -> None:
    scorer = _build_scorer()
    e_single = scorer.score("q", "a")
    e_batch = scorer.score_batch("q", ["a"])[0]
    assert e_single == pytest.approx(e_batch)


def test_select_best_picks_argmin() -> None:
    scorer = _build_scorer()
    selection = scorer.select_best("q", ["a", "b", "c"])
    assert isinstance(selection, EBRMSelection)
    assert selection.energy == min(selection.all_energies)
    assert selection.candidate == ["a", "b", "c"][selection.index]
    assert len(selection.all_energies) == 3


def test_select_best_empty_raises() -> None:
    scorer = _build_scorer()
    with pytest.raises(ValueError, match="at least one candidate"):
        scorer.select_best("q", [])


def test_score_batch_empty_returns_empty() -> None:
    scorer = _build_scorer()
    assert scorer.score_batch("q", []) == []
