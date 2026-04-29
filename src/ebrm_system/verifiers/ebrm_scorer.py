"""EBRM-as-verifier: score (question, candidate_solution) → energy.

Loads only the trainable heads (pooler + projector + energy_head) from a
HuggingFace checkpoint and pairs them with a frozen base encoder. Lower
energy = better candidate. Used to re-rank N candidate solutions sampled
from any LLM (gpt-4o-mini, Qwen, vLLM, …) without requiring the full
Langevin inference loop.

Default checkpoint: ``piyushptiwari/ebrm-v2-qwen3-4b`` (math-only training:
GSM8K + MATH + 5k synthetic). A future ``ebrm-v3-general`` checkpoint
(BBH + StrategyQA + LogiQA + …) will be a drop-in replacement.

Example
-------
>>> scorer = EBRMScorer.from_pretrained()  # doctest: +SKIP
>>> energies = scorer.score_batch(
...     question="If 3x + 5 = 20, what is x?",
...     candidates=["x = 5", "x = 15/3 = 5", "x = 25/3"],
... )  # doctest: +SKIP
>>> best_idx = scorer.select_best(question, candidates).index  # doctest: +SKIP
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

DEFAULT_REPO_ID = "piyushptiwari/ebrm-v2-qwen3-4b"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B"
DEFAULT_LATENT_DIM = 768
DEFAULT_MAX_LEN = 512
DEFAULT_CHECKPOINT_FILE = "ebrm_inference.pt"


@dataclass(frozen=True)
class EBRMSelection:
    """Result of ranking N candidates by energy."""

    index: int
    candidate: str
    energy: float
    all_energies: tuple[float, ...]


class EBRMScorer:
    """Energy-based re-ranker for candidate reasoning chains.

    Parameters
    ----------
    base_encoder
        A loaded ``transformers`` causal LM whose final hidden states feed the
        pooler. Should be the same architecture the heads were trained on.
    tokenizer
        Tokenizer paired with ``base_encoder``.
    pooler, projector, energy_head
        Trained heads from the upstream EBRM checkpoint.
    max_length
        Token cap for both question and candidate. The training script used
        512; longer caps will degrade scoring quality unless the heads are
        retrained with the higher cap.
    device
        Device for the heads. Encoder placement follows ``device_map="auto"``.
    """

    def __init__(
        self,
        base_encoder: object,
        tokenizer: object,
        pooler: object,
        projector: object,
        energy_head: object,
        *,
        max_length: int = DEFAULT_MAX_LEN,
        device: str | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "EBRMScorer requires torch. Install with: pip install 'ebrm-system[model]'"
            ) from exc

        self.encoder = base_encoder
        self.tokenizer = tokenizer
        self.pooler = pooler
        self.projector = projector
        self.energy_head = energy_head
        self.max_length = max_length
        self.device = (
            torch.device(device)
            if device is not None
            else next(iter(pooler.parameters())).device  # type: ignore[attr-defined]
        )
        # Switch to eval mode so dropout/spectral-norm power-iters are stable.
        for module in (self.encoder, self.pooler, self.projector, self.energy_head):
            module.eval()  # type: ignore[attr-defined]

    # --------------------------------------------------------------- factory

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = DEFAULT_REPO_ID,
        *,
        base_model: str = DEFAULT_BASE_MODEL,
        latent_dim: int = DEFAULT_LATENT_DIM,
        max_length: int = DEFAULT_MAX_LEN,
        load_in_4bit: bool = True,
        device: str | None = None,
        cache_dir: str | Path | None = None,
        checkpoint_file: str = DEFAULT_CHECKPOINT_FILE,
    ) -> EBRMScorer:
        """Download a HuggingFace checkpoint and assemble a scorer.

        Requires the optional ``model`` extra (``torch``, ``transformers``)
        plus ``huggingface_hub`` (transitive dep of ``transformers``) and,
        when ``load_in_4bit=True``, ``bitsandbytes``.
        """
        try:
            import torch
            from huggingface_hub import hf_hub_download
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "EBRMScorer.from_pretrained requires the 'model' extra: "
                "pip install 'ebrm-system[model]' huggingface_hub bitsandbytes"
            ) from exc

        from ebrm_system.verifiers._ebrm_arch import (
            CrossAttentionEnergy,
            GatedProjector,
            WeightedPooler,
        )

        ckpt_path = hf_hub_download(
            repo_id=repo_id, filename=checkpoint_file, cache_dir=str(cache_dir) if cache_dir else None
        )

        quant_config = None
        if load_in_4bit:
            try:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            except Exception:  # pragma: no cover - bitsandbytes optional
                quant_config = None

        try:
            encoder = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=quant_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                attn_implementation="sdpa",
            )
        except Exception:
            encoder = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        for p in encoder.parameters():
            p.requires_grad = False

        hidden_dim = encoder.config.hidden_size
        target_device = (
            torch.device(device) if device is not None else next(encoder.parameters()).device
        )

        pooler = WeightedPooler(hidden_dim, num_heads=4).to(target_device)
        projector = GatedProjector(hidden_dim, latent_dim).to(target_device)
        energy_head = CrossAttentionEnergy(latent_dim).to(target_device)

        state = torch.load(ckpt_path, map_location=target_device, weights_only=False)
        # Checkpoint may be the raw state_dict or wrapped under "model" / "ema".
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model" in state and "pooler.combine.weight" not in state:
            state = state["model"]

        _load_subset(pooler, state, prefix="pooler.")
        _load_subset(projector, state, prefix="projector.")
        _load_subset(energy_head, state, prefix="energy_head.")

        return cls(
            base_encoder=encoder,
            tokenizer=tokenizer,
            pooler=pooler,
            projector=projector,
            energy_head=energy_head,
            max_length=max_length,
            device=str(target_device),
        )

    # --------------------------------------------------------------- scoring

    def _encode(self, texts: list[str]) -> torch.Tensor:
        import torch

        enc = self.tokenizer(  # type: ignore[operator]
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        device = self.device
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            outputs = self.encoder(  # type: ignore[operator]
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-1].float()
            pooled = self.pooler(hidden, attention_mask)  # type: ignore[operator]
            latent: torch.Tensor = self.projector(pooled)  # type: ignore[operator]
        return latent

    def score(self, question: str, candidate: str) -> float:
        """Return scalar energy for a single (question, candidate). Lower = better."""
        return self.score_batch(question, [candidate])[0]

    def score_batch(self, question: str, candidates: list[str]) -> list[float]:
        """Score N candidates against a single question. Lower = better."""
        if not candidates:
            return []
        import torch

        # Format candidates the way training did:
        # f"{question}\n\nSolution: {steps_and_final_answer}"
        # Callers may already have done this; we don't re-wrap if the candidate
        # already starts with the question text.
        formatted = [
            c if c.lstrip().startswith(question.strip()[:40]) else f"{question}\n\nSolution: {c}"
            for c in candidates
        ]

        problem_state = self._encode([question])  # (1, D)
        candidate_states = self._encode(formatted)  # (N, D)
        problem_expanded = problem_state.expand(candidate_states.shape[0], -1)
        with torch.no_grad():
            energies = self.energy_head(candidate_states, problem_expanded)  # type: ignore[operator]
        out: list[float] = energies.detach().cpu().tolist()
        return out

    def select_best(self, question: str, candidates: list[str]) -> EBRMSelection:
        """Return the lowest-energy candidate."""
        if not candidates:
            raise ValueError("select_best() requires at least one candidate")
        energies = self.score_batch(question, candidates)
        best_idx = min(range(len(energies)), key=energies.__getitem__)
        return EBRMSelection(
            index=best_idx,
            candidate=candidates[best_idx],
            energy=energies[best_idx],
            all_energies=tuple(energies),
        )


# --------------------------------------------------------------- utilities


def _load_subset(module: object, full_state: dict[str, object], *, prefix: str) -> None:
    """Load only the keys under ``prefix`` from ``full_state`` into ``module``.

    Strips the prefix and uses ``strict=True`` so a key mismatch (i.e. arch
    drift between this vendored copy and the trained checkpoint) raises
    immediately rather than silently scoring on random weights.
    """
    sub = {
        k[len(prefix) :]: v
        for k, v in full_state.items()
        if k.startswith(prefix)
    }
    if not sub:
        raise RuntimeError(
            f"No keys with prefix '{prefix}' found in checkpoint. "
            f"Available top-level prefixes: "
            f"{sorted({k.split('.', 1)[0] for k in full_state})}"
        )
    missing, unexpected = module.load_state_dict(sub, strict=False)  # type: ignore[attr-defined]
    # Spectral-norm modules carry derived buffers (.weight) alongside the
    # parameter (.weight_orig); these are recomputed on first forward, so
    # missing-on-load is fine. Anything else is a real mismatch.
    real_missing = [k for k in missing if not k.endswith(".weight")]
    if real_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint architecture mismatch for '{prefix}': "
            f"missing={real_missing}, unexpected={unexpected}"
        )
