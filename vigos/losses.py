"""Distribution losses used by ViGOS."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def kl_tokens(
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    temperature: float = 1.0,
    token_clip: float | None = None,
) -> torch.Tensor:
    """Compute full-vocabulary token-level KL(P_source || P_target)."""
    return full_vocab_kl_tokens(
        source_logits,
        target_logits,
        temperature=temperature,
        token_clip=token_clip,
    )


def full_vocab_kl_tokens(
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    temperature: float = 1.0,
    token_clip: float | None = None,
) -> torch.Tensor:
    """Compute exact token-level KL over the provided full vocabulary.

    PyTorch's KLDivLoss expects the model distribution as ``input`` and the
    reference distribution as ``target``. To preserve the public
    ``KL(P_source || P_target)`` semantics, the log-prob tensors are passed in
    the reversed mathematical order. ViGOS clips after summing KL over the
    vocabulary for each active token.
    """
    kl = _full_vocab_kl_summands(
        source_logits,
        target_logits,
        temperature=temperature,
    )
    return _reduce_kl_summands(
        kl,
        token_clip=token_clip,
    )


def masked_kl_loss(
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    token_mask: torch.Tensor,
    temperature: float = 1.0,
    token_clip: float | None = None,
    post_clip_mask: torch.Tensor | None = None,
    post_token_clip: float | None = None,
) -> torch.Tensor:
    """Compute masked KL loss over active token positions only."""
    if source_logits.shape != target_logits.shape:
        raise ValueError(
            "source_logits and target_logits must have the same shape, got "
            f"{tuple(source_logits.shape)} and {tuple(target_logits.shape)}."
        )
    if token_mask.shape != source_logits.shape[:-1]:
        raise ValueError(
            "token_mask must match logits without the vocabulary dimension, got "
            f"{tuple(token_mask.shape)} for logits {tuple(source_logits.shape)}."
        )
    if post_clip_mask is not None and post_clip_mask.shape != token_mask.shape:
        raise ValueError(
            "post_clip_mask must match token_mask, got "
            f"{tuple(post_clip_mask.shape)} and {tuple(token_mask.shape)}."
        )

    token_mask = token_mask.to(device=source_logits.device, dtype=torch.bool)
    if not bool(token_mask.any()):
        return (source_logits.sum() + target_logits.sum()) * 0.0

    active_post_clip_mask = None
    if post_clip_mask is not None:
        active_post_clip_mask = post_clip_mask.to(
            device=source_logits.device,
            dtype=torch.bool,
        )[token_mask]
    source_logits = source_logits[token_mask]
    target_logits = target_logits[token_mask]
    kl = _full_vocab_kl_summands(
        source_logits,
        target_logits,
        temperature=temperature,
    )
    token_losses = _reduce_kl_summands(
        kl,
        token_clip=token_clip,
    )
    if (
        active_post_clip_mask is not None
        and post_token_clip is not None
        and post_token_clip > 0
    ):
        token_losses = torch.where(
            active_post_clip_mask,
            token_losses.clamp(max=post_token_clip),
            token_losses,
        )
    return token_losses.sum() / token_mask.sum().to(dtype=kl.dtype).clamp_min(1.0)

def _full_vocab_kl_summands(
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    source_log_probs = F.log_softmax(source_logits / temperature, dim=-1)
    target_log_probs = F.log_softmax(target_logits / temperature, dim=-1)
    return F.kl_div(
        target_log_probs,
        source_log_probs,
        reduction="none",
        log_target=True,
    )

def _reduce_kl_summands(
    kl_summands: torch.Tensor,
    *,
    token_clip: float | None,
) -> torch.Tensor:
    if token_clip is None or token_clip <= 0:
        return kl_summands.sum(dim=-1)
    return kl_summands.sum(dim=-1).clamp(max=token_clip)


def js_tokens(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Compute token-level Jensen-Shannon divergence."""

    if top_k is not None and top_k > 0 and top_k < first_logits.shape[-1]:
        indices = torch.topk(first_logits, k=top_k, dim=-1).indices
        first_logits = torch.gather(first_logits, dim=-1, index=indices)
        second_logits = torch.gather(second_logits, dim=-1, index=indices)

    first_log_probs = F.log_softmax(first_logits / temperature, dim=-1)
    second_log_probs = F.log_softmax(second_logits / temperature, dim=-1)
    first_probs = first_log_probs.exp()
    second_probs = second_log_probs.exp()
    mixture = 0.5 * (first_probs + second_probs)
    mixture_log_probs = torch.log(mixture.clamp_min(torch.finfo(mixture.dtype).tiny))
    return 0.5 * (
        (first_probs * (first_log_probs - mixture_log_probs)).sum(dim=-1)
        + (second_probs * (second_log_probs - mixture_log_probs)).sum(dim=-1)
    )
