"""Batched autoregressive decoding of one or more output modalities with the nanoHEP GPT.

Supports both ``output_layout`` modes of the dataset:

- ``grouped``: model emits internal ``<MOD_OUT:m>`` boundary tokens
  between modality blocks; helper partitions by those boundaries.

- ``interleaved``: model emits only content tokens (and final ``<EOS>``);
  helper partitions emitted content into 6-token elements (or
  per-modality element widths) and assigns each to a modality by looking
  up the vocab range of its content tokens.

Callers bucket events by prefix length (``GPT.forward`` has no attention
mask) and decode each bucket as one batch.

Optional features:
- ``temperature``: scalar > 0 for stochastic sampling. ``argmax=True`` overrides.
- ``structural_decode``: at each step, mask logits to enforce well-formed
  streams (boundary-then-content for grouped; content-only inside OUT
  region for interleaved). Default True for argmax mode.
"""
from __future__ import annotations

import inspect
import logging
from typing import Dict, List, Optional, Sequence

import torch

from .vocab import Vocab

logger = logging.getLogger(__name__)


@torch.no_grad()
def batched_ar_decode(
    model,
    vocab: Vocab,
    prefixes: torch.Tensor,
    output_modalities: Sequence[str],
    *,
    output_layout: str = "interleaved",
    temperature: float = 1.0,
    argmax: bool = False,
    max_new_tokens: int = 1024,
    structural_decode: Optional[bool] = None,
    device: Optional[str] = None,
    use_kv: bool = True,
) -> Dict[str, List[torch.Tensor]]:
    """Decode a batch of events from same-length prefixes.

    Parameters
    ----------
    model : GPT
        nanoHEP model in eval mode.
    vocab : Vocab
        Used for boundary IDs and the vocab-range modality lookup.
    prefixes : Tensor (B, T_prefix) long
        Same-length AR prefixes for the batch. Each prefix must end with
        the helper-supplied ``<MOD_OUT:m_first>`` cue (in alphabetical
        order of ``output_modalities``).
    output_modalities : sequence[str]
        Which OUT modalities are declared in the header. Used to validate
        boundary tokens (grouped) and partition emissions (interleaved).
    output_layout : "grouped" | "interleaved"
    temperature : float
        Softmax temperature; ignored if ``argmax`` is True.
    argmax : bool
        Take ``argmax(dim=-1)`` instead of sampling.
    max_new_tokens : int
    structural_decode : bool, optional
        If True, mask out invalid logits at each step. Default: True for
        argmax, False for sampling.
    device : str, optional
        Device override; if None, use ``next(model.parameters()).device``.
    use_kv : bool
        Use the incremental KV cache (default True) — prime on the prefix, then
        feed one token per step. Identical outputs to the full-recompute path
        (tested) at O(T) instead of O(T^2) per step. False recomputes the whole
        sequence at every step.

    Returns
    -------
    dict[str, list[Tensor]]
        Per-output-modality, per-event jagged token tensors. Each tensor
        has shape ``(n_elements_for_event * element_token_width,)`` and
        contains the global token IDs the model emitted for that
        modality (raw stream; pass to ``vocab.decode_element_with_pos``
        or feed directly to the ``Detokenizer``).
    """
    if output_layout not in ("grouped", "interleaved"):
        raise ValueError(f"output_layout must be grouped|interleaved, got {output_layout!r}")
    if structural_decode is None:
        structural_decode = argmax

    # alphabetical order, as in training
    output_modalities = sorted(output_modalities)
    if not output_modalities:
        raise ValueError("output_modalities must be non-empty")

    if device is None:
        device = next(model.parameters()).device
    prefixes = prefixes.to(device)
    B, T_prefix = prefixes.shape

    # The prefix ends with the <MOD_OUT:m_first> cue.
    m_first = output_modalities[0]
    expected_first_cue = vocab.mod_out[m_first]
    if not (prefixes[:, -1] == expected_first_cue).all():
        bad = (prefixes[:, -1] != expected_first_cue).nonzero().squeeze(-1).tolist()
        raise ValueError(
            f"prefix[:, -1] should be <MOD_OUT:{m_first}>={expected_first_cue}; "
            f"{len(bad)} rows differ (e.g. row {bad[0]} has {int(prefixes[bad[0], -1])})"
        )

    eos = vocab.eos
    pad = vocab.pad
    mod_out_ids = {vocab.mod_out[m]: m for m in output_modalities}
    mod_out_id_set = set(mod_out_ids.keys())

    # Per-row state
    cur = prefixes.clone()  # (B, T) grows
    done = torch.zeros(B, dtype=torch.bool, device=device)
    # Per-row, per-modality emitted streams (held on CPU as python lists).
    streams: List[Dict[str, List[int]]] = [{m: [] for m in output_modalities} for _ in range(B)]
    # Per-row currently-active OUT modality (starts at m_first for both layouts).
    active_mod: List[str] = [m_first for _ in range(B)]

    # For interleaved: per-row count of content tokens emitted INTO the current element.
    # Once it hits element_token_width(active_mod), the element is complete and we
    # decide modality for the NEXT element from the next token's vocab range.
    elem_progress: List[int] = [0 for _ in range(B)]
    elem_width: List[int] = [vocab.element_token_width(m_first) for _ in range(B)]

    # Never exceed the model context: rows still generating when the context
    # fills are finalized as-is (their streams truncate at the last complete
    # element downstream). Without this cap the forward pass asserts.
    model_block = getattr(getattr(model, "config", None), "block_size", None)
    if model_block is not None:
        budget = model_block - T_prefix
        if budget < max_new_tokens:
            logger.debug("capping max_new_tokens %d -> %d (prefix %d + cap = model block size %d)",
                        max_new_tokens, budget, T_prefix, model_block)
            max_new_tokens = max(budget, 0)

    # ---- KV cache (default ON) ----------------------------------------------
    # GPT.forward supports an incremental cache (past_kv_list + shifted
    # positions). Prime once on the full prefix, then feed a single token per
    # step instead of recomputing the whole growing sequence. Identical to the
    # recompute path (tests/unit/test_nano_hep_decode_kv.py).
    can_kv = use_kv and (
        "return_kv_list" in inspect.signature(type(model).forward).parameters
    )
    kv_list = None
    primed_logits = None
    if can_kv:
        primed_logits, _, kv_list = model(cur, return_kv_list=True)

    for step in range(max_new_tokens):
        if done.all():
            break

        # ---- forward ----
        if can_kv:
            # next-token logits: from the prefix prime (step 0) or the previous
            # step's single-token feed (step > 0).
            logits = primed_logits
        else:
            logits, _ = model(cur)  # (B, 1 or T, V); GPT returns last-pos only when targets=None
        if logits.shape[1] != 1:
            logits = logits[:, -1:, :]
        last = logits[:, -1, :]  # (B, V)

        # ---- structural masking (optional) ----
        if structural_decode:
            mask = _structural_mask(
                vocab=vocab,
                B=B,
                active_mod=active_mod,
                elem_progress=elem_progress,
                elem_width=elem_width,
                done=done,
                output_modalities=output_modalities,
                output_layout=output_layout,
                device=device,
            )
            # mask is (B, V) bool, True = allowed
            last = last.masked_fill(~mask, float("-inf"))

        # ---- sample / argmax ----
        if argmax:
            nxt = last.argmax(dim=-1)  # (B,)
        else:
            probs = torch.softmax(last / max(temperature, 1e-6), dim=-1)
            nxt = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Pad already-done rows with PAD so they don't perturb the sequence.
        nxt = torch.where(done, torch.full_like(nxt, pad), nxt)

        # ---- consume per row ----
        nxt_cpu = nxt.detach().cpu().tolist()
        for b in range(B):
            if done[b]:
                continue
            t = nxt_cpu[b]
            if t == eos:
                done[b] = True
                continue
            if t in mod_out_id_set:
                # Boundary tokens are not recorded in the streams. Grouped: switch
                # modality. Interleaved: a stray boundary (possible only when
                # sampling without structural_decode) is ignored.
                if output_layout == "grouped":
                    new_mod = mod_out_ids[t]
                    active_mod[b] = new_mod
                    elem_progress[b] = 0
                    elem_width[b] = vocab.element_token_width(new_mod)
                continue

            # Content token: add to active modality's stream and track element progress.
            if output_layout == "interleaved":
                # In interleaved mode, the modality of THIS token is read from its
                # vocab range. We don't trust active_mod to track it.
                tok_mod = vocab.modality_of(t)
                if tok_mod and tok_mod in streams[b]:
                    streams[b][tok_mod].append(t)
                    # Once we've completed an element worth of THIS modality, next
                    # token may be a different modality.
                    elem_progress[b] += 1
                    # Determine the "active" modality based on the most recently emitted
                    # content (so structural_decode on the next step knows what's valid).
                    # In interleaved we generally allow ANY OUT modality + EOS.
                    if elem_progress[b] >= vocab.element_token_width(tok_mod):
                        elem_progress[b] = 0
                # else: vocab-range lookup failed (token in IN range, special, etc.).
                # Structural decode should prevent this in argmax mode; in sampling
                # mode it's a model error -- skip the token.
            else:  # grouped
                streams[b][active_mod[b]].append(t)
                elem_progress[b] += 1
                if elem_progress[b] >= elem_width[b]:
                    elem_progress[b] = 0  # ready for a boundary or next element of same mod

        # ---- append to running sequence ----
        cur = torch.cat([cur, nxt.unsqueeze(1)], dim=1)

        # ---- KV: feed the just-emitted token to obtain the next step's logits.
        # nxt already has PAD substituted for finished rows (parity with the
        # recompute path, which appends those same PADs to `cur`). Skip on the
        # final iteration and once every row is done.
        if can_kv and step + 1 < max_new_tokens and not done.all():
            primed_logits, _, kv_list = model(
                nxt.unsqueeze(1), past_kv_list=kv_list, return_kv_list=True
            )

    # ---- pack per-event per-modality tensors ----
    out: Dict[str, List[torch.Tensor]] = {m: [] for m in output_modalities}
    for b in range(B):
        for m in output_modalities:
            ids = streams[b][m]
            out[m].append(torch.tensor(ids, dtype=torch.long))
    return out


def _structural_mask(
    *,
    vocab: Vocab,
    B: int,
    active_mod: List[str],
    elem_progress: List[int],
    elem_width: List[int],
    done: torch.Tensor,
    output_modalities: Sequence[str],
    output_layout: str,
    device,
) -> torch.Tensor:
    """Return ``(B, V)`` bool mask: True at positions the model is allowed
    to emit next.

    Rules:
      - PAD never emittable.
      - In ``grouped``:
          * Inside an element (progress < width): only content tokens of
            the active modality are allowed.
          * At element boundary (progress == 0 OR == width): either a
            content token of the active mod (continue) OR ``<MOD_OUT:m>``
            of a different declared output modality (switch) OR ``<EOS>``.
      - In ``interleaved``:
          * Inside an element (progress > 0): only content tokens of the
            most-recently-emitted modality are allowed (to complete this
            element). We approximate this by allowing content tokens of
            the active modality only.
          * At element boundary (progress == 0): any content token of any
            declared output modality, OR ``<EOS>``.

    Note: the active-modality bookkeeping in interleaved is best-effort -- we
    set ``active_mod[b]`` after each emitted content token to the modality
    of THAT token, then ``elem_progress`` counts up to ``elem_width(active_mod)``.
    """
    V = vocab.total
    mask = torch.zeros((B, V), dtype=torch.bool, device=device)

    declared_mod_out_ids = [vocab.mod_out[m] for m in output_modalities]

    for b in range(B):
        if done[b]:
            mask[b, :] = True  # already done -- masking irrelevant; sample anything
            continue

        am = active_mod[b]
        ew = vocab.element_token_width(am) if am else 1
        elem_width[b] = ew  # keep in sync

        at_boundary = (elem_progress[b] == 0) or (elem_progress[b] >= ew)

        # Always: PAD disallowed (we use a separate done flag).
        # Content range for ALL declared OUT mods is allowed at boundaries.
        if at_boundary:
            mask[b, vocab.eos] = True
            for m in output_modalities:
                # A new element starts with content quantizer 0 ONLY (elements
                # are encoded content-first): allowing the full content/pos
                # range let SAMPLED decodes start an element mid-quantizer,
                # which detokenizes to out-of-range codebook indices (device
                # assert in the VQ embedding lookup). Argmax never hit this.
                c_lo = vocab.offsets[m]
                mask[b, c_lo:c_lo + vocab.codebook_sizes[m]] = True
            if output_layout == "grouped":
                # Allow modality-switch boundaries to OTHER declared OUTs.
                for t in declared_mod_out_ids:
                    # Don't force-allow current mod; staying within is via content.
                    mask[b, t] = True
            # else: interleaved -- never emit MOD_OUT inside the body.
        else:
            # Inside an element: only the token range of the CURRENT slot
            # (slot k of an element = content quantizer k for k < n_q, else
            # pos quantizer k - n_q; encoding is content-first, see
            # vocab.encode_element_with_pos). Sampling outside the slot's
            # codebook range detokenizes to invalid indices.
            k = elem_progress[b]
            n_q = vocab.num_quantizers[am]
            if k < n_q:
                lo = vocab.offsets[am] + k * vocab.codebook_sizes[am]
                mask[b, lo:lo + vocab.codebook_sizes[am]] = True
            else:
                kp = k - n_q
                lo = vocab.pos_offsets[am] + kp * vocab.pos_codebook_sizes[am]
                mask[b, lo:lo + vocab.pos_codebook_sizes[am]] = True

    return mask


# --------------------------------------------------------------------------
# Prefix-length bucketing helper
# --------------------------------------------------------------------------

def bucket_by_prefix_length(prefixes: List[torch.Tensor]) -> Dict[int, List[int]]:
    """Group event indices by their prefix's length.

    Returns ``{prefix_len: [event_idx, ...]}``. ``GPT.forward`` has no attention
    mask, so all rows in a batched call must share the same prefix length.
    """
    buckets: Dict[int, List[int]] = {}
    for i, p in enumerate(prefixes):
        L = int(p.shape[-1])
        buckets.setdefault(L, []).append(i)
    return buckets
