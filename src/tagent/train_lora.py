"""mlx_lm.lora with a memory fix for Qwen3.5 training on 18 GB.

  .venv/bin/python scripts/llm_locked.py .venv/bin/python -m tagent.train_lora --config configs/train/sft_v3_lora.yaml

In training mode, mlx-lm (0.32.2) runs Qwen3.5's gated-delta recurrence as a Python loop
over tokens (the Metal kernel has no backward pass), and autograd keeps the [B, H, Dv, Dk]
state of every token: ~1.8k-token prompts run out of memory even for the 2B model. This
patch runs the loop in checkpointed chunks, so backward keeps only chunk-boundary states
and recomputes the steps inside each chunk. Same math, same outputs.

Faster path (scalar gate, no padding mask: Qwen3.5 SFT at batch size 1): the chunked
"WY" form used by flash-linear-attention. Within a chunk of C tokens the recurrence is
solved with C x C matrix products instead of C sequential steps:
  U = (I + A)^-1 (diag(beta) V - diag(beta * G) K S0^T),  A_ti = beta_t (G_t/G_i) k_t.k_i  (i < t)
  Y = diag(G) Q S0^T + [(G_t/G_i) q_t.k_i]_(i<=t) U
  S_C = G_C S0 + U^T diag(G_C/G) K
where G = cumulative gate product inside the chunk and S0 the state entering it.
(I + A)^-1 is computed by forward substitution, row by row, for all chunks at once (C
steps per layer, not per chunk). Repeated squaring of the nilpotent A is exact in theory
but overflowed fp32 on real activations (neighbouring keys are nearly parallel).
Checked against the reference loop (outputs, final state, gradients) in tests/test_train_lora.py.
"""
from __future__ import annotations

import mlx.core as mx
from mlx_lm.models import gated_delta

CHUNK = 64


def _chunk(q, k, v, g, beta, state, mask=None):
    ys = []
    for t in range(q.shape[1]):
        y, state = gated_delta._gated_delta_step_ops(
            q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t], state,
            None if mask is None else mask[:, t])
        ys.append(y)
    return mx.stack(ys, axis=1), state


_chunk_ckpt = mx.checkpoint(lambda q, k, v, g, beta, state: _chunk(q, k, v, g, beta, state))
_chunk_ckpt_mask = mx.checkpoint(_chunk)


WY_CHUNK = 32


def gated_delta_wy(q, k, v, g, beta, state):
    """q, k: [B, T, H, Dk] (heads already repeated); v: [B, T, H, Dv]; g, beta: [B, T, H];
    state: [B, H, Dv, Dk]. Returns y [B, T, H, Dv] (q.dtype) and the final state (fp32)."""
    B, T, H, Dk = q.shape
    out_dtype = q.dtype
    C = WY_CHUNK
    pad = (-T) % C
    f32 = mx.float32
    q, k, v, beta = (x.astype(f32) for x in (q, k, v, beta))
    logg = mx.log(mx.maximum(g.astype(f32), 1e-38))
    if pad:  # padded steps: zero key/value/beta and gate 1 -> state unchanged
        z = lambda x: mx.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2))
        q, k, v, beta, logg = z(q), z(k), z(v), z(beta), z(logg)
    n = (T + pad) // C
    # [B, H, n, C, D] / [B, H, n, C]
    to_chunks = lambda x: x.reshape(B, n, C, H, *x.shape[3:]).moveaxis(3, 1)
    q, k, v, beta, logg = (to_chunks(x) for x in (q, k, v, beta, logg))
    L = mx.cumsum(logg, axis=-1)
    idx = mx.arange(C)
    strict = idx[:, None] > idx[None, :]
    incl = idx[:, None] >= idx[None, :]
    D = L[..., :, None] - L[..., None, :]
    decay_strict = mx.exp(mx.where(strict, D, -1e9))
    decay_incl = mx.exp(mx.where(incl, D, -1e9))
    A = beta[..., :, None] * decay_strict * (k @ k.swapaxes(-1, -2))
    # (I + A)^-1 by forward substitution: row t = e_t - sum_{i<t} A_ti row_i
    eye = mx.eye(C, dtype=f32)
    rows = [mx.broadcast_to(eye[0], A.shape[:-2] + (C,))]
    for t in range(1, C):
        rows.append(eye[t] - (A[..., t:t + 1, :t] @ mx.stack(rows, axis=-2))[..., 0, :])
    Tinv = mx.stack(rows, axis=-2)
    G = mx.exp(L)
    Gend = mx.exp(L[..., -1:] - L)
    QK = decay_incl * (q @ k.swapaxes(-1, -2))
    rhs_v = beta[..., None] * v
    bgk = (beta * G)[..., None] * k
    S = state.astype(f32)
    ys = []
    for c in range(n):
        St = S.swapaxes(-1, -2)  # [B, H, Dk, Dv]
        U = Tinv[:, :, c] @ (rhs_v[:, :, c] - bgk[:, :, c] @ St)
        ys.append(G[:, :, c, :, None] * (q[:, :, c] @ St) + QK[:, :, c] @ U)
        S = G[:, :, c, -1, None, None] * S + (U * Gend[:, :, c, :, None]).swapaxes(-1, -2) @ k[:, :, c]
    y = mx.stack(ys, axis=2).reshape(B, H, n * C, -1)[:, :, :T].moveaxis(1, 2)
    return y.astype(out_dtype), S


def gated_delta_ops_chunked(q, k, v, g, beta, state=None, mask=None):
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    if (repeat_factor := Hv // Hk) > 1:
        q = mx.repeat(q, repeat_factor, -2)
        k = mx.repeat(k, repeat_factor, -2)
    if mask is None and g.ndim == 3:
        y, S = gated_delta_wy(q, k, v, g, beta, state)
        return y, S.astype(state.dtype)
    ys = []
    for s in range(0, T, CHUNK):
        sl = slice(s, s + CHUNK)
        if mask is None:
            y, state = _chunk_ckpt(q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl], state)
        else:
            y, state = _chunk_ckpt_mask(q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl], state, mask[:, sl])
        ys.append(y)
    return mx.concatenate(ys, axis=1), state


gated_delta.gated_delta_ops = gated_delta_ops_chunked

if __name__ == "__main__":
    from mlx_lm.lora import main
    main()
