"""
train_gpt_ppt.py — NCA Pre-Pre-Training (PPT) experiment.

Based on train_gpt.py with added NCA pre-pre-training phase from:
  https://github.com/danihyunlee/nca-pre-pretraining
  Paper: "Training Language Models via Neural Cellular Automata" (arXiv:2603.10055)

Pipeline: Warmup → NCA Pre-Pre-Training → Text Training (normal 10min run)

The PPT phase trains the transformer blocks on next-token prediction over
serialized NCA grid dynamics before language data. Embeddings are re-initialized
for text after PPT. Only transformer block weights transfer.

This is NOT part of the competition — purely exploratory research.
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Depth recurrence: repeat specified layers for free virtual depth.
    # E.g. RECUR_LAYERS="3,4" repeats layers 3,4 creating 11 virtual layers from 9 physical.
    recur_layers = os.environ.get("RECUR_LAYERS", "").strip()
    recur_start_frac = float(os.environ.get("RECUR_START_FRAC", "0.5"))

    # Multi-Token Prediction (training-only auxiliary loss, DeepSeek V3 / Nemotron 3 style).
    # Sequential MTP modules predict future tokens using ground-truth token feeding.
    # Each module: Concat(RMSNorm(hidden), embed(gt_token)) → projection → shared TRM block.
    # All MTP parameters are discarded at export — zero artifact size cost.
    mtp_num_depths = int(os.environ.get("MTP_NUM_DEPTHS", 0))
    mtp_loss_weight = float(os.environ.get("MTP_LOSS_WEIGHT", 0.3))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    # Muon Newton-Schulz algorithm selection:
    #   "standard" = original Newton-Schulz (zeropower_via_newtonschulz5)
    #   "gram"     = Gram Newton-Schulz from https://github.com/Dao-AILab/gram-newton-schulz
    muon_ns_algorithm = os.environ.get("MUON_NS_ALGORITHM", "standard")
    # Coefficient preset for GNS mode: "you" (best tested) or "polar_express"
    muon_gns_coefficients = os.environ.get("MUON_GNS_COEFFICIENTS", "you")
    # LR adjustment for GNS mode: "none" (best tested), "rms_norm", or "spectral_norm"
    muon_gns_adjust_lr = os.environ.get("MUON_GNS_ADJUST_LR", "none")
    # Weight decay for GNS Muon (decoupled); enables model compression for larger architectures
    muon_gns_weight_decay = float(os.environ.get("MUON_GNS_WEIGHT_DECAY", 0.1))

    # -------------------------------------------------------------------
    # NCA Pre-Pre-Training (PPT) settings
    # -------------------------------------------------------------------
    # Pipeline: Warmup → PPT (NCA dynamics) → Text Training
    # PPT trains transformer blocks on serialized NCA grid trajectories using
    # next-token prediction. After PPT, embeddings are re-initialized for text.
    # Only transformer block weights (attention, MLP, norms) transfer.
    #
    # Reference: "Training Language Models via Neural Cellular Automata"
    #            https://arxiv.org/abs/2603.10055
    ppt_enable = bool(int(os.environ.get("PPT_ENABLE", "1")))
    ppt_iterations = int(os.environ.get("PPT_ITERATIONS", 500))       # PPT training steps
    ppt_lr = float(os.environ.get("PPT_LR", 1e-4))                   # Adam LR for PPT
    ppt_batch_size = int(os.environ.get("PPT_BATCH_SIZE", 32))        # Batch size per GPU
    ppt_seq_len = int(os.environ.get("PPT_SEQ_LEN", 1024))           # Sequence length for NCA tokens
    ppt_grid = int(os.environ.get("PPT_GRID", 12))                   # NCA grid size (H=W)
    ppt_patch = int(os.environ.get("PPT_PATCH", 2))                  # Patch size for tokenization
    ppt_num_colors = int(os.environ.get("PPT_NUM_COLORS", 10))       # NCA state space size
    ppt_num_rules = int(os.environ.get("PPT_NUM_RULES", 2000))       # Unique NCA rules to generate
    ppt_num_sims = int(os.environ.get("PPT_NUM_SIMS", 16000))        # Trajectories generated per data refresh
    ppt_filter_threshold = float(os.environ.get("PPT_FILTER_THRESHOLD", 0.5))  # Gzip complexity lower bound
    ppt_filter_upper = float(os.environ.get("PPT_FILTER_UPPER", 1.0))          # Gzip complexity upper bound
    ppt_dT = int(os.environ.get("PPT_DT", 1))                       # Timesteps between NCA snapshots
    ppt_init_rollout = int(os.environ.get("PPT_INIT_ROLLOUT", 10))   # Initial rollout steps to skip
    ppt_identity_bias = float(os.environ.get("PPT_IDENTITY_BIAS", 0.0))
    ppt_temperature = float(os.environ.get("PPT_TEMPERATURE", 1e-4)) # NCA transition temperature
    ppt_warmup_frac = float(os.environ.get("PPT_WARMUP_FRAC", 0.1))  # Fraction of PPT iters for LR warmup
    ppt_log_every = int(os.environ.get("PPT_LOG_EVERY", 50))         # Log frequency during PPT
    ppt_regen_epochs = int(os.environ.get("PPT_REGEN_EPOCHS", 1))    # Regenerate rules every N epochs

# -----------------------------
# MUON OPTIMIZER 
# -----------------------------
# 
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


# -----------------------------
# GRAM NEWTON-SCHULZ COEFFICIENTS & ALGORITHM
# -----------------------------
#
# From https://github.com/Dao-AILab/gram-newton-schulz
# Gram Newton-Schulz iterates on the smaller Gram matrix XX^T instead of X,
# reducing FLOPs for non-square matrices and enabling per-step tuned coefficients.

# Per-step coefficients from You Jiacheng: https://x.com/YouJiacheng/status/1905861218138804534
YOU_COEFFICIENTS = [
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
]

# Polar Express coefficients from https://arxiv.org/pdf/2505.16932, with safety scaling
_UNMODIFIED_POLAR_EXPRESS = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
]
_SAFETY = 1.05
POLAR_EXPRESS_COEFFICIENTS = [
    (a / _SAFETY, b / _SAFETY**3, c / _SAFETY**5)
    for (a, b, c) in _UNMODIFIED_POLAR_EXPRESS
]

GNS_COEFFICIENT_PRESETS = {
    "you": YOU_COEFFICIENTS,
    "polar_express": POLAR_EXPRESS_COEFFICIENTS,
}


def zeropower_via_gram_newton_schulz(
    G: Tensor,
    coefficients: list[tuple[float, float, float]],
    restart_at: list[int] | None = None,
    eps: float = 1e-7,
) -> Tensor:
    """Gram Newton-Schulz orthogonalization (pure PyTorch, no custom kernels).

    For non-square matrices, iterates on the smaller Gram matrix R = XX^T,
    with periodic restarts for numerical stability. For square matrices,
    falls back to standard Newton-Schulz (matching official behaviour).

    Reference: https://github.com/Dao-AILab/gram-newton-schulz
    """
    if restart_at is None:
        restart_at = [2]  # default from GNS repo for 5-step configs

    original_dtype = G.dtype
    X = G.float()
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T

    X /= X.norm() + eps
    X = X.half()  # fp16 for iteration speed (matching GNS reference)

    is_square = X.size(0) == X.size(1)
    if is_square:
        # Standard Newton-Schulz: no FLOP advantage from Gram trick on square matrices.
        for a, b, c in coefficients:
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    else:
        # Gram Newton-Schulz: iterate on R = XX^T (smaller than X for non-square).
        n = X.size(0)
        I = torch.eye(n, device=X.device, dtype=X.dtype)

        R = X @ X.T
        Q: Tensor | None = None

        for i, (a, b, c) in enumerate(coefficients):
            if i in restart_at and i != 0:
                X = Q @ X  # type: ignore[union-attr]
                R = X @ X.T
                Q = None

            Z = b * R + c * (R @ R)

            if Q is None:
                Q = Z + a * I
            else:
                Q = Q @ Z + a * Q

            if i < len(coefficients) - 1 and (i + 1) not in restart_at:
                RZ = R @ Z + a * R
                R = Z @ RZ + a * RZ

        X = Q @ X  # type: ignore[union-attr]

    if transposed:
        X = X.T

    return X.to(original_dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed and world_size > 1:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


class MuonGNS(torch.optim.Optimizer):
    """Muon optimizer using Gram Newton-Schulz orthogonalization.

    Drop-in replacement for Muon that uses GNS (iterating on the Gram matrix
    XX^T) instead of standard Newton-Schulz. Follows the GNS reference:
    https://github.com/Dao-AILab/gram-newton-schulz

    Key differences from standard Muon:
    - Uses per-step tuned coefficients instead of a single (a,b,c) triple
    - Iterates on the smaller Gram matrix for non-square weight matrices
    - Uses fp16 for the NS iteration (matching GNS reference)
    - LR scaling follows 'rms_norm' convention: 0.2 * sqrt(max(fan_out, fan_in))
    - Supports weight decay (decoupled, applied after the update)
    """

    def __init__(
        self,
        params,
        lr: float,
        momentum: float,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_coefficients: list[tuple[float, float, float]] | None = None,
        ns_restart_at: list[int] | None = None,
        adjust_lr: str = "rms_norm",
    ):
        if ns_coefficients is None:
            ns_coefficients = YOU_COEFFICIENTS
        if ns_restart_at is None:
            ns_restart_at = [2]
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_coefficients=ns_coefficients,
            ns_restart_at=ns_restart_at,
            adjust_lr=adjust_lr,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _adjust_lr(lr: float, shape: tuple[int, ...], method: str) -> float:
        fan_out, fan_in = shape[0], shape[1]
        if method == "rms_norm":
            return lr * 0.2 * math.sqrt(max(fan_out, fan_in))
        elif method == "spectral_norm":
            return lr * math.sqrt(fan_out / fan_in)
        return lr

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            weight_decay = group["weight_decay"]
            coefficients = group["ns_coefficients"]
            restart_at = group["ns_restart_at"]
            adjust_lr_method = group["adjust_lr"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    # Gram Newton-Schulz orthogonalization
                    g = zeropower_via_gram_newton_schulz(
                        g, coefficients=coefficients, restart_at=restart_at
                    )
                    # LR adjustment per-shape (following GNS reference)
                    adjusted_lr = self._adjust_lr(lr, g.shape, adjust_lr_method)
                    g = g * adjusted_lr
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1).to(torch.bfloat16)
                curr += p.numel()

            if distributed and world_size > 1:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                # Decoupled weight decay (applied at base_lr, not adjusted lr)
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)
                p.add_(g, alpha=-1.0)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP 
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
# Instead, we get approximately the same model (with a small hit) by quantizing the model to int8 & zlib compressing.
# We can then decompress the model and run in higher precision for evaluation, after closing in under the size limit.

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    # Single supported clean-script export format:
    # - per-row int8 for 2D float tensors
    # - per-tensor int8 for other float tensors
    # - exact passthrough for non-floats
    # - passthrough for small float tensors, stored as fp16 to save bytes
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            # Broadcast the saved row scale back across trailing dimensions.
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING 
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    # Reads shards sequentially and wraps around forever. The training loop therefore
    # has deterministic, simple streaming behavior with no sampling or workers.
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    # Each call consumes a contiguous chunk from the shared token stream, then slices out
    # one disjoint span per rank. The extra "+1" token lets us build (x, y) by shifting.
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


# -----------------------------
# MULTI-TOKEN PREDICTION (MTP)
# -----------------------------
#
# DeepSeek V3 / Nemotron 3 Super style sequential MTP modules.
# Each module receives backbone hidden states + ground-truth next-token embeddings,
# projects them through a shared transformer block, and predicts future tokens.
# All MTP modules are training-only — discarded before export (zero artifact cost).
# Reference: DeepSeek V3 paper Section 3.3, Nemotron 3 Super 120B model card.


class MTPProjection(nn.Module):
    """Per-depth projection for MTP: Concat(RMSNorm(hidden), embed) → Linear(2D, D)."""

    def __init__(self, model_dim: int):
        super().__init__()
        self.norm = RMSNorm()
        self.proj = CastedLinear(2 * model_dim, model_dim, bias=False)
        self.proj._zero_init = True

    def forward(self, hidden: Tensor, token_embed: Tensor) -> Tensor:
        return self.proj(torch.cat([self.norm(hidden), token_embed], dim=-1))


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        recur_layers: list[int] | None = None,
        mtp_num_depths: int = 0,
        mtp_loss_weight: float = 0.3,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.num_physical_layers = num_layers
        self.tok_emb = nn.Embedding(vocab_size, model_dim)

        # Depth recurrence: build virtual-to-physical layer mappings.
        # recur_layers lists physical layer indices to repeat once after their
        # first occurrence.  E.g. recur_layers=[3,4] with 9 physical layers
        # yields v2p_recur = [0,1,2,3,4, 3,4, 5,6,7,8] (11 virtual layers).
        self.recur_layer_indices = sorted(set(recur_layers or []))
        for rl in self.recur_layer_indices:
            if not 0 <= rl < num_layers:
                raise ValueError(f"recur layer {rl} out of range [0, {num_layers})")
        if self.recur_layer_indices:
            cutoff = max(self.recur_layer_indices) + 1
            self._v2p_recur = list(range(cutoff)) + self.recur_layer_indices + list(range(cutoff, num_layers))
        else:
            self._v2p_recur = list(range(num_layers))
        self._v2p_no_recur = list(range(num_layers))

        # U-Net encoder/decoder split for both modes.
        n_virt = len(self._v2p_recur)
        self._enc_recur = n_virt // 2
        self._dec_recur = n_virt - self._enc_recur
        self._enc_no_recur = num_layers // 2
        self._dec_no_recur = num_layers - self._enc_no_recur

        # Start in non-recurrent mode; skip_weights sized for the larger (recurrent) mode.
        self.num_skip_weights = min(self._enc_recur, self._dec_recur)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

        # The actual state: default to non-recurrent.
        self._recurrence_active = False
        self.v2p = self._v2p_no_recur
        self.num_encoder_layers = self._enc_no_recur
        self.num_decoder_layers = self._dec_no_recur

        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True

        # MTP (Multi-Token Prediction): training-only sequential modules.
        # Per-depth projections (unique) + one shared transformer block (reused across depths).
        # All discarded at export — zero artifact size cost.
        self.mtp_num_depths = mtp_num_depths
        self.mtp_loss_weight = mtp_loss_weight
        self.mtp_projections = nn.ModuleList(
            [MTPProjection(model_dim) for _ in range(mtp_num_depths)]
        )
        self.mtp_shared_block = (
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            if mtp_num_depths > 0
            else None
        )
        self.mtp_norm = RMSNorm() if mtp_num_depths > 0 else None

        self._init_weights()

    def set_recurrence_active(self, active: bool) -> None:
        """Toggle depth recurrence on/off. Adjusts v2p and encoder/decoder split."""
        self._recurrence_active = bool(active) and bool(self.recur_layer_indices)
        if self._recurrence_active:
            self.v2p = self._v2p_recur
            self.num_encoder_layers = self._enc_recur
            self.num_decoder_layers = self._dec_recur
        else:
            self.v2p = self._v2p_no_recur
            self.num_encoder_layers = self._enc_no_recur
            self.num_decoder_layers = self._dec_no_recur

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def _compute_logits(self, hidden_flat: Tensor) -> Tensor:
        """Shared logit computation for main head and MTP heads."""
        if self.tie_embeddings:
            logits_proj = F.linear(hidden_flat, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(hidden_flat)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        bsz, seqlen = input_ids.shape
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        v2p = self.v2p

        # First half stores skips; second half reuses them in reverse order.
        # v2p maps virtual layer indices to physical Block indices, enabling
        # depth recurrence (some physical blocks run more than once).
        for i in range(self.num_encoder_layers):
            x = self.blocks[v2p[i]](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[v2p[self.num_encoder_layers + i]](x, x0)

        x_normed = self.final_norm(x)
        main_logits = self._compute_logits(x_normed.reshape(-1, x_normed.size(-1)))
        main_loss = F.cross_entropy(main_logits.float(), target_ids.reshape(-1), reduction="mean")

        # MTP: sequential multi-token prediction (training only).
        # Each depth k predicts token[t+k+2] given hidden[t] + embed(token[t+k+1]).
        # Uses ground-truth token feeding → shared projection → shared TRM block → shared LM head.
        if self.training and self.mtp_num_depths > 0 and self.mtp_loss_weight > 0:
            mtp_loss_sum = main_loss.new_zeros(())
            mtp_count = 0
            prev_hidden = x_normed  # [B, T, D] — backbone output after final_norm
            for k, mtp_proj in enumerate(self.mtp_projections):
                # For depth k: predict target_ids[:, k+1:] from hidden[:, :-(k+1)]
                # using ground-truth embed of target_ids[:, k:-(1 if k+1<seqlen else seqlen)]
                shift = k + 1
                valid_len = seqlen - shift
                if valid_len <= 1:
                    break
                # Ground-truth embeddings of the token we're conditioning on
                gt_embeds = self.tok_emb(target_ids[:, k : k + valid_len])  # [B, valid_len, D]
                gt_embeds = F.rms_norm(gt_embeds, (gt_embeds.size(-1),))
                h_slice = prev_hidden[:, :valid_len, :]  # [B, valid_len, D]
                # Per-depth projection: Concat(Norm(hidden), embed) → Linear(2D, D)
                mtp_input = mtp_proj(h_slice, gt_embeds)  # [B, valid_len, D]
                # Shared transformer block processes the combined representation
                mtp_hidden = self.mtp_shared_block(mtp_input, mtp_input)  # [B, valid_len, D]
                mtp_normed = self.mtp_norm(mtp_hidden)  # type: ignore[misc]
                # Predict the token AFTER the ground-truth token we conditioned on
                mtp_targets = target_ids[:, shift : shift + valid_len]  # [B, valid_len]
                mtp_logits = self._compute_logits(mtp_normed.reshape(-1, mtp_normed.size(-1)))
                mtp_loss_sum = mtp_loss_sum + F.cross_entropy(
                    mtp_logits.float(), mtp_targets.reshape(-1), reduction="mean"
                )
                mtp_count += 1
                prev_hidden = mtp_normed  # Sequential chaining for next depth
            if mtp_count > 0:
                main_loss = main_loss + self.mtp_loss_weight * (mtp_loss_sum / mtp_count)

        return main_loss


# -----------------------------
# NCA PRE-PRE-TRAINING (PPT)
# -----------------------------
#
# Ported from https://github.com/danihyunlee/nca-pre-pretraining
# NCA simulation is done in JAX (fast batched rollouts on CPU/GPU).
# Tokenization converts 12×12 grids into patch-based token sequences.
# The transformer is trained with standard next-token prediction on these
# serialized NCA grid trajectories before seeing any natural language.
#
# Architecture alignment with original repo:
# - NCA dynamics: random conv rules (3×3→ReLU→1×1), wrap-around padding
# - Tokenizer: 2×2 patches → base-10 encoding → start/end tokens per timestep
# - Training: Adam optimizer, CrossEntropyLoss (ignore_index=-100 for masking)
# - Transfer: block weights → text model, embeddings re-initialized


def _gzip_complexity(data: bytes) -> float:
    """Compute gzip compression ratio (matching official compute_rule_gzip_batch)."""
    import gzip as gz
    import io as _io
    buf = _io.BytesIO()
    with gz.GzipFile(fileobj=buf, mode='wb', compresslevel=9) as f:
        f.write(data)
    return len(buf.getvalue()) / max(len(data), 1)


def generate_nca_data_jax(seed: int, num_sims: int, grid: int, d_state: int,
                          identity_bias: float, temperature: float,
                          num_examples: int, dT: int, start_step: int,
                          num_rules: int,
                          filter_threshold: float = 0.0,
                          filter_upper: float = 1.0):
    """Generate NCA simulation data using JAX with random conv transition rules.

    Following the NCA repo's approach: random conv kernels define transition rules,
    simulations are rolled out with jax.lax.scan, and all sims are vectorized via jax.vmap.
    Optionally filters rules by gzip complexity (matching official --filter_rules).

    Returns: numpy array of shape (num_sims, num_examples, grid, grid) with integer cell states
    """
    import jax
    import jax.numpy as jnp
    import jax.lax as jlax
    from jax.random import split as jax_split

    # Ensure num_rules <= num_sims to avoid tiling issues
    # (original repo generates num_sims = train_num_sim * batch_size * grad_accum >> num_rules)
    effective_rules = min(num_rules, num_sims)

    rng = jax.random.PRNGKey(seed)

    # Total timesteps to simulate per trajectory
    total_steps = start_step + dT * num_examples
    # Snapshot indices
    snapshot_idx = jnp.arange(start_step, start_step + dT * num_examples, dT)

    def make_rule(rule_rng):
        """Create random conv-based NCA transition rule matching official NCANetwork exactly:
        pad(wrap,1 on ALL dims) → Conv(d_state+2→4, 3×3) → Conv(4→16, 1×1) → ReLU → Conv(16→d_state, 1×1)
        """
        r1, r2, r3 = jax.random.split(rule_rng, 3)
        # Layer 1: 3×3 conv, d_state+2 → 4 channels (input is padded on ALL dims including channels)
        k1 = jax.random.normal(r1, (3, 3, d_state + 2, 4)) * 0.5
        # Layer 2: 1×1 conv, 4 → 16 channels (matching nn.Conv(features=16, kernel_size=(1,1)))
        k2 = jax.random.normal(r2, (1, 1, 4, 16)) * 0.5
        # Layer 3: 1×1 conv, 16 → d_state (matching nn.Conv(features=self.d_state, kernel_size=(1,1)))
        k3 = jax.random.normal(r3, (1, 1, 16, d_state)) * 0.5
        return k1, k2, k3

    def apply_rule(rule_params, state_oh):
        """Apply NCA transition rule matching official NCANetwork:
        pad(wrap,1) → Conv(3×3, d_state+2→4) → Conv(1×1, 4→16) → ReLU → Conv(1×1, 16→d_state)

        NOTE: Official NCANetwork uses jnp.pad(x, pad_width=1, mode='wrap') which pads ALL
        dimensions including channels (H+2, W+2, C+2). This gives the first conv d_state+2
        input channels.
        """
        k1, k2, k3 = rule_params
        # Wrap-around padding matching official: jnp.pad(x, pad_width=1, mode='wrap')
        # pads ALL dims including channels: (H,W,C) → (H+2, W+2, C+2)
        padded = jnp.pad(state_oh, ((1, 1), (1, 1), (1, 1)), mode='wrap')
        # Layer 1: 3×3 conv
        h = jlax.conv_general_dilated(
            padded[None], k1, window_strides=(1, 1), padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )[0]
        # Layer 2: 1×1 conv
        h = jlax.conv_general_dilated(
            h[None], k2, window_strides=(1, 1), padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )[0]
        # ReLU (matching official: x = nn.relu(x))
        h = jax.nn.relu(h)
        # Layer 3: 1×1 conv → logits
        logits = jlax.conv_general_dilated(
            h[None], k3, window_strides=(1, 1), padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )[0]
        return logits

    def rollout_one(sim_rng, rule_rng):
        """Roll out one NCA trajectory with a random rule."""
        rule_params = make_rule(rule_rng)

        # Initialize random starting grid (matching NCA repo's init_state exactly:
        #   init = random_normal(rng, (n_groups, d_state)) repeated over H×W
        #   state = categorical(rng, init, axis=-1)
        # This produces non-uniform starting states biased by random logits)
        sim_rng, init_rng = jax_split(sim_rng)
        init_logits = jax.random.normal(init_rng, (d_state,))
        init_logits_grid = jnp.broadcast_to(init_logits, (grid, grid, d_state))
        state = jax.random.categorical(
            init_rng, init_logits_grid, axis=-1
        )

        # Generate per-step RNG keys
        step_rngs = jax.random.split(sim_rng, total_steps)

        def step_fn(state, step_rng):
            """One NCA step: one-hot → rule → categorical sample (matching NCA repo's step_state)."""
            state_oh = jax.nn.one_hot(state, d_state)
            logits = apply_rule(rule_params, state_oh)
            # Identity bias + temperature (matching NCA repo)
            next_state = jax.random.categorical(
                step_rng,
                (logits + state_oh * identity_bias) / jnp.maximum(temperature, 1e-10),
                axis=-1
            )
            return next_state, state

        _, trajectory = jax.lax.scan(step_fn, state, step_rngs)
        # Return snapshots at selected indices
        return trajectory[snapshot_idx]

    # Generate rule seeds — handle num_sims ≥ or < num_rules
    rng, rules_rng = jax_split(rng)
    rule_seeds_base = jax.random.split(rules_rng, effective_rules)

    # Tile/truncate to exactly num_sims
    if num_sims <= effective_rules:
        rule_seeds = rule_seeds_base[:num_sims]
    else:
        # Tile and trim
        repeats = (num_sims + effective_rules - 1) // effective_rules
        rule_seeds = jnp.tile(rule_seeds_base, (repeats, 1))[:num_sims]

    # Generate all simulations (vectorized)
    sim_rngs = jax.random.split(rng, num_sims)
    sims = jax.vmap(rollout_one)(sim_rngs, rule_seeds)
    sims_np = np.array(sims)

    # Gzip complexity filtering (matching official --filter_rules with mode='gzip')
    # Keep only simulations whose tokenized gzip compression ratio falls within
    # [filter_threshold, filter_upper]. This selects "interesting" NCA dynamics:
    # not too simple (repetitive) and not too random (incompressible).
    if filter_threshold > 0.0 or filter_upper < 1.0:
        N_H = grid // 2  # patch=2
        N_W = grid // 2
        powers = d_state ** np.arange(4)  # patch=2 → 4 cells per patch
        keep_mask = np.ones(sims_np.shape[0], dtype=bool)
        for b in range(sims_np.shape[0]):
            # Tokenize this sim's grid data (simplified — just flatten patches)
            sim_flat = sims_np[b].reshape(-1)
            byte_data = sim_flat.astype(np.uint8).tobytes()
            ratio = _gzip_complexity(byte_data)
            if ratio < filter_threshold or ratio > filter_upper:
                keep_mask[b] = False
        kept = sims_np[keep_mask]
        if len(kept) > 0:
            sims_np = kept
        # If filtering removes too many, just use all (fallback)

    return sims_np


def nca_tokenize(sims: np.ndarray, patch: int, num_colors: int, seq_len: int,
                 min_grid: int = 1):
    """Tokenize NCA grid simulations into token sequences.

    Faithfully follows the NCA_Tokenizer + NCADataset from the reference repo.

    CRITICAL: NCADataset.__getitem__ builds targets from seq (NOT from encode_task's
    target output). Line 101: `target = torch.where(seq < 0, -100, seq)` — since all
    tokens are non-negative, target == seq. Then it masks first min_grid timesteps.
    The model IS trained to predict start/end tokens (they are not masked).

    Pipeline:
    1. NCA_Tokenizer.encode_task: grid → patches → base-10 encoding → add start/end
    2. NCADataset.__getitem__: target = seq (with min_grid masking), then shift by 1
    3. Loss: CrossEntropyLoss() with default ignore_index=-100

    Reference: nca-pre-pretraining/utils/tokenizers.py (NCA_Tokenizer.encode_task)
               nca-pre-pretraining/src/nca_ppt.py (NCADataset.__getitem__ lines 101-107)

    Args:
        sims: (num_sims, num_timesteps, grid_H, grid_W) integer grids
        patch: patch size (e.g. 2 means 2×2 patches)
        num_colors: number of cell states
        seq_len: maximum sequence length to produce
        min_grid: number of initial timesteps to mask in targets (ICL context)

    Returns:
        input_ids: (num_seqs, seq_len) int64 tensor — input tokens
        target_ids: (num_seqs, seq_len) int64 tensor — shifted targets with -100 on first timesteps
        vocab_size: int — total vocabulary size (patch_vocab + start + end)
    """
    B, T, H, W = sims.shape
    N_H = H // patch
    N_W = W // patch
    grid_len = N_H * N_W + 2  # tokens per timestep (patches + start + end)

    # Reshape into patches: (B, T, N_H, N_W, patch, patch) → (B, T, N_H*N_W, patch*patch)
    grids = sims.reshape(B, T, N_H, patch, N_W, patch)
    grids = grids.transpose(0, 1, 2, 4, 3, 5)
    grids = grids.reshape(B, T, N_H * N_W, patch * patch)

    # Encode patches as tokens: base-num_colors encoding (matching NCA_Tokenizer)
    powers = num_colors ** np.arange(patch * patch)
    patch_tokens = np.einsum('btlp,p->btl', grids, powers)  # (B, T, N_H*N_W)

    # Special tokens (matching NCA_Tokenizer)
    start_tk = num_colors ** (patch ** 2)
    end_tk = start_tk + 1
    vocab_size = end_tk + 1

    # Build token sequence: [START, patch1, ..., patch36, END] per timestep
    start_col = np.full((B, T, 1), start_tk, dtype=patch_tokens.dtype)
    end_col = np.full((B, T, 1), end_tk, dtype=patch_tokens.dtype)
    tokens = np.concatenate([start_col, patch_tokens, end_col], axis=-1)  # (B, T, grid_len)

    # Flatten timesteps → (B, T*grid_len)
    flat_tokens = tokens.reshape(B, -1)

    # Build targets: same as tokens (NCADataset.__getitem__ line 101:
    #   target = torch.where(seq < 0, -100, seq)  — all tokens are ≥0, so target == seq)
    flat_targets = flat_tokens.copy()

    # Mask first min_grid timesteps (NCADataset.__getitem__ line 102:
    #   target[:(min_grid*grid_len)] = -100)
    MASK = -100
    flat_targets[:, :min_grid * grid_len] = MASK

    # Shift to create next-token prediction pairs (NCADataset.__getitem__ lines 105-106:
    #   seq = seq[:-1], targets = target[1:])
    all_inputs = []
    all_targets = []
    for i in range(B):
        inp_seq = flat_tokens[i, :-1]    # seq[:-1]
        tgt_seq = flat_targets[i, 1:]    # target[1:]
        total_len = len(inp_seq)

        # Split into chunks of seq_len (matching NCADataset truncation)
        n_chunks = max(total_len // seq_len, 1)
        for c in range(n_chunks):
            start = c * seq_len
            end = start + seq_len
            if end > total_len:
                break
            all_inputs.append(inp_seq[start:end])
            all_targets.append(tgt_seq[start:end])

    if not all_inputs:
        raise ValueError("No valid sequences generated from NCA data")

    input_ids = torch.tensor(np.stack(all_inputs), dtype=torch.int64)
    target_ids = torch.tensor(np.stack(all_targets), dtype=torch.int64)
    return input_ids, target_ids, vocab_size


def run_ppt_phase(
    args: Hyperparameters,
    base_model: GPT,
    device: torch.device,
    rank: int,
    world_size: int,
    log_fn,
) -> None:
    """Run NCA Pre-Pre-Training phase.

    1. Generate NCA simulation data (JAX, CPU)
    2. Tokenize into sequences
    3. Temporarily swap embeddings for NCA vocab
    4. Train with Adam + next-token prediction
    5. Save block weights, restore text embeddings
    """
    log_fn("=" * 60)
    log_fn("PPT: Starting NCA Pre-Pre-Training phase")
    log_fn(f"PPT: iterations={args.ppt_iterations} lr={args.ppt_lr} batch_size={args.ppt_batch_size}")
    log_fn(f"PPT: grid={args.ppt_grid} patch={args.ppt_patch} colors={args.ppt_num_colors}")
    log_fn(f"PPT: num_rules={args.ppt_num_rules} num_sims={args.ppt_num_sims}")
    log_fn(f"PPT: filter_threshold={args.ppt_filter_threshold} dT={args.ppt_dT}")
    log_fn("=" * 60)

    t_ppt_start = time.perf_counter()

    # ---- Step 1: Generate NCA data ----
    log_fn("PPT: Generating NCA simulation data (JAX, CPU)...")
    grid_len = (args.ppt_grid // args.ppt_patch) ** 2 + 2  # tokens per timestep
    num_examples = int(math.ceil(args.ppt_seq_len / grid_len))

    sims = generate_nca_data_jax(
        seed=args.seed + rank,  # Different seed per rank for diversity
        num_sims=args.ppt_num_sims,
        grid=args.ppt_grid,
        d_state=args.ppt_num_colors,
        identity_bias=args.ppt_identity_bias,
        temperature=args.ppt_temperature,
        num_examples=num_examples,
        dT=args.ppt_dT,
        start_step=args.ppt_init_rollout,
        num_rules=args.ppt_num_rules,
        filter_threshold=args.ppt_filter_threshold,
        filter_upper=args.ppt_filter_upper,
    )
    log_fn(f"PPT: Generated {sims.shape[0]} simulations, shape={sims.shape}")

    # ---- Step 2: Tokenize ----
    input_ids, target_ids, nca_vocab_size = nca_tokenize(
        sims, args.ppt_patch, args.ppt_num_colors, args.ppt_seq_len
    )
    log_fn(f"PPT: Tokenized into {input_ids.shape[0]} sequences of length {input_ids.shape[1]}")
    log_fn(f"PPT: NCA vocab_size={nca_vocab_size}")

    # ---- Step 3: Swap embeddings for NCA vocab ----
    model_dim = args.model_dim
    # Save original text embeddings
    orig_tok_emb_weight = base_model.tok_emb.weight.data.clone()
    orig_lm_head_weight = None
    if base_model.lm_head is not None:
        orig_lm_head_weight = base_model.lm_head.weight.data.clone()

    # Create temporary NCA embeddings
    nca_tok_emb = nn.Embedding(nca_vocab_size, model_dim).to(device)
    nn.init.normal_(nca_tok_emb.weight, mean=0.0, std=0.02)
    # For tied embeddings, we use a separate lm_head during PPT
    nca_lm_head = nn.Linear(model_dim, nca_vocab_size, bias=False).to(device)
    nn.init.normal_(nca_lm_head.weight, mean=0.0, std=0.02)

    # Replace in the model
    base_model.tok_emb = nca_tok_emb
    # During PPT, we don't use the tied embedding logic — use explicit lm_head
    _orig_tie_embeddings = base_model.tie_embeddings
    _orig_lm_head = base_model.lm_head
    base_model.tie_embeddings = False
    base_model.lm_head = nca_lm_head

    # ---- Step 4: PPT Training Loop ----
    # Collect all model parameters for Adam optimizer
    ppt_params = list(base_model.parameters())
    ppt_optimizer = torch.optim.Adam(ppt_params, lr=args.ppt_lr, betas=(0.9, 0.95))

    warmup_steps = max(int(args.ppt_iterations * args.ppt_warmup_frac), 1)
    dataset_size = input_ids.shape[0]

    base_model.train()
    base_model.set_recurrence_active(False)  # No recurrence during PPT

    ppt_loss_sum = 0.0
    ppt_loss_count = 0

    # Data regeneration: following the official repo's --generate_train flag,
    # we regenerate fresh NCA rules/sims periodically to avoid overfitting.
    # Official regenerates every epoch (1000 steps); we do every regen_interval steps.
    regen_interval = max(dataset_size // args.ppt_batch_size, 100)  # ~1 epoch through data
    next_regen_step = regen_interval
    regen_seed = args.seed + rank + 1000  # Different from initial generation

    for ppt_step in range(args.ppt_iterations):
        # Regenerate NCA data periodically (matching --generate_train --generate_rules 1)
        if ppt_step == next_regen_step:
            regen_seed += 1
            log_fn(f"PPT: Regenerating NCA data at step {ppt_step} (seed={regen_seed})")
            sims = generate_nca_data_jax(
                seed=regen_seed,
                num_sims=args.ppt_num_sims,
                grid=args.ppt_grid,
                d_state=args.ppt_num_colors,
                identity_bias=args.ppt_identity_bias,
                temperature=args.ppt_temperature,
                num_examples=num_examples,
                dT=args.ppt_dT,
                start_step=args.ppt_init_rollout,
                num_rules=args.ppt_num_rules,
                filter_threshold=args.ppt_filter_threshold,
                filter_upper=args.ppt_filter_upper,
            )
            input_ids, target_ids, _ = nca_tokenize(
                sims, args.ppt_patch, args.ppt_num_colors, args.ppt_seq_len
            )
            dataset_size = input_ids.shape[0]
            next_regen_step = ppt_step + regen_interval
            del sims

        # LR schedule: linear warmup → cosine decay (matching official repo's get_lr_scheduler)
        if ppt_step < warmup_steps:
            lr_scale = float(ppt_step + 1) / float(max(1, warmup_steps))
        else:
            progress = float(ppt_step - warmup_steps) / float(max(1, args.ppt_iterations - warmup_steps))
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg in ppt_optimizer.param_groups:
            pg['lr'] = args.ppt_lr * lr_scale

        # Sample a random batch
        batch_idx = torch.randint(0, dataset_size, (args.ppt_batch_size,))
        x = input_ids[batch_idx].to(device)
        y = target_ids[batch_idx].to(device)

        ppt_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            # Can't use base_model(x, y) directly because GPT.forward() uses
            # F.cross_entropy without ignore_index=-100, and our NCA targets
            # contain -100 masked positions. Compute forward + loss manually.
            bsz_ppt, seqlen_ppt = x.shape
            h = base_model.tok_emb(x)
            h = F.rms_norm(h, (h.size(-1),))
            h0 = h
            ppt_skips: list[Tensor] = []
            ppt_v2p = base_model.v2p
            for vi in range(base_model.num_encoder_layers):
                h = base_model.blocks[ppt_v2p[vi]](h, h0)
                ppt_skips.append(h)
            for vi in range(base_model.num_decoder_layers):
                if ppt_skips:
                    h = h + base_model.skip_weights[vi].to(dtype=h.dtype)[None, None, :] * ppt_skips.pop()
                h = base_model.blocks[ppt_v2p[base_model.num_encoder_layers + vi]](h, h0)
            h = base_model.final_norm(h)
            logits = base_model._compute_logits(h.reshape(-1, h.size(-1)))
            # CrossEntropyLoss with ignore_index=-100 (matching original repo's criterion)
            loss = F.cross_entropy(logits.float(), y.reshape(-1), ignore_index=-100, reduction="mean")
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(ppt_params, 1.0)
        ppt_optimizer.step()

        ppt_loss_sum += loss.item()
        ppt_loss_count += 1

        if (ppt_step + 1) % args.ppt_log_every == 0 or ppt_step == 0:
            avg_loss = ppt_loss_sum / ppt_loss_count
            elapsed = time.perf_counter() - t_ppt_start
            log_fn(
                f"PPT step:{ppt_step + 1}/{args.ppt_iterations} "
                f"loss:{loss.item():.4f} avg_loss:{avg_loss:.4f} "
                f"lr:{args.ppt_lr * lr_scale:.6f} "
                f"elapsed:{elapsed:.1f}s"
            )
            ppt_loss_sum = 0.0
            ppt_loss_count = 0

    # ---- Step 5: Restore text embeddings ----
    # Save the trained block weights (they are already updated in-place)
    # Only need to restore the embedding layers

    # Create fresh text embeddings (re-initialized)
    text_tok_emb = nn.Embedding(args.vocab_size, model_dim).to(device)
    if _orig_tie_embeddings:
        nn.init.normal_(text_tok_emb.weight, mean=0.0, std=args.tied_embed_init_std)
    else:
        nn.init.normal_(text_tok_emb.weight, mean=0.0, std=0.02)

    base_model.tok_emb = text_tok_emb
    base_model.tie_embeddings = _orig_tie_embeddings

    if _orig_tie_embeddings:
        base_model.lm_head = None  # Tied: use tok_emb.weight for logits
    else:
        # Create fresh lm_head for text
        text_lm_head = CastedLinear(model_dim, args.vocab_size, bias=False).to(device)
        nn.init.zeros_(text_lm_head.weight)
        text_lm_head._zero_init = True
        base_model.lm_head = text_lm_head

    # Restore low-dim params to fp32 (PPT may have shifted dtypes)
    restore_low_dim_params_to_fp32(base_model)

    # Average weights across ranks (each rank trained on different NCA rules/seeds).
    # This is like federated PPT — the averaged model benefits from 8× data diversity.
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        for p in base_model.parameters():
            dist.all_reduce(p.data, op=dist.ReduceOp.SUM)
            p.data /= world_size

    elapsed_total = time.perf_counter() - t_ppt_start
    n_block_params = sum(p.numel() for p in base_model.blocks.parameters())
    log_fn(f"PPT: Complete! Elapsed={elapsed_total:.1f}s")
    log_fn(f"PPT: Transferred {n_block_params:,} block parameters to text model")
    log_fn(f"PPT: Re-initialized embeddings for text vocab_size={args.vocab_size}")
    log_fn("=" * 60)

    # Clean up NCA data from GPU/CPU memory
    del input_ids, target_ids, nca_tok_emb, nca_lm_head, ppt_optimizer
    torch.cuda.empty_cache()


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5
    global zeropower_via_gram_newton_schulz

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
    zeropower_via_gram_newton_schulz = torch.compile(zeropower_via_gram_newton_schulz)

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl")
        if world_size > 1:
            dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    # Parse depth recurrence layer list from comma-separated string.
    recur_layers: list[int] = []
    if args.recur_layers:
        recur_layers = sorted(set(int(x.strip()) for x in args.recur_layers.split(",") if x.strip()))

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        recur_layers=recur_layers,
        mtp_num_depths=args.mtp_num_depths,
        mtp_loss_weight=args.mtp_loss_weight,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    # -----------------------------
    # NCA PRE-PRE-TRAINING PHASE
    # -----------------------------
    # PPT trains on NCA data, then we extract block weights and load them into
    # a FRESH model for torch.compile. This avoids the issue where replacing
    # nn.Embedding/lm_head in-place causes torch.compile to produce suboptimal graphs.
    if args.ppt_enable:
        log0(f"ppt:enabled iterations={args.ppt_iterations} lr={args.ppt_lr}")
        run_ppt_phase(args, base_model, device, rank, world_size, log0)

        # Extract PPT-trained block weights and create a completely fresh model.
        # This ensures torch.compile sees an unmodified model for optimal graph caching.
        ppt_block_sd = {}
        for name, param in base_model.state_dict().items():
            if name.startswith("blocks.") or name.startswith("final_norm.") or name == "skip_weights":
                ppt_block_sd[name] = param.clone()
        log0(f"ppt:extracted {len(ppt_block_sd)} block weight tensors for transfer")

        # Create fresh model (identical architecture, clean Python objects)
        del base_model
        torch.cuda.empty_cache()
        base_model = GPT(
            vocab_size=args.vocab_size,
            num_layers=args.num_layers,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            mlp_mult=args.mlp_mult,
            tie_embeddings=args.tie_embeddings,
            tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap,
            rope_base=args.rope_base,
            qk_gain_init=args.qk_gain_init,
            recur_layers=recur_layers,
            mtp_num_depths=args.mtp_num_depths,
            mtp_loss_weight=args.mtp_loss_weight,
        ).to(device).bfloat16()
        for module in base_model.modules():
            if isinstance(module, CastedLinear):
                module.float()
        restore_low_dim_params_to_fp32(base_model)

        # Load PPT-trained block weights into the fresh model
        base_model.load_state_dict(ppt_block_sd, strict=False)
        log0(f"ppt:loaded block weights into fresh model")
        del ppt_block_sd
    else:
        log0("ppt:disabled")

    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed and world_size > 1 else compiled_model

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    # MTP module params: projections + shared block + norm (training-only, use Adam with scalar_lr).
    mtp_params: list[nn.Parameter] = []
    for mod in [base_model.mtp_projections, base_model.mtp_shared_block, base_model.mtp_norm]:
        if mod is not None:
            mtp_params.extend(p for p in mod.parameters())
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    # Select Muon variant based on MUON_NS_ALGORITHM env var
    use_gns = args.muon_ns_algorithm == "gram"
    if use_gns:
        gns_coeff_name = args.muon_gns_coefficients
        gns_coefficients = GNS_COEFFICIENT_PRESETS.get(gns_coeff_name)
        if gns_coefficients is None:
            raise ValueError(
                f"Unknown MUON_GNS_COEFFICIENTS={gns_coeff_name!r}, "
                f"choices: {list(GNS_COEFFICIENT_PRESETS.keys())}"
            )
        gns_adjust = args.muon_gns_adjust_lr if args.muon_gns_adjust_lr != "none" else "none"
        optimizer_muon = MuonGNS(
            matrix_params,
            lr=args.matrix_lr,
            momentum=args.muon_momentum,
            weight_decay=args.muon_gns_weight_decay,
            ns_coefficients=gns_coefficients,
            adjust_lr=gns_adjust,
        )
    else:
        optimizer_muon = Muon(
            matrix_params,
            lr=args.matrix_lr,
            momentum=args.muon_momentum,
            backend_steps=args.muon_backend_steps,
        )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)
    if mtp_params:
        optimizer_mtp = torch.optim.Adam(
            [{"params": mtp_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.append(optimizer_mtp)

    n_params = sum(p.numel() for p in base_model.parameters())
    n_mtp_params = sum(p.numel() for p in mtp_params)
    log0(f"model_params:{n_params} (backbone:{n_params - n_mtp_params} mtp_training_only:{n_mtp_params})")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")
    if use_gns:
        log0(
            f"muon_ns:gram_newton_schulz coefficients:{args.muon_gns_coefficients} "
            f"adjust_lr:{args.muon_gns_adjust_lr} weight_decay:{args.muon_gns_weight_decay}"
        )
    else:
        log0(f"muon_ns:standard backend_steps:{args.muon_backend_steps}")
    if recur_layers:
        log0(
            f"depth_recurrence:layers={recur_layers} start_frac={args.recur_start_frac} "
            f"v2p_recur={base_model._v2p_recur} virtual_layers={len(base_model._v2p_recur)} "
            f"physical_layers={base_model.num_physical_layers}"
        )
    else:
        log0("depth_recurrence:disabled")
    if args.mtp_num_depths > 0:
        log0(
            f"mtp:depths={args.mtp_num_depths} loss_weight={args.mtp_loss_weight} "
            f"training_only_params={n_mtp_params}"
        )
    else:
        log0("mtp:disabled")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    # When depth recurrence is configured, we warmup BOTH modes (non-recur and recur)
    # so torch.compile caches both graph variants. This is critical for mid-training
    # activation — without it, the compiled graph ignores the v2p/encoder/decoder changes.
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]

        def _run_warmup(label: str, n_steps: int) -> None:
            model.train()
            for warmup_step in range(n_steps):
                zero_grad_all()
                for micro_step in range(grad_accum_steps):
                    if distributed:
                        model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                    x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                        warmup_loss = model(x, y)
                    (warmup_loss * grad_scale).backward()
                for opt in optimizers:
                    opt.step()
                zero_grad_all()
                if n_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == n_steps:
                    log0(f"warmup_step:{warmup_step + 1}/{n_steps} ({label})")

        # Phase 1: warmup non-recur mode (default)
        base_model.set_recurrence_active(False)
        _run_warmup("base", args.warmup_steps)
        # Phase 2: if recurrence configured, also warmup recur mode to cache its compiled graph
        if recur_layers:
            base_model.set_recurrence_active(True)
            _run_warmup("recur", args.warmup_steps)
            base_model.set_recurrence_active(False)
            log0(f"warmup:both_modes_primed virtual_layers_recur={len(base_model._v2p_recur)}")

        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        # Activate depth recurrence mid-training (wallclock-based).
        # Training the cheaper non-recurrent model first, then switching to recurrent
        # mode avoids the initial loss spike and lets us train more steps cheaply.
        if recur_layers and not base_model._recurrence_active:
            elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
            frac_elapsed = elapsed_ms / max_wallclock_ms if max_wallclock_ms else step / max(args.iterations, 1)
            if frac_elapsed >= args.recur_start_frac:
                base_model.set_recurrence_active(True)
                log0(
                    f"depth_recurrence:activated step:{step} frac:{frac_elapsed:.3f} "
                    f"virtual_layers:{len(base_model.v2p)} v2p={base_model.v2p}"
                )

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and world_size > 1 and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
    # the compressed int8+zlib artifact and validate the round-tripped weights.

    # Strip MTP training-only params before export (zero artifact cost).
    MTP_EXPORT_EXCLUDE = ("mtp_projections.", "mtp_shared_block.", "mtp_norm.")
    full_sd = base_model.state_dict()
    export_sd = {k: v for k, v in full_sd.items() if not any(k.startswith(p) for p in MTP_EXPORT_EXCLUDE)}
    excluded_mtp_params = sum(v.numel() for k, v in full_sd.items() if any(k.startswith(p) for p in MTP_EXPORT_EXCLUDE))
    if excluded_mtp_params > 0:
        log0(f"export:excluding {excluded_mtp_params} MTP training-only params")

    if master_process:
        torch.save(export_sd, "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(export_sd)
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed and world_size > 1:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    # strict=False: MTP training-only params are excluded from the export artifact.
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=(excluded_mtp_params == 0))
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
