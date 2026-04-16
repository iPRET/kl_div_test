"""
Test: verify that ChunkedKLDivFunction (custom autograd with analytical gradient)
produces the same output and gradients as the original LogitsKLLoss from NeMo.

Run with torchrun on 4 GPUs (one node, TP=4):
    torchrun --nproc_per_node 4 test_kl_loss.py

What this test does:
    1. Initializes TP=4 using Megatron's parallel_state.
    2. Creates random student logits (requires_grad) and teacher logits (detached).
    3. Runs ORIGINAL forward + backward -> gets loss and gradient.
    4. Runs CHUNKED (custom autograd) forward + backward -> gets loss and gradient.
    5. Compares losses and gradients are numerically close.
    6. Prints PASS or FAIL.

How the chunked version saves memory:
    Forward:  compute loss under no_grad, chunked along sequence dim. Save only inputs.
    Backward: for each chunk, recompute log_probs (under no_grad), then compute
              the gradient analytically:  grad = student_softmax - teacher_softmax.
              No autograd or nested backward calls needed.
    Peak memory = one chunk's worth of (chunk, batch, vocab) tensors at a time.

Why the analytical gradient works:
    KL(q || p) = sum_v q_v * (log q_v - log p_v)
    d(KL)/d(z_v) = softmax(z)_v - q_v = p_v - q_v
    where z = student logits (input to log_softmax), p = student probs, q = teacher probs.
    This holds per-position, and with TP each rank computes its local shard.
"""

import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from megatron.core import parallel_state


# ──────────────────────────────────────────────────────────────
# Helper: autograd-safe all-reduce (copied from NeMo loss.py)
# ──────────────────────────────────────────────────────────────
class _AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, op, group, tensor):
        ctx.group, ctx.op = group, op
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=op, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, _AllReduce.apply(ctx.op, ctx.group, grad_output))


def all_reduce_autograd(tensor, op=dist.ReduceOp.SUM, group=dist.group.WORLD):
    return _AllReduce.apply(op, group, tensor)


# ──────────────────────────────────────────────────────────────
# ORIGINAL: LogitsKLLoss forward (from NeMo, unchanged)
# ──────────────────────────────────────────────────────────────
def original_kl_loss(output_student, output_teacher, tp_group):
    """
    Original NeMo KL loss for TP > 1.
    Inputs: fp32 logits, already temperature-scaled.
    Returns per-token loss: shape (seq, batch).
    """
    # Teacher softmax
    teacher_logits_max, _ = torch.max(output_teacher, dim=-1)
    dist.all_reduce(teacher_logits_max, op=dist.ReduceOp.MAX, group=tp_group)
    output_teacher = output_teacher - teacher_logits_max.unsqueeze(dim=-1)

    denom_teacher = torch.sum(torch.exp(output_teacher), dim=-1)
    denom_teacher = all_reduce_autograd(denom_teacher, group=tp_group)

    # Student softmax
    student_logits_max, _ = torch.max(output_student, dim=-1)
    dist.all_reduce(student_logits_max, op=dist.ReduceOp.MAX, group=tp_group)
    output_student = output_student - student_logits_max.unsqueeze(dim=-1).detach()

    denom_student = torch.sum(torch.exp(output_student), dim=-1)
    denom_student = all_reduce_autograd(denom_student, group=tp_group)

    slen, bsz, sharded_vocab_size = output_student.shape
    student_log_prob = output_student - torch.log(denom_student).view(slen, bsz, 1).expand(
        slen, bsz, sharded_vocab_size
    )
    teacher_log_prob = output_teacher - torch.log(denom_teacher).view(slen, bsz, 1).expand(
        slen, bsz, sharded_vocab_size
    )

    loss = torch.sum(
        F.kl_div(student_log_prob, teacher_log_prob, reduction="none", log_target=True),
        dim=-1,
    )
    return loss


# ──────────────────────────────────────────────────────────────
# CHUNKED: custom autograd with analytical gradient
# ──────────────────────────────────────────────────────────────

def _compute_log_probs(student_chunk, teacher_chunk, tp_group):
    """
    Compute student and teacher log-softmax for one chunk, using TP all-reduces.
    No autograd — called under no_grad in both forward and backward.

    Args:
        student_chunk: (chunk, batch, sharded_vocab) fp32
        teacher_chunk: (chunk, batch, sharded_vocab) fp32
        tp_group: tensor parallel process group

    Returns:
        student_log_prob: (chunk, batch, sharded_vocab) fp32
        teacher_log_prob: (chunk, batch, sharded_vocab) fp32
    """
    clen, bsz, svocab = student_chunk.shape

    # Teacher log-softmax (numerically stable with TP-global max)
    t_max, _ = torch.max(teacher_chunk, dim=-1)
    dist.all_reduce(t_max, op=dist.ReduceOp.MAX, group=tp_group)
    t_shifted = teacher_chunk - t_max.unsqueeze(-1)
    t_denom = torch.sum(torch.exp(t_shifted), dim=-1)
    dist.all_reduce(t_denom, op=dist.ReduceOp.SUM, group=tp_group)
    t_log_prob = t_shifted - torch.log(t_denom).unsqueeze(-1)

    # Student log-softmax (numerically stable with TP-global max)
    s_max, _ = torch.max(student_chunk, dim=-1)
    dist.all_reduce(s_max, op=dist.ReduceOp.MAX, group=tp_group)
    s_shifted = student_chunk - s_max.unsqueeze(-1)
    s_denom = torch.sum(torch.exp(s_shifted), dim=-1)
    dist.all_reduce(s_denom, op=dist.ReduceOp.SUM, group=tp_group)
    s_log_prob = s_shifted - torch.log(s_denom).unsqueeze(-1)

    return s_log_prob, t_log_prob


class ChunkedKLDivFunction(torch.autograd.Function):
    """
    Custom autograd for chunked KL divergence.

    Forward:  no_grad, compute loss chunk by chunk, save only inputs.
    Backward: no_grad, recompute log_probs chunk by chunk, apply analytical gradient:
              grad = grad_output * (student_softmax - teacher_softmax)
    """

    @staticmethod
    def forward(ctx, output_student, output_teacher, tp_group, chunk_size):
        ctx.save_for_backward(output_student, output_teacher)
        ctx.tp_group = tp_group
        ctx.chunk_size = chunk_size

        slen = output_student.shape[0]
        loss_chunks = []

        with torch.no_grad():
            for start in range(0, slen, chunk_size):
                end = min(start + chunk_size, slen)
                s_log_prob, t_log_prob = _compute_log_probs(
                    output_student[start:end],
                    output_teacher[start:end],
                    tp_group,
                )
                # KL div summed over vocab -> (chunk, batch)
                chunk_loss = torch.sum(
                    F.kl_div(s_log_prob, t_log_prob, reduction="none", log_target=True),
                    dim=-1,
                )
                loss_chunks.append(chunk_loss)
                # s_log_prob, t_log_prob, kl_div result freed here.

        return torch.cat(loss_chunks, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        output_student, output_teacher = ctx.saved_tensors
        tp_group = ctx.tp_group
        chunk_size = ctx.chunk_size

        slen = output_student.shape[0]
        grad_student = torch.empty_like(output_student)

        with torch.no_grad():
            for start in range(0, slen, chunk_size):
                end = min(start + chunk_size, slen)

                # Recompute log_probs for this chunk.
                s_log_prob, t_log_prob = _compute_log_probs(
                    output_student[start:end],
                    output_teacher[start:end],
                    tp_group,
                )

                # Analytical gradient: d(KL)/d(z) = softmax(z) - teacher_probs = p - q
                # Then multiply by upstream grad_output per position.
                grad_chunk = torch.exp(s_log_prob)      # student probs (p)
                grad_chunk -= torch.exp(t_log_prob)      # p - q (in-place)
                grad_chunk *= grad_output[start:end].unsqueeze(-1)  # chain rule

                grad_student[start:end] = grad_chunk
                # s_log_prob, t_log_prob, grad_chunk freed here.

        # grad for: output_student, output_teacher, tp_group, chunk_size
        return grad_student, None, None, None


def chunked_kl_loss(output_student, output_teacher, tp_group, chunk_size=1024):
    """Wrapper that calls ChunkedKLDivFunction.apply."""
    return ChunkedKLDivFunction.apply(output_student, output_teacher, tp_group, chunk_size)


# ──────────────────────────────────────────────────────────────
# Test
# ──────────────────────────────────────────────────────────────
def main():
    # --- Init distributed + TP ---
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=1,
    )
    tp_group = parallel_state.get_tensor_model_parallel_group()
    rank = dist.get_rank()

    # --- Test parameters ---
    # Small sizes so the test runs fast. Math is identical regardless of size.
    seq_len = 4096
    batch = 1
    full_vocab = 131072
    sharded_vocab = full_vocab // 4  # 32768 per TP rank
    chunk_size = 1024

    # --- Create identical inputs for both runs ---
    torch.manual_seed(42 + rank)
    student_logits_base = torch.randn(seq_len, batch, sharded_vocab, device="cuda", dtype=torch.float32)
    teacher_logits_base = torch.randn(seq_len, batch, sharded_vocab, device="cuda", dtype=torch.float32)

    # ──────────────── Run ORIGINAL ────────────────
    student_orig = student_logits_base.clone().requires_grad_(True)
    teacher_orig = teacher_logits_base.clone().detach()

    loss_orig = original_kl_loss(student_orig, teacher_orig, tp_group)
    loss_orig_scalar = loss_orig.sum()
    loss_orig_scalar.backward()
    grad_orig = student_orig.grad.clone()

    # ──────────────── Run CHUNKED ────────────────
    student_chunked = student_logits_base.clone().requires_grad_(True)
    teacher_chunked = teacher_logits_base.clone().detach()

    loss_chunked = chunked_kl_loss(student_chunked, teacher_chunked, tp_group, chunk_size=chunk_size)
    loss_chunked_scalar = loss_chunked.sum()
    loss_chunked_scalar.backward()
    grad_chunked = student_chunked.grad.clone()

    # ──────────────── Compare ────────────────
    loss_close = torch.allclose(loss_orig, loss_chunked, atol=1e-5, rtol=1e-5)
    grad_close = torch.allclose(grad_orig, grad_chunked, atol=1e-5, rtol=1e-5)

    loss_max_diff = (loss_orig - loss_chunked).abs().max().item()
    grad_max_diff = (grad_orig - grad_chunked).abs().max().item()

    if rank == 0:
        print("=" * 60)
        print("KL LOSS CHUNKING TEST (analytical gradient)")
        print("=" * 60)
        print(f"  seq_len={seq_len}, batch={batch}, sharded_vocab={sharded_vocab}")
        print(f"  chunk_size={chunk_size}, TP=4")
        print()
        print(f"  Loss shape:     {loss_orig.shape}")
        print(f"  Loss max diff:  {loss_max_diff:.2e}")
        print(f"  Loss match:     {'YES' if loss_close else 'NO'}")
        print()
        print(f"  Grad shape:     {grad_orig.shape}")
        print(f"  Grad max diff:  {grad_max_diff:.2e}")
        print(f"  Grad match:     {'YES' if grad_close else 'NO'}")
        print()
        if loss_close and grad_close:
            print("  >>> PASS <<<")
        else:
            print("  >>> FAIL <<<")
            if not loss_close:
                print("      Loss values differ beyond tolerance!")
            if not grad_close:
                print("      Gradient values differ beyond tolerance!")
        print("=" * 60)

    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
