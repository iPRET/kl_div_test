"""
Test: verify that ChunkedLogitsKLLoss produces the same output and gradients
as the original LogitsKLLoss from NeMo's distillation code.

Run with torchrun on 4 GPUs (one node, TP=4):
    torchrun --nproc_per_node 4 test_kl_loss.py

What this test does:
    1. Initializes TP=4 (tensor parallel) using Megatron's parallel_state.
    2. Creates random student logits (requires_grad=True) and teacher logits (detached).
       Shape: (seq_len, batch, sharded_vocab_size) where sharded_vocab_size = vocab_size / TP.
    3. Runs ORIGINAL LogitsKLLoss.forward() -> gets loss and student gradient.
    4. Runs CHUNKED LogitsKLLoss.forward() -> gets loss and student gradient.
    5. Compares:
       - Losses are close (atol=1e-5, rtol=1e-5)
       - Gradients are close (atol=1e-5, rtol=1e-5)
    6. Prints PASS or FAIL.
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
# ORIGINAL: LogitsKLLoss (from NeMo, unchanged)
# ──────────────────────────────────────────────────────────────
def original_kl_loss(output_student, output_teacher, tp_group):
    """
    Original NeMo KL loss for TP > 1.
    Inputs are already fp32 and temperature-scaled.
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
# CHUNKED: same math, but processes sequence in chunks
# ──────────────────────────────────────────────────────────────
def chunked_kl_loss(output_student, output_teacher, tp_group, chunk_size=1024):
    """
    Chunked KL loss: processes the sequence dimension in slices of chunk_size.
    Each chunk independently computes its softmax (with TP all-reduce) and KL div,
    then sums over vocab (dim=-1) to get a small (chunk, batch) result.
    The big (chunk, batch, vocab) intermediates are freed between chunks.

    Returns per-token loss: shape (seq, batch) -- same as original.
    """
    slen = output_student.shape[0]
    loss_chunks = []

    for start in range(0, slen, chunk_size):
        end = min(start + chunk_size, slen)
        student_chunk = output_student[start:end]  # (chunk, b, vocab) -- this is a view, no copy
        teacher_chunk = output_teacher[start:end]  # (chunk, b, vocab) -- this is a view, no copy

        # Teacher softmax for this chunk
        t_max, _ = torch.max(teacher_chunk, dim=-1)
        dist.all_reduce(t_max, op=dist.ReduceOp.MAX, group=tp_group)
        teacher_shifted = teacher_chunk - t_max.unsqueeze(dim=-1)

        t_denom = torch.sum(torch.exp(teacher_shifted), dim=-1)
        t_denom = all_reduce_autograd(t_denom, group=tp_group)

        clen, bsz, sharded_vocab = student_chunk.shape

        t_log_prob = teacher_shifted - torch.log(t_denom).view(clen, bsz, 1).expand(clen, bsz, sharded_vocab)
        del teacher_shifted, t_denom  # free teacher intermediates

        # Student softmax for this chunk
        s_max, _ = torch.max(student_chunk, dim=-1)
        dist.all_reduce(s_max, op=dist.ReduceOp.MAX, group=tp_group)
        student_shifted = student_chunk - s_max.unsqueeze(dim=-1).detach()

        s_denom = torch.sum(torch.exp(student_shifted), dim=-1)
        s_denom = all_reduce_autograd(s_denom, group=tp_group)

        s_log_prob = student_shifted - torch.log(s_denom).view(clen, bsz, 1).expand(clen, bsz, sharded_vocab)

        # KL div for this chunk -> (chunk, batch)
        chunk_loss = torch.sum(
            F.kl_div(s_log_prob, t_log_prob, reduction="none", log_target=True),
            dim=-1,
        )
        loss_chunks.append(chunk_loss)
        # At this point, s_log_prob, t_log_prob, and the kl_div result (vocab-sized)
        # are freed. Only chunk_loss (chunk, batch) survives.

    return torch.cat(loss_chunks, dim=0)


# ──────────────────────────────────────────────────────────────
# Test
# ──────────────────────────────────────────────────────────────
def main():
    # --- Init distributed + TP ---
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    # Initialize Megatron parallel state with TP=4, PP=1, no other parallelism.
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=1,
    )
    tp_group = parallel_state.get_tensor_model_parallel_group()
    rank = dist.get_rank()

    # --- Test parameters ---
    # Using smaller sizes than production so the test runs fast and doesn't OOM.
    # The math is identical regardless of size.
    seq_len = 4096
    batch = 1
    full_vocab = 131072
    sharded_vocab = full_vocab // 4  # 32768 per TP rank
    chunk_size = 1024

    # --- Create identical inputs for both runs ---
    # We use a fixed seed per rank so each TP rank gets different vocab shards
    # but both runs (original and chunked) see the same data.
    torch.manual_seed(42 + rank)

    student_logits_base = torch.randn(seq_len, batch, sharded_vocab, device="cuda", dtype=torch.float32)
    teacher_logits_base = torch.randn(seq_len, batch, sharded_vocab, device="cuda", dtype=torch.float32)

    # --- Run ORIGINAL ---
    student_orig = student_logits_base.clone().requires_grad_(True)
    teacher_orig = teacher_logits_base.clone().detach()

    loss_orig = original_kl_loss(student_orig, teacher_orig, tp_group)
    loss_orig_scalar = loss_orig.sum()
    loss_orig_scalar.backward()
    grad_orig = student_orig.grad.clone()

    # --- Run CHUNKED ---
    student_chunked = student_logits_base.clone().requires_grad_(True)
    teacher_chunked = teacher_logits_base.clone().detach()

    loss_chunked = chunked_kl_loss(student_chunked, teacher_chunked, tp_group, chunk_size=chunk_size)
    loss_chunked_scalar = loss_chunked.sum()
    loss_chunked_scalar.backward()
    grad_chunked = student_chunked.grad.clone()

    # --- Compare ---
    loss_match = torch.allclose(loss_orig, loss_chunked, atol=1e-5, rtol=1e-5)
    grad_match = torch.allclose(grad_orig, grad_chunked, atol=1e-5, rtol=1e-5)

    loss_max_diff = (loss_orig - loss_chunked).abs().max().item()
    grad_max_diff = (grad_orig - grad_chunked).abs().max().item()

    if rank == 0:
        print("=" * 60)
        print("KL LOSS CHUNKING TEST")
        print("=" * 60)
        print(f"  seq_len={seq_len}, batch={batch}, sharded_vocab={sharded_vocab}")
        print(f"  chunk_size={chunk_size}, TP=4")
        print()
        print(f"  Loss shape:     {loss_orig.shape}")
        print(f"  Loss max diff:  {loss_max_diff:.2e}")
        print(f"  Loss match:     {'YES' if loss_match else 'NO'}")
        print()
        print(f"  Grad shape:     {grad_orig.shape}")
        print(f"  Grad max diff:  {grad_max_diff:.2e}")
        print(f"  Grad match:     {'YES' if grad_match else 'NO'}")
        print()
        if loss_match and grad_match:
            print("  RESULT: PASS")
        else:
            print("  RESULT: FAIL")
        print("=" * 60)

    # Cleanup
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
