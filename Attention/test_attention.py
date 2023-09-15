import os
import sys
import torch
import argparse
import torch.utils.benchmark as benchmark
from torch.backends.cuda import sdp_kernel, SDPBackend
import torch.nn.functional as F
# Lets define a helpful benchmarking function:


def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return round(t0.blocked_autorange().mean * 1e6, 4)


def main(args):
    batch_size = args.batch_size
    max_sequence_len = args.seq_len
    num_heads = args.num_heads
    #embed_dimension = max_sequence_len // num_heads
    embed_dimension = args.emb_dim
    print(
        f"batch_size:{batch_size}, seqlen:{max_sequence_len}, emb_dim:{embed_dimension}")

    dtype = torch.float16
    device = "cuda"
    max_time = 0
    min_time = sys.maxsize
    try:
        query = torch.rand(batch_size, num_heads, max_sequence_len,
                           embed_dimension, device=device, dtype=dtype)
        key = torch.rand(batch_size, num_heads, max_sequence_len,
                         embed_dimension, device=device, dtype=dtype)
        value = torch.rand(batch_size, num_heads, max_sequence_len,
                           embed_dimension, device=device, dtype=dtype)

        default_time = benchmark_torch_function_in_microseconds(
            F.scaled_dot_product_attention, query, key, value)
        # Lets explore the speed of each of the 3 implementations
        # Helpful arg mapper
        backend_map = {
            SDPBackend.MATH: {"enable_math": True, "enable_flash": False, "enable_mem_efficient": False},
            SDPBackend.FLASH_ATTENTION: {"enable_math": False, "enable_flash": True, "enable_mem_efficient": False},
            SDPBackend.EFFICIENT_ATTENTION: {
                "enable_math": False, "enable_flash": False, "enable_mem_efficient": True}
        }

        math_time = "-"
        with sdp_kernel(**backend_map[SDPBackend.MATH]):
            math_time = benchmark_torch_function_in_microseconds(
                F.scaled_dot_product_attention, query, key, value)
            max_time = max(max_time, math_time)
            min_time = min(min_time, math_time)

        flash_time = "-"
        with sdp_kernel(**backend_map[SDPBackend.FLASH_ATTENTION]):
            try:
                flash_time = benchmark_torch_function_in_microseconds(
                    F.scaled_dot_product_attention, query, key, value)
                max_time = max(max_time, flash_time)
                min_time = min(min_time, flash_time)
            except RuntimeError:
                print("FlashAttention is not supported. See warnings for reasons.")
        memory_efficient_time = "-"
        with sdp_kernel(**backend_map[SDPBackend.EFFICIENT_ATTENTION]):
            try:
                memory_efficient_time = benchmark_torch_function_in_microseconds(
                    F.scaled_dot_product_attention, query, key, value)
                max_time = max(max_time, memory_efficient_time)
                min_time = min(min_time, memory_efficient_time)
            except RuntimeError:
                print("EfficientAttention is not supported. See warnings for reasons.")
    except torch.cuda.OutOfMemoryError:
        print(
            f"batch_size:{batch_size}, seq_len:{max_sequence_len}, num_heads:{num_heads} OutofMemory")
        return

    # [batch, num_head, seqlength, embeding_dim] -> [batch, seqlength, num_head, embeding_dim]
    before_shape = query.shape
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    print(before_shape, "->", query.shape)
    flash_att2_time = "-"
    flash_att_varlen_time = "-"
    try:
        from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
        # q: (batch_size, seqlen, nheads, headdim)
        # k: (batch_size, seqlen, nheads_k, headdim)
        # v: (batch_size, seqlen, nheads_k, headdim)
        flash_att2_time = benchmark_torch_function_in_microseconds(
            flash_attn_func, query, key, value, dropout_p=0.0, softmax_scale=None, causal=False)
        max_time = max(max_time, flash_att2_time)
        min_time = min(min_time, flash_att2_time)

        from einops import rearrange

        def flash_varlen(q, k, v, device):
            batch_size, seqlen_q = q.shape[0], q.shape[1]
            q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
            cu_seqlens_k = cu_seqlens_q = torch.arange(
                0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=device)
            output = flash_attn_varlen_func(
                q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_q, causal=True)
            return output

        flash_att_varlen_time = benchmark_torch_function_in_microseconds(
            flash_varlen, query, key, value, device)
    except RuntimeError as e:
        print(e)
        print("flash attention is not supported. See warnings for reasons.")
    xformers_time = "-"
    try:
        import xformers.ops as xops

        xformers_time = benchmark_torch_function_in_microseconds(
            xops.memory_efficient_attention, query, key, value)
        max_time = max(max_time, xformers_time)
        min_time = min(min_time, xformers_time)
    except RuntimeError:
        print("xformers is not supported. See warnings for reasons.")
    # main(args)
    gpu_type = os.popen(
        "nvidia-smi -i 0 -q | grep 'Product Name' | awk -F ' : ' '{ print $2 }'").readlines()[0].strip()
    print(f"gpu_type, batchsize-seqlen-head_n-emb_dim, torch-math(ms), torch-flashatt(ms), torch-mem_efficient(ms), flashatt2(ms), xformers(ms), speedup")
    print(f"{gpu_type}, {batch_size}-{max_sequence_len}-{num_heads}-{embed_dimension}, {math_time}, {flash_time}, {memory_efficient_time}, {flash_att2_time}, {xformers_time}, {max_time/min_time:.4f}")
    print(f"Default implement runtime {default_time:.4f} ms")
    print(
        f"flashattention varlen implement runtime {flash_att_varlen_time} ms")
    # print(f"xformers implement runtime {xformers_time} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="batch_size")
    parser.add_argument("--seq_len", type=int,
                        default=1024, help="sequence length")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="the number of heads")
    parser.add_argument("--emb_dim", type=int, default=64,
                        help="the number of heads")
    args = parser.parse_args()
    main(args)
