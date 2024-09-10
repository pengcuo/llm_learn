import torch
import triton
import flag_attn

NUM_BLOCKS = 1000
warmup = 200
rep = 200
TEST = True

try:
    # from vllm._C import ops as vllm_ops
    from vllm import _custom_ops as vllm_ops

    HAS_VLLM = True

    # required vllm 0.3.0
    import vllm

    print("vllm.__version__", vllm.__version__)
except BaseException:
    HAS_VLLM = False

HAS_VLLM = False

try:
    from flash_attn import flash_attn_func
    FLASH_VER = 2
except BaseException:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func
        FLASH_VER = 1
    except BaseException:
        FLASH_VER = None
HAS_FLASH = FLASH_VER is not None



torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)  # 如果你在使用多 GPU
import numpy as np
np.random.seed(0)

import random
random.seed(0)
import torch.backends.cudnn as cudnn

cudnn.deterministic = True
cudnn.benchmark = False

def vllm_paged_attention(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    max_context_len: int,
    PARTITION_SIZE: int = 512,
    version: int = 1,
):
    if version == 1:
        vllm_ops.paged_attention_v1(
            out,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            None,  # alibi_slopes
            "auto",  # kv_cache_dtype for vllm 0.3.0
            1.
        )
    elif version == 2:
        num_partitions = (max_context_len + PARTITION_SIZE - 1) // PARTITION_SIZE
        assert PARTITION_SIZE % block_size == 0
        num_seqs, num_heads, head_size = out.shape
        tmp_out = torch.empty(
            size=(num_seqs, num_heads, num_partitions, head_size),
            dtype=out.dtype,
            device=out.device,
        )
        exp_sums = torch.empty(
            size=(num_seqs, num_heads, num_partitions),
            dtype=torch.float32,
            device=out.device,
        )
        max_logits = torch.empty_like(exp_sums)
        vllm_ops.paged_attention_v2(
            out,
            exp_sums,
            max_logits,
            tmp_out,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            None,
            "auto",  # vllm 0.3.0
            1.
        )
    else:
        raise AssertionError(f"Unknown version: {version}")


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["context_len"],
            # x_vals=[2**i for i in range(9, 15)],
            x_vals=[2**i for i in range(9, 15)],
            # x_vals=[2 ** 10],
            line_arg="provider",
            line_vals=["triton"] + (["vllm"] if HAS_VLLM else []) + (["flash"] if HAS_FLASH else []),
            # line_vals=["triton"] + (["flash"] if HAS_FLASH else []),
            line_names=["mla(us)"] + ([f"vllm-{vllm.__version__}(us)"] if HAS_VLLM else []) + ([f"flash_attention_v2(us)"] if HAS_FLASH else []),
            styles=[("red", "-"), ("blue", "-")],
            ylabel="tflop/s",
            plot_name=f"vllm_paged_attention-B{num_seqs}-G{query_group_size}-D{head_size}-bs{block_size}-v{version}",
            args={
                "num_seqs": num_seqs,
                "num_query_heads": 16,
                "query_group_size": query_group_size,
                "head_size": head_size,
                "block_size": block_size,
                "vllm_version": version,
                "dtype": dtype,
            },
        )
        # for num_seqs in [2, 4, 8]
        for num_seqs in [4]
        for query_group_size in [1]
        for head_size in [128]
        for block_size in [16]
        for version in [1]
        for dtype in [torch.float16]
    ]
)
def paged_attention_benchmark_with_vllm(
    num_seqs,
    num_query_heads,
    query_group_size,
    head_size,
    block_size,
    context_len,
    vllm_version,
    provider,
    dtype=torch.float16,
    device="cuda",
):
    num_kv_heads = num_query_heads // query_group_size

    context_lens = torch.zeros(num_seqs, dtype=torch.int32, device=device) + context_len
    max_num_blocks_per_seq = (context_len + block_size - 1) // block_size

    q_lora_rank = 256
    rope_size = head_size // 2
    rope_size = 64
    attn_scale = (head_size + rope_size)**-0.5
    q_latent = torch.randn(num_seqs, q_lora_rank, dtype=dtype, device=device)
    weight_q_latent = torch.randn(q_lora_rank, num_query_heads * (head_size + rope_size), dtype=dtype, device=device)
    q = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype, device=device)
    q = (q_latent @ weight_q_latent).view(num_seqs, num_query_heads, head_size + rope_size)

    # q = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype, device=device)

    q.uniform_(-attn_scale, attn_scale)
    print(f"query shape : {q.shape}")

    out = torch.empty_like(q)
    vllm_out = torch.empty_like(q)

    kv_lora_rank = 256

    #[num_blocks, 16, kv_lora_rank]
    kv_latent = torch.rand(NUM_BLOCKS, block_size, kv_lora_rank, dtype=dtype, device=device)
    kv_latent.uniform_(-attn_scale, attn_scale)

    weight_k_latent = torch.randn(kv_lora_rank, num_kv_heads * head_size, dtype=dtype, device=device)
    weight_v_latent = torch.randn(kv_lora_rank, num_kv_heads * head_size, dtype=dtype, device=device)

    k_cache = kv_latent @ weight_k_latent
    v_cache = kv_latent @ weight_v_latent

    rope_k_cache = torch.randn(NUM_BLOCKS, 1, block_size, rope_size, dtype=dtype, device=device)
    # rope_k_cache.uniform_(-attn_scale, attn_scale)
    # print(f"rope_k_cache : {rope_k_cache.shape}")
    repeat_rope_k_cache = rope_k_cache.repeat(1, num_kv_heads, 1, 1)
    # print(f"repeat_rope_k_cache : {repeat_rope_k_cache.shape} {repeat_rope_k_cache.stride(0)}")
    # print(f"repeat_rope_k_cache : {repeat_rope_k_cache.shape} {repeat_rope_k_cache.stride(1)}")
    # print(f"repeat_rope_k_cache : {repeat_rope_k_cache.shape} {repeat_rope_k_cache.stride(2)}")
    # print(f"repeat_rope_k_cache : {repeat_rope_k_cache.shape} {repeat_rope_k_cache.stride(3)}")

    #[num_kv_heads, head_size, kv_lora_rank]
    weight_k_latent = weight_k_latent.reshape(kv_lora_rank, num_kv_heads, head_size).permute(1, 2, 0)
    #[num_kv_heads, kv_lora_rank, head_size]
    weight_v_latent = weight_v_latent.reshape(kv_lora_rank, num_kv_heads, head_size).permute(1, 0, 2)

    # for concat rope
    k_cache = k_cache.reshape(NUM_BLOCKS, block_size, num_kv_heads, head_size).permute(0, 2, 1, 3)
    v_cache = v_cache.reshape(NUM_BLOCKS, block_size, num_kv_heads, head_size)

    print(f"k_cache {k_cache.shape} {context_len}")
    print(f"v_cache {v_cache.shape} {context_len}")
    k_cache = torch.concat((k_cache, repeat_rope_k_cache), dim=-1)
    k_cache = k_cache.permute(0, 2, 1, 3)
    print(f"k_cache_with_rope {k_cache.shape} {context_len}")


    vllm_k_cache = k_cache.clone().permute(0, 2, 1, 3).contiguous()
    row_v_size = (int)(16 / dtype.itemsize)
    vllm_k_cache = vllm_k_cache.view(-1, num_kv_heads, block_size, head_size // row_v_size, row_v_size).permute(0, 1, 3, 2, 4).contiguous()
    vllm_v_cache = v_cache.clone().permute(0, 2, 1, 3)
    vllm_v_cache = vllm_v_cache.permute(0, 1, 3, 2).contiguous()

    # (NUM_SEQS, MAX_NUM_BLOCKS_PER_SEQ)
    block_tables = torch.randint(
        0,
        NUM_BLOCKS,
        (num_seqs, max_num_blocks_per_seq),
        dtype=torch.int32,
        device=device,)

    if TEST:
        fn = lambda: flag_attn.paged_mla_attention(
            q,
            context_lens,
            block_tables,
            attn_scale,
            context_len,
            0,
            kv_latent,
            weight_k_latent,
            weight_v_latent,
            rope_k_cache
        )
        flag_attn_out = fn()

        flag_attn_out_double = flag_attn_out.to(torch.float64).reshape(-1)

        num_kv_blocks = context_len // block_size
        print(f"num_kv_blocks : {num_kv_blocks}")
        real_k = []
        real_v = []
        for s in range(num_seqs):
            bt = block_tables[s]

            buff_k = []
            buff_v = []
            # i = 0
            for idx in bt:
                # print(f"seqs {s} {i}: {idx}")
                # i += 1
                buff_k.append(k_cache[idx])
                buff_v.append(v_cache[idx])
                # print(f"k_cache[idx] : {k_cache[idx].shape}")
                # print(f"v_cache[idx] : {v_cache[idx].shape}")

            buff_k = torch.concat(buff_k)
            buff_v = torch.concat(buff_v)

            # print(f"buff_k {buff_k.shape}")
            # print(f"buff_v {buff_v.shape}")
            real_k.append(buff_k)
            real_v.append(buff_v)


        real_k = torch.concat(real_k).view(num_seqs, context_len, num_kv_heads, head_size + rope_size).to(torch.float64)
        real_v = torch.concat(real_v).view(num_seqs, context_len, num_kv_heads, head_size).to(torch.float64)

        print(f"real_k {real_k.shape}")
        print(f"real_v {real_v.shape}")

        real_v = real_v.permute(0, 2, 1, 3)
        print(f"permute real_v {real_v.shape}")

        check_q = q.clone().view(num_seqs, 1, num_kv_heads, head_size + rope_size).to(torch.float64)
        check_qk = torch.sum(check_q * real_k, axis=3)

        check_qk *= attn_scale
        attn_weight = torch.softmax(check_qk, axis=1)
        attn_weight = attn_weight.permute(0, 2, 1).view(num_seqs, num_kv_heads, 1, context_len)
        check_output = (attn_weight @ real_v).view(num_seqs, num_kv_heads, head_size)

        print(f"\nflag_attn_out result : {flag_attn_out.abs().sum()} {flag_attn_out.shape} {flag_attn_out.dtype}")
        print([f"{num:.5f}" for num in flag_attn_out.view(-1)[:10]])
        print(f"\ncheck_output   result : {check_output.abs().sum()} {check_output.shape} {check_output.dtype}")
        print([f"{num:.5f}" for num in check_output.view(-1)[:10]])

        check_output_double0 = check_output.to(torch.float64).reshape(-1)


        # # DOUBLE CHECK
        # dc_q = q.clone().view(num_seqs, 1, num_kv_heads, head_size + rope_size)[..., :head_size].to(torch.float64)
        # dc_k = real_k[..., :head_size].to(torch.float64)
        # dc_q = dc_q.contiguous()
        # dc_k = dc_k.contiguous()
        # print("@" * 50)
        # print(f"dc_q {dc_q.shape}")
        # print(f"dc_k {dc_k.shape}")
        # print("@" * 50)
        # check_qk = torch.sum(dc_q * dc_k, axis=3)


        # check_qk *= attn_scale
        # attn_weight = torch.softmax(check_qk, axis=1)
        # print(f"attn_weight {attn_weight.shape}")
        # attn_weight = attn_weight.permute(0, 2, 1).view(num_seqs, num_kv_heads, 1, context_len)
        # print(f"permute attn_weight {attn_weight.shape}")

        # check_output = (attn_weight @ real_v).view(num_seqs, num_kv_heads, head_size)

        # print(f"\ncheck_output   result ii : {check_output.abs().sum()} {check_output.shape} {check_output.dtype}")
        # print([f"{num:.5f}" for num in check_output.view(-1)[:10]])
        # check_output_double1 = check_output.to(torch.float64).reshape(-1)

        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        cos0_v = cos(flag_attn_out_double, check_output_double0)
        # cos1_v = cos(flag_attn_out_double, check_output_double1)
        # base_cos = cos(check_output_double0, check_output_double1)
        print(f"similarity cos0_v : {cos0_v}")
        # print(f"similarity cos1_v : {cos1_v}")
        # print(f"similarity base_cos : {base_cos}")

        # END

    if provider == "triton":
        fn = lambda: flag_attn.paged_mla_attention(
            q,
            context_lens,
            block_tables,
            attn_scale,
            context_len,
            0,
            kv_latent,
            weight_k_latent,
            weight_v_latent,
            rope_k_cache
        )
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)


        # print(f"xxx ms {ms}")

    if provider == "vllm":
        # Correctness error, does not affect performance results
        fn = lambda: vllm_paged_attention(
            out,
            q,
            k_cache,
            v_cache,
            num_kv_heads,
            attn_scale,
            block_tables,
            context_lens,
            block_size,
            context_len,
            PARTITION_SIZE=512,
            version=vllm_version,
        )
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        print(f"vllm out shape {out.shape}")

    if provider == "flash":
        BATCH = num_seqs
        H = num_kv_heads
        D_HEAD = head_size
        N_CTX = context_len
        causal = True
        q = torch.randn((BATCH, 1, H, D_HEAD), dtype=dtype, device="cuda")
        k = torch.randn((BATCH, N_CTX, H, D_HEAD), dtype=dtype, device="cuda")
        v = torch.randn((BATCH, N_CTX, H, D_HEAD), dtype=dtype, device="cuda")
        fn = lambda: flash_attn_func(q, k, v, causal=causal)
        ms = triton.testing.do_bench(fn)

    total_flops = 2.0 * num_seqs * num_query_heads * 2 * context_len * head_size
    # return total_flops / ms * 1e-9
    return ms * 1e3 # us


# if HAS_VLLM:
paged_attention_benchmark_with_vllm.run(print_data=True)#
