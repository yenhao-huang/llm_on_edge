optimum-cli export executorch \
    --model "/Users/yenhaohuang/Desktop/models/qwen3-0.6b/" \
    --task "text-generation" \
    --recipe "xnnpack" \
    --use_custom_sdpa \
    --use_custom_kv_cache \
    --qlinear 8da4w \
    --qembedding 8w \
    --output_dir="qwen3_0.6b_executorch"