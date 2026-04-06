cd executorch
python -m extension.llm.export.export_llm \
  --config examples/models/qwen3_5/config/qwen3_5_xnnpack_bp16.yaml \
  +base.model_class="qwen3_5_0_8b" \
  +base.params="examples/models/qwen3_5/config/0_8b_config.json" \
  +export.output_name="qwen3_5_0_8b_fp32.pte"