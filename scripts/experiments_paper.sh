#!/bin/zsh

# ----- acip_compress ------
python -m acip.entrypoints.acip_compress experiment=paper/llama1_7b/compress
python -m acip.entrypoints.acip_compress experiment=paper/llama1_13b/compress
python -m acip.entrypoints.acip_compress experiment=paper/llama2_7b/compress
python -m acip.entrypoints.acip_compress experiment=paper/llama2_13b/compress
python -m acip.entrypoints.acip_compress experiment=paper/llama31_8b/compress
python -m acip.entrypoints.acip_compress experiment=paper/mistral03_7b/compress
python -m acip.entrypoints.acip_compress experiment=paper/qwen25_3b/compress
python -m acip.entrypoints.acip_compress experiment=paper/qwen25_7b/compress
python -m acip.entrypoints.acip_compress experiment=paper/qwen25_14b/compress

# ablations
python -m acip.entrypoints.acip_compress --multirun experiment=paper/llama1_7b/ablation/compress_post_tune acip.post_tune_steps=null,100,500,1000
python -m acip.entrypoints.acip_compress experiment=paper/llama1_7b/ablation/compress_mask_only
python -m acip.entrypoints.acip_compress --multirun experiment=paper/llama1_7b/ablation/compress_sweep_stop acip.stop_ratio=0.2,0.3,0.4,0.6,0.8,0.95
python -m acip.entrypoints.acip_compress --multirun experiment=paper/llama1_7b/ablation/compress_sweep_stop_score acip.stop_score_map_at_ratio=null,0.6,0.8,0.95,0.99
python -m acip.entrypoints.acip_compress experiment=paper/llama1_7b/ablation/compress_sv_score_map
python -m acip.entrypoints.acip_compress experiment=paper/llama1_7b/ablation/compress_runtime
python -m acip.entrypoints.acip_compress experiment=paper/llama1_7b/ablation/compress_quantized
# requires ACIP model from experiment=paper/llama1_7b/ablation/compress_quantized
python -m acip.entrypoints.acip_compress experiment=paper/llama1_7b/ablation/compress_quantized_transfer
python -m acip.entrypoints.acip_compress experiment=paper/llama2_13b/ablation/compress_with_up_proj
python -m acip.entrypoints.acip_eval experiment=paper/llama1_7b/ablation/eval_pruning_reset_layers acip.pruning_config.reset_layers=true acip.pruning_config.reset_adapters=true
python -m acip.entrypoints.acip_eval experiment=paper/llama1_7b/ablation/eval_pruning_reset_layers acip.pruning_config.reset_layers=false acip.pruning_config.reset_adapters=false

# ----- acip_finetune (requires ACIP models from acip_compress) ------
python -m acip.entrypoints.acip_finetune --multirun experiment=paper/llama1_7b/finetune acip.prune_to_ratio=0.4,0.5,0.6,0.7,0.8,0.9,1.0
python -m acip.entrypoints.acip_finetune --multirun experiment=paper/llama1_13b/finetune acip.prune_to_ratio=0.4,0.5,0.6,0.7,0.8,0.9,1.0
python -m acip.entrypoints.acip_finetune --multirun experiment=paper/llama2_7b/finetune acip.prune_to_ratio=0.4,0.5,0.6,0.7,0.8,0.9,1.0
python -m acip.entrypoints.acip_finetune --multirun experiment=paper/llama2_13b/finetune acip.prune_to_ratio=0.4,0.5,0.6,0.7,0.8,0.9,1.0
python -m acip.entrypoints.acip_finetune --multirun experiment=paper/llama31_8b/finetune acip.prune_to_ratio=0.5,0.6,0.7,0.8,0.9,1.0
python -m acip.entrypoints.acip_finetune --multirun experiment=paper/mistral03_7b/finetune acip.prune_to_ratio=0.4,0.5,0.6,0.7,0.8,0.9,1.0
python -m acip.entrypoints.acip_finetune --multirun experiment=paper/qwen25_3b/finetune acip.prune_to_ratio=0.4,0.5,0.6,0.7,0.8,0.9,1.0
python -m acip.entrypoints.acip_finetune --multirun experiment=paper/qwen25_7b/finetune acip.prune_to_ratio=0.4,0.5,0.6,0.7,0.8,0.9,1.0
python -m acip.entrypoints.acip_finetune --multirun experiment=paper/qwen25_14b/finetune acip.prune_to_ratio=0.4,0.5,0.6,0.7,0.8,0.9,1.0

# ablations
python -m acip.entrypoints.acip_finetune --multirun experiment=paper/llama1_7b/ablation/finetune_quantized acip.prune_to_ratio=0.4,0.5,0.6,0.7,0.8,0.9,1.0
python -m acip.entrypoints.acip_finetune --multirun experiment=paper/llama1_7b/ablation/finetune_sv_score_map acip.prune_to_ratio=0.4,0.5,0.6,0.7,0.8,0.9,1.0
