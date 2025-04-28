#!/bin/zsh

python -m acip.entrypoints.acip_push_to_hub model/base@model=llama1_7b entrypoints/hub@acip=llama1_7b
python -m acip.entrypoints.acip_push_to_hub model/base@model=llama1_13b entrypoints/hub@acip=llama1_13b
python -m acip.entrypoints.acip_push_to_hub model/base@model=llama2_7b entrypoints/hub@acip=llama2_7b
python -m acip.entrypoints.acip_push_to_hub model/base@model=llama2_13b entrypoints/hub@acip=llama2_13b
python -m acip.entrypoints.acip_push_to_hub model/base@model=llama31_8b entrypoints/hub@acip=llama31_8b
python -m acip.entrypoints.acip_push_to_hub model/base@model=mistral03_7b entrypoints/hub@acip=mistral03_7b
python -m acip.entrypoints.acip_push_to_hub model/base@model=qwen25_3b entrypoints/hub@acip=qwen25_3b
python -m acip.entrypoints.acip_push_to_hub model/base@model=qwen25_7b entrypoints/hub@acip=qwen25_7b
python -m acip.entrypoints.acip_push_to_hub model/base@model=qwen25_14b entrypoints/hub@acip=qwen25_14b
