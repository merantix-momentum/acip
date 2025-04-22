#!/bin/zsh

python -m acip.entrypoints.acip_push_to_hub model/base@model=llama1_7b acip.hub.license=other acip.hub.license_text="The license is inherited from the base model jeffwan/llama-7b-hf."
python -m acip.entrypoints.acip_push_to_hub model/base@model=llama1_13b acip.hub.license=other acip.hub.license_text="The license is inherited from the base model jeffwan/llama-13b-hf."
python -m acip.entrypoints.acip_push_to_hub model/base@model=llama2_7b acip.hub.license=llama2
python -m acip.entrypoints.acip_push_to_hub model/base@model=llama2_13b acip.hub.license=llama2
python -m acip.entrypoints.acip_push_to_hub model/base@model=llama31_8b acip.hub.license=llama3.1
python -m acip.entrypoints.acip_push_to_hub model/base@model=mistral03_7b acip.hub.license=apache-2.0
python -m acip.entrypoints.acip_push_to_hub model/base@model=qwen25_3b acip.hub.license=apache-2.0
python -m acip.entrypoints.acip_push_to_hub model/base@model=qwen25_7b acip.hub.license=apache-2.0
python -m acip.entrypoints.acip_push_to_hub model/base@model=qwen25_14b acip.hub.license=apache-2.0
