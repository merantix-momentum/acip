---
license: {{LICENSE}}
datasets:
- allenai/c4
language:
- en
metrics:
- perplexity
- accuracy
base_model:
- {{BASE_MODEL}}
pipeline_tag: text-generation
library_name: transformers
---
<div align="center">
  <img width="30%" alt="logo" src="https://imgur.com/A0MCHPq.png">
</div>

<div align="center">
    <a href="https://github.com/merantix-momentum/acip"><img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white.svg" alt="github" style="display: inline-block; vertical-align: middle;"></a>
    <a href="https://arxiv.org/abs/2502.01717"><img src="https://img.shields.io/badge/arXiv-2502.01717-b31b1b.svg" alt="arxiv" style="display: inline-block; vertical-align: middle;"></a>
    <a href="https://acip.merantix-momentum.cloud"><img alt="website" src="https://img.shields.io/website/https/acip.merantix-momentum.cloud.svg?down_color=red&down_message=offline&up_message=online" style="display: inline-block; vertical-align: middle;"></a>
    <a href="LICENSE"><img alt="license" src="https://img.shields.io/badge/license-Apache%202.0-blue" style="display: inline-block; vertical-align: middle;"></a>
</div>

<h2 align="center">
    <p> [
        <a href="https://github.com/merantix-momentum/acip">🤖 GitHub</a> |
        <a href="https://arxiv.org/abs/2502.01717">📄 Paper</a> |
        <a href="https://acip.merantix-momentum.cloud/">🌐 Website</a>
        ]
    </p>
</h2>

<h1 align="center">
    <p>ACIP applied to {{BASE_MODEL}}</p>
</h1>

This model repository is part of the ACIP Project and provides a compressible version of [`{{BASE_MODEL}}`](https://huggingface.co/{{BASE_MODEL}}). For more details, please visit our [code repo](https://github.com/merantix-momentum/acip).

# Quick Start

Just load the ACIP model via `from_pretrained`:
```python
from transformers import AutoModel

model = AutoModel.from_pretrained("{{REPO_ID}}", trust_remote_code=True)
```
This will download and create a fully parameterized ACIP model that can be pruned to any compression ratio you wish.
For example,
```python
model.prune_model_by_score(compression_ratio=0.4)
```
will prune `model` to 40% if its original size measured in number of parameters, i.e., 60% compression rate.
A unique feature of ACIP is that this operation is revertible in the sense that you can rerun `model.prune_model_by_score` as often as you like to evaluate your model at different sizes. Finally, you can "commit" to a certain ratio and run
```python
model.compress()
```
which will discard all pruned mask values of compressible linear layers. 
Now the model is actually compressed and you should observe a significant decrease of memory usage (this step is not revertible without reloading the ACIP model).
If you like, you can also run
```python
model.quantize()
```
to save even more memory (we have only tested 4bit quantization with `bitsandbytes`, but you could also customize this).

**🚀 That's it! You can now use your compressed model for inference or fine-tuning as any other Causal Language Model from 🤗 transformers.**

**Note**: The parameter `compression_ratio` ranges from 1.0 to 0.0, indicating the model size after compression. For example, 0.4 means that the model has only 40% of the original number of parameters and 1.0 means no compression at all.

# Dependencies

To run an ACIP model from our hub, you only need minimal dependencies, namely `torch`, `transformers`, `peft`, and optionally, `bitsandbytes` in case you want to quantize your model.
See [requirements.txt](requirements.txt) for pip-installable dependencies with exact version pins (newer version should work as well).

# License

{{LICENSE_TEXT}}

# Citation

When using or referring to this model, please cite our [paper](https://arxiv.org/abs/2502.01717):
```bibtex
@article{mxm2025acip,
  title={Choose Your Model Size: Any Compression by a Single Gradient Descent}, 
  author={M. Genzel, P. Putzky, P. Zhao, S. Schulze, M. Mollenhauer, R. Seidel, S. Dietzel, T. Wollmann},
  year={2025},
  journal={Preprint arXiv:2502.01717}
}
```



