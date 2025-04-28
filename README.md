<h3 align="center">
    <img alt="logo" src="https://imgur.com/A0MCHPq.png">
</h3>

<p align="center">
    <a href="https://arxiv.org/abs/2502.01717"><img src="https://img.shields.io/badge/arXiv-2502.01717-b31b1b.svg" alt="arxiv"></a>
    <a href="https://acip.merantix-momentum.com/"><img alt="website" src="https://img.shields.io/website/https/acip.merantix-momentum.com.svg?down_color=red&down_message=offline&up_message=online"></a>
    <a href="https://huggingface.co/collections/MerantixMomentum/acip-67fe8f7b9f3132468a117ea6"><img alt="models" src="https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000.svg"></a>
    <a href="https://huggingface.co/meta-llama"><img alt="llama" src="https://img.shields.io/badge/LLaMA 1,2,3-pink"></a>
    <a href="https://huggingface.co/mistralai/Mistral-7B-v0.3"><img alt="llama" src="https://img.shields.io/badge/Mistral v0.3-pink"></a>
    <a href="https://huggingface.co/Qwen"><img alt="llama" src="https://img.shields.io/badge/Qwen 2.5-pink"></a>
    <a href="LICENSE"><img alt="license" src="https://img.shields.io/badge/license-Apache%202.0-blue"></a>
</p>

<h4 align="center">
    <p> [
        <a href="https://arxiv.org/abs/2502.01717">📄 Paper</a> |
        <a href="https://acip.merantix-momentum.com/">🌐 Website</a> |
        <a href="https://huggingface.co/collections/MerantixMomentum/acip-67fe8f7b9f3132468a117ea6">🤗 Models</a>
        ]
    </p>
</h4>

<h3 align="center">
    <p>Compressing Large Language Models as Intuitively as Images</p>
</h3>

Official implementation of ACIP (Adaptive Compression by Iterative Pruning).
Just give it a try with only 3 lines of code:
```python
from transformers import AutoModel

model = AutoModel.from_pretrained("MerantixMomentum/ACIP-llama2-7b", trust_remote_code=True)
model.prune_model_by_score(size_ratio=0.5).compress()
```
See our [project website](https://acip.merantix-momentum.com/) for a quick overview of the ACIP algorithm or dive into the full details with our paper

> [**Choose Your Model Size: Any Compression by a Single Gradient Descent**](https://arxiv.org/abs/2502.01717) </br>
*Martin Genzel\*, Patrick Putzky\*, Pengfei Zhao\*, Sebastian Schulze, Mattes Mollenhauer, Robert Seidel, Stefan Dietzel, Thomas Wollmann* (* equal contribution) <br>

This work was developed at [Merantix Momentum](https://merantix-momentum.com). If you are using it, [please cite it](#citation).

# Getting Started

## Quick Start

The easiest way to get started with ACIP is to download a ready-for-use model from our [Merantix Momentum 🤗 Hub](https://huggingface.co/collections/MerantixMomentum/acip-67fe8f7b9f3132468a117ea6). 
For this, you don't have to clone this repo and only minimal dependencies are required (`torch`, `transformers`, `peft`, and optionally, `bitsandbytes` in case you want to quantize your model). 
See [acip/core/requirements.txt](acip/core/requirements.txt) for pip-installable dependencies.

Just select any ACIP model and load it via `from_pretrained` like this one:
```python
from transformers import AutoModel

model = AutoModel.from_pretrained("MerantixMomentum/ACIP-llama2-7b", trust_remote_code=True)
```
This will download and create a fully parameterized [ACIP model](acip/core/acip_model.py) that can be pruned to any compression rate you wish.
For example,
```python
model.prune_model_by_score(size_ratio=0.4)
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

> ℹ️ The parameter `size_ratio` ranges from 1.0 to 0.0, indicating the model size after compression. For example, 0.4 means that the model has only 40% of the original number of parameters and 1.0 means no compression at all. Alternatively, you can also set `compression_rate` in `prune_model_by_score`, which is equivalent to `size_ratio = 1.0 - compression_rate`.

## Installation

To run the ACIP code to compress or fine-tune your own model, please clone this repo:
```bash
git clone https://github.com/MerantixMomentum/acip.git
```
To install all dependencies, we recommend using [uv](https://docs.astral.sh/uv/) with Python 3.11 as base interpreter (Python 3.12 should work as well).
Once uv is set up, you can just run
```bash
uv sync
```
to install the requirements as well as the [acip](acip) package (see [pyproject.toml](pyproject.toml) for details).

If you want to use a different package manager like Conda, you can also simply install all pinned dependencies from the provided [requirements.txt](requirements.txt).

> ❗️ Custom environment variables are managed via dot-env. Before using the repo, please create a `.env` file from [`.env.example`](.env.example) and fill in the required values.

## Running ACIP

To try out ACIP on your own model, you can run the [acip_compress](acip/entrypoints/acip_compress.py) entrypoint with
```bash
python -m acip.entrypoints.acip_compress model.base_model_name_or_path=<HF Repo or Local Path> model.identifier=<model name>
```
Here, `base_model_name_or_path` is passed to [`PreTrainedModel.from_pretrained`](https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) to load the base model and `identifier` specifies the run id and output directory name. You may omit `identifier`, which will set it to `"default"`.

ACIP will now run for a while and produce a prunable version of your base model, which is finally saved to an output directory (by default this is `<Project Root>/artifacts/runs/default/compress_<model.identifier>/model`).

Next, you can revisit [Quick Start](#quick-start) and load your ACIP from disk via `from_pretrained` — just replace the 🤗 Repo name with the local model output directory.
That's it! You can now work with your ACIP model as with the ones from our Hub. 

There are of course many more options and tweaks we skipped here for simplicity.
Please find more details on available [ACIP Entrypoints](#acip-entrypoints) and the underlying [Code Design](#code-structure--design) below.

## Paper Experiments

To make the experiments from our [paper](https://arxiv.org/abs/2502.01717) as reproducible as possible, we have compiled all necessary Python run-commands in [scripts/experiments_paper.sh](scripts/experiments_paper.sh). 
The corresponding Hydra configs of our experiments can be found in [config/experiment/paper](config/experiment/paper).
Note that the finetuning runs and some ablations require a ready-to-use ACIP model as input. 
So you first need to perform the corresponding ACIP compression run (or load the model from our [🤗 Hub](https://huggingface.co/collections/MerantixMomentum/acip-67fe8f7b9f3132468a117ea6)).

# Advanced Usage

## ACIP Entrypoints

All entrypoints of the ACIP project are based on [Hydra config management](https://hydra.cc/docs/intro/).
We currently provide the following entrypoints:
 - [**acip_compress**](#acip_compress): Runs the ACIP algorithm on a given base model to produce a prunable [ACIP model](acip/core/acip_model.py).
 - [**acip_finetune**](#acip_finetune): Fine-tunes a given ACIP model with [LoRA](https://arxiv.org/abs/2106.09685).
 - [**acip_eval**](#acip_eval): Loads and evaluates a given ACIP model.

The basic CLI syntax to run these entrypoints is as follows:
```bash
python -m acip.entrypoints.<Entrypoint Name> <Hydra Config Args>
```
You have already seen a typical example [above](#running-acip).
Below, we outline what options you have for the Hydra Config Args in [general](#general-tweaks) and for each of the above entrypoints. 
For a detailed discussion of Hydra's basic override syntax, please see their [docs](https://hydra.cc/docs/advanced/override_grammar/basic/).

### General Config & Tweaks

The above-mentioned entrypoints all share the same base class, [`ACIPEntrypoint`](acip/entrypoints/acip_entrypoint.py), which is based on our [MxM Scaffold package](https://github.com/merantix-momentum/scaffold-core).
So all entrypoints basically run the same code but with different configurations, which are determined by the accompanying (structured) config class [`ACIPEntrypointConf`](acip/entrypoints/acip_entrypoint.py).
Technically, `ACIPEntrypointConf` is just a dataclass-like container that aggregates all sub-configs required for the run.
Please see below for more details on the individual sub-configs and global overrides, which can be tweaked via the `<Hydra Config Args>`.

> ℹ️️ All config arguments described below have sensible defaults, so that all overrides are fully optional.
> Moreover, we only focus on the most relevant arguments in this documentation. For even more information and docs, please use the links to navigate to the actual (sub-)config files.

> ️️ℹ️ To explore and debug your entrypoint config, use `run.dry_run=true`, which will compile and print the full config of your experiment without running it.

#### Entrypoint Sub-Configs

<details id="run-config">
  <summary>🛠️ <a href="config/entrypoints/base.yaml"><code><b>run</b></code></a></summary>

Basic information & config of the run. Important options are:
- `run.id`: Descriptive identifier for the run. Also determines the name of the output directory.
- `run.group`: Group identifier for the run. By default, runs are grouped by their `model.identifier`, `data.identifier`, and `run.series`.
- `run.series`: Series identifier for the run, typically the name of an entire experiment series.
- `run.path`: The output directory for the run. Defaults to `<paths.run_dir>/<run.id>`.
- `run.tags_custom`: List of additional tags for the run, which will be also used as W&B tags if applicable.
- `run.dry_run`: If `true`, the entrypoint will not run the actual experiment but instead print the full config.
- `run.save`: List of artifact types to save. Available options: `config`, `results`, `models`.
</details>

<details id="data-config">
  <summary>🛠️ <a href="config/data/base.yaml"><code><b>data</b></code></a></summary>

Configures the [dataset (factories)](acip/data/dataset.py) for the entrypoint. Important options are:
- Currently available datasets: [`data=c4`](config/data/c4.yaml) (default) and [`data=wikitext2`](config/data/wikitext2.yaml).
- `data.identifier`: Descriptive identifier for the dataset.
- `data.train_dataset_factory.shuffle`: Whether to shuffle the train dataset or not. Similar options exist for `val_dataset_factory` and `test_dataset_factory`.
- `data.train_dataset_factory.seed`: Shuffle seed for the train dataset. By default it is set to `training.seed`.
</details>

<details id="model-config">
  <summary>🛠️ <a href="config/model"><code><b>model</b></code></a></summary>

Configures the [model factory](acip/model/model_factory.py) and [tokenizer factory](acip/model/tokenizer_factory.py) for the entrypoint. The resulting [`ACIPModelFactory`](acip/model/model_factory.py) is used to instantiate or load an [ACIP model](acip/core/acip_model.py).
Important options are:
- `model.identifier`: Descriptive identifier for the base model.
- `model.base_model_name_or_path`: Huggingface repo or local path pointing to the base model to be loaded and compressed by ACIP.
- `model.ctx_length`: Context length to use for perplexity evaluation (see [here](config/eval/evaluator/ppl.yaml)).

> ❗ `model.base_model_name_or_path` is a **required** parameter to specify a base model. Instead of setting it manually, you can define or choose a base model config [here](config/model/base) and inject it by an override, e.g., `model/base@model=llama1_7b`.

Details on sub-configs:
- [`model.model_factory`](config/model/model_factory): Configures an [`ACIPModelFactory`](acip/model/model_factory.py). Its key parameters are managed by the [acip sub-config](#acip-config) of each entrypoint.
- The precise config of the [ACIP model](acip/core/acip_model.py) is defined in [config/model/model_factory/acip_config](config/model/model_factory/acip_config), aggregating sub-configs for the base model, parametrization, adapters, and optional quantization (see also [`ParametrizedModelConfig`](acip/core/parametrized_model.py)). You can expand and modify them to your needs, but as for `model_factory`, the most important parameters are managed by the [acip sub-config](#acip-config) of each entrypoint. Also note that you can ignore these sub-configs if you load an ACIP model from disk or repo.
- [`model.tokenizer_factory`](config/model/tokenizer_factory): Configures a [`TokenizerFactory`](acip/model/tokenizer_factory.py). By [default](config/model/tokenizer_factory/default.yaml), we use the pre-trained tokenizer associated with the base model, but you can also use a custom one like [llama.yaml](config/model/tokenizer_factory/llama.yaml) and inject it by an override `model/tokenizer_factory@model=llama`.
</details>

<details id="training-config">
  <summary>🛠️ <a href="config/training/base.yaml"><code><b>training</b></code></a></summary>

Configures the training-related parts of the ACIP algorithm and optional fine-tuning, based on PyTorch Lightning. Important options are:
- `training.seed`: Global training seed used for PL's [seed_everything](https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/fabric/utilities/seed.py) and [dataset factories](#data-config).
- `training.batch_size`: Batch size for train, val, and test dataloaders.
- `training.log_every_n_train_steps`: Logging frequency of model monitoring while training. Set to `null` to disable.
- `training.data_module`: Keyword arguments for the [BaseDataModule](acip/training/pl_module.py).
- `training.trainer`: Keyword arguments for the [PL Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html). Here, you can specify important training parameters (devices, train steps, precision, etc.).

Details on sub-configs:
- [`training.objective`](config/training/objective): Configures the [`Objective`](acip/training/objective.py) to be optimized by [`BaseLitModule`](acip/training/pl_module.py). This sub-config is highly entrypoint-specific and selected by the (top-level) [configs](config/entrypoints).
- [`training.optimizer_factory`](config/training/optimizer): Configures the [optimizer (factory)](acip/training/optimizer.py) used by [`BaseLitModule`](acip/training/pl_module.py). As for `training.objective`, this sub-config is highly entrypoint-specific and selected by the (top-level) [configs](config/entrypoints).
- `training.callbacks`: Following PL's best practices, we make use of several callbacks to flexibly extent the training process by additional functionality. `training.callbacks` compiles a dictionary of all callbacks that will be passed to the PL Trainer. The injection of callbacks is managed by the (top-level) [entrypoint configs](config/entrypoints) and is organized in three different sub-classes:
  - [`training.acip`](config/training/acip/default.yaml): Schedules the ACIP algorithm and score map updates, see also [here](acip/training/acip.py). **Note**: Only used by the [acip_compress entrypoint](#acip_compress) and all key parameters are conveniently managed by the [acip sub-config](#acip-config).
  - [`training.monitoring`](config/training/monitoring): Configures one or more [callbacks](acip/training/monitoring.py) that monitor the training process and important model characteristics (e.g., size ratio and gradient norms) with frequency `training.log_every_n_train_steps`.
  - [`training.benchmarking`](config/training/benchmarking): Configures one or more [callbacks](acip/training/benchmarking.py) that benchmark the (ACIP) model at the beginning and end of training. Conceptually, these callbacks are similar to `training.monitoring` but can involve a more extensive evaluation that is not practical during training.
</details>

<details id="eval-config">
  <summary>🛠️ <a href="config/eval/default.yaml"><code><b>eval</b></code></a></summary>

Helper sub-config that configures a dictionary collection of [`ModelEvaluator`](acip/eval/evaluator.py) instances that can be used to evaluate an (ACIP) model at any point in training. See [config/eval/evaluator](config/eval/evaluator) and the corresponding [classes](acip/eval) for details about the individual evaluators.

The configured evaluators are primarily used by the [monitoring](config/training/monitoring/model_monitor_compress.yaml) and [benchmarking](config/training/benchmarking/default.yaml) callbacks.
</details>

<details id="wandb-config">
  <summary>🛠️ <a href="config/entrypoints/base.yaml"><code><b>wandb</b></code></a></summary>

Specifies the config for W&B logging. Important options are:

- `wandb.name`: Display name the W&B run, which is also used to generate a unique, but human-readable W&B run id. Defaults to `<run.id>`.
- `wandb.dir`: Local output directory for W&B logs. Defaults to `/tmp/wandb`.
- By default, `wandb.base_url`, `wandb.entity`, and `wandb.project` are set by the environment variables `WANDB_BASE_URL`, `WANDB_ENTITY`, and `WANDB_PROJECT`, respectively (see [dot-env](.env.example)). 
- You can fully disable W&B logging by setting `wandb=null`.
</details>

<details id="paths-config">
  <summary>🛠️ <a href="config/entrypoints/paths/default.yaml"><code><b>paths</b></code></a></summary>

Specifies the project root path and where to store artifacts (run outputs, models, datasets, cache, etc.).
Important options are:
- `paths.artifact_dir`: Parent directory for all artifacts. Defaults to `<path.root_dir>/artifacts`.
- `paths.run_dir`: Parent directory for all run outputs. Defaults to `<paths.artifact_dir>/runs`.
- `paths.data_dir`: Parent directory for all local datasets. Defaults to `<paths.artifact_dir>/data`.
- `paths.cache_dir`: Parent directory for cache, in particular, HuggingFace (`HF_HOME` is set via [dot-env](.env.example)). Defaults to `<paths.artifact_dir>/cache`.
</details>

<details id="acip-config">
  <summary>🛠️ <code><b>acip</b></code></summary>

This sub-config configures all ACIP-related parameters of a run. It is highly entrypoint-specific and managed by the (top-level) [entrypoints configs](config/entrypoints). Please find more details on available tweaks and options of the individual entrypoints in the sections below.
</details>

#### Global Overrides

<details id="experiment-config">
  <summary>🛠️ <a href="config/experiment"><code><b>experiment</b></code></a></summary>

While the [ACIP entrypoint configs](config/entrypoints) set sensible defaults, they can be easily overwritten or modified by an [experiment config](config/experiment) to design a custom run. Each of these configs operates on the top-level (global) entrypoint config and can therefore override any parameter or sub-config.
Here is a typical example that runs an experiments from our paper:
```bash
python -m acip.entrypoints.acip_compress experiment=paper/llama1_7b/compress
```
Please see [config/experiment/paper](config/experiment/paper) many more examples of experiments.
You may also provide a list of experiment configs to apply multiple overrides.
</details>

<details id="options-config">
  <summary>🛠️ <a href="config/options"><code><b>options</b></code></a></summary>

Technically, this override follows the same syntax as [experiment](#experiment-config) and they can be combined with each other. The purpose of "options", however, is to tweak specific parts of the entrypoint config, especially for debugging purposes.
You can add you own config to [config/options](config/options) or choose one or more from the following list:
- `no_benchmarking`: Disable the [benchmarking](config/training/benchmarking) sub-config.
- `no_monitoring`: Disable the [monitoring](config/training/monitoring) sub-config.
- `no_output`: Disable any output to files, no saving, and no W&B logging (stdout/stderr is still printed).
- `verbose`: Set Hydra's job logging level to `DEBUG`.
</details>

### [acip_compress](config/entrypoints/acip_compress.yaml)

<details>

This is the central entrypoint to run the ACIP algorithm from our [paper](https://arxiv.org/abs/2502.01717), see there for more conceptual details.
For this particular entrypoint, the following options of the [acip sub-config](#acip-config) are important:
- `acip.stop_ratio`: Size ratio at which to stop ACIP.
- `acip.post_tune_steps`: How many steps to continue optimizing the adapters after ACIP stopped.
- `acip.lr`: Global learning rate for AdamW to tune the mask parameters and adapters.
- `acip.test_ratios`: Size ratios at which to [benchmark](config/training/benchmarking/default.yaml) the final ACIP model. These results will be also save to the output directory.
- `acip.quantize_weights`: If true, the U and V weights of the SVD parametrization will be quantized according to a [quantization config](config/model/model_factory/acip_config/weight_quantization/bnb_4bit_fp4.yaml).
- The [ACIP regularization parameter scheduler](acip/training/acip.py) can be controlled through `reg_scheduler_start_weight`, `reg_scheduler_update_every`, and `reg_scheduler_update_factor`.
- `acip.save.path`: Where to save the final ACIP model. Defaults to `<run.path>/model`.

Relevant sub-config overrides:
- (**Required**) [`model/base@model=...`](config/model/base): Specify a pre-configured base model to be compressed. Alternatively, you can override `model.base_model_name_or_path` like in our [introductory example](#running-acip).
- [`training/monitoring@training=...`](config/training/monitoring): Specify one or more monitoring callbacks.
- [`training/benchmarking@training=...`](config/training/benchmarking): Specify one or more benchmarking callbacks.
- [`storage@acip=...`](config/entrypoints/storage): If [`acip_compress_compact`](config/entrypoints/storage/acip_compress_compact.yaml), only the mask parameters and adapters will be saved. This will save a lot of disk space, but requires to fully parametrize the initial ACIP model again when loading it. In that case, you have to set `acip.load.init_empty_weights=false` in [acip_finetune](#acip_finetune) and [acip_eval](#acip_eval).
- [`data=...`](config/data): Specify a pre-configured dataset.

You can fine more examples of overrides in our [paper experiments](config/experiment/paper).
</details>

### [acip_finetune](config/entrypoints/acip_finetune.yaml)

<details>

This is a complementary entrypoint that allows you to fine-tune an ACIP model obtained from [acip_compress](#acip_compress). Note that fine-tuning only concerns the adapters (LoRA parameters), not the mask parameters, which remain frozen.
For this particular entrypoint, the following options of the [acip sub-config](#acip-config) are important:
- `acip.finetune_steps`: How many steps to fine-tune the adapters.
- `acip.prune_to_ratio`: Size ratio to which the loaded ACIP model is to be pruned (and compressed). This operation is not revertible and you will obtain a fine-tuned ACIP model at this particular size ratio.
- `acip.lr`: Global learning rate for AdamW to tune the adapters.
- `acip.quantize_weights`: If true, the U and V weights of the SVD parametrization will be quantized according to a [quantization config](config/model/model_factory/acip_config/weight_quantization/bnb_4bit_fp4.yaml).
- `acip.load.model_name_or_path`: Path to the ACIP model to be fine-tuned. Could also be an ACIP model from our [🤗 Hub](https://huggingface.co/collections/MerantixMomentum/acip-67fe8f7b9f3132468a117ea6).
- `acip.save.path`: Where to save the final ACIP model. Defaults to `<run.path>/model`. By default, only the mask parameters and adapters are stored to save disk space, see also storage override below. 

> ❗ When loading a fine-tuned ACIP model from disk, you need to set `acip.load.init_empty_weights=false`.

Relevant sub-config overrides:
- (**Required**) [`model/base@model=...`](config/model/base): Specify a pre-configured base model to be compressed. Note that the base model weights are implicitly loaded with the ACIP model, but this sub-config is still required to configure a suitable tokenizer (factory).
- [`training/objective@training=...`](config/training/objective): Specify a custom [`Objective`](acip/training/objective.py) implementation.
- [`training/optimizer@training=...`](config/training/optimizer): Specify a custom [`OptimizerFactory`](acip/training/optimizer.py) implementation.
- [`training/monitoring@training=...`](config/training/monitoring): Specify one or more monitoring callbacks.
- [`training/benchmarking@training=...`](config/training/benchmarking): Specify one or more benchmarking callbacks.
- [`storage@acip=...`](config/entrypoints/storage): If [`acip_compress_full`](config/entrypoints/storage/acip_finetune_full.yaml), the full fine-tuned ACIP is saved, which allows you to quickly load it from disk.
- [`data=...`](config/data): Specify a pre-configured dataset.
</details>

### [acip_eval](config/entrypoints/acip_eval.yaml)

<details>

This is entrypoint is similar to [acip_finetune](#acip_finetune), but only evaluates a ready-to-use ACIP model without any fine-tuning. The specific evaluation routine is controlled by the [`training/benchmarking`](config/training/benchmarking) sub-config, see below.
For this particular entrypoint, the following options of the [acip sub-config](#acip-config) are important:
- `acip.prune_to_ratio`: Size ratio to which the loaded ACIP model is to be pruned. If `null` (default), the model is loaded as is.
- `acip.test_ratios`: Size ratios at which to evaluate the ACIP model. These results will be also save to the output directory. If `null`, the model is only evaluated at `acip.prune_to_ratio`.
- `acip.compress_and_unparametrize`: Whether to actually compress the model (cannot be reverted). Activating this flag only makes sense if `acip.prune_to_ratio` is not `null` and `acip.test_ratios=null`.
- `acip.quantize_weights`: If true, the U and V weights of the SVD parametrization will be quantized according to a [quantization config](config/model/model_factory/acip_config/weight_quantization/bnb_4bit_fp4.yaml).
- `acip.load.model_name_or_path`: Path to the ACIP model to be evaluated. Could also be an ACIP model from our [🤗 Hub](https://huggingface.co/collections/MerantixMomentum/acip-67fe8f7b9f3132468a117ea6). 
 
> ❗ If the ACIP model was saved in compact format, i.e., only mask parameters and adapters were saved, you need to set `acip.load.init_empty_weights=false`.

Relevant sub-config overrides:
- (**Required**) [`model/base@model=...`](config/model/base): Specify a pre-configured base model to be compressed. Note that the base model weights are implicitly loaded with the ACIP model, but this sub-config is still required to configure a suitable tokenizer (factory) used for evaluation.
- [`training/benchmarking@training=...`](config/training/benchmarking): Specify one or more benchmarking callbacks that will evaluate the loaded ACIP model.
</details>

### Code Structure & Design

```
├── acip             # ACIP package source code
│   ├── core         # Self-containted core package defining ACIPModel and required components
│   ├── data         # Factories to create ready-to-use datasets
│   ├── entrypoints  # ACIP entrypoints
│   ├── eval         # Model evaluators for monitoring and benchmarking
│   ├── model        # Factories for ACIP models and corresponding tokenizers
│   ├── training     # Training-related code (Lightning modules, optimizers, callbacks, etc.)
│   └── utils        # Utility functions
├── artifacts        # Parent directory for all artifacts
│   ├── cache        # Cache directory, in particular for 🤗 artifacts
│   ├── data         # Local datasets
│   └── runs         # Parent directory for all run outputs
├── config           # Hydra configs (subdirectories mirror the "acip" package structure)
├── scripts          # Utility scripts
└── test             # Unit tests
```

#### Important Design Concepts

[`acip.core`](acip/core) is designed as a fully self-contained package:
- The central object of `acip.core` is the [`ACIPModel`](acip/core/acip_model.py), which implements the central pruning functionality via `ACIPModel.prune_model_by_score` and is the outcome of the ACIP algorithm.
- Its base class is [`ParametrizedModel`](acip/core/parametrized_model.py), which manages the underlying parametrization of an ACIP model. Moreover, it allows you to equip the model with 🤗-PEFT adapters and perform quantization if needed.
- The parametrization mechanism itself is based on [`Parametrization`](acip/core/parametrized_layer.py), which enabled in-place modification of existing (linear) models layers. [`SVDLinearParametrization`](acip/core/projected_layer.py) is a child class that implements the SVD decomposition used in ACIP.
- Both `ACIPModel` and `ParametrizedModel` are implemented as [`PreTrainedModel`](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel), wrapping the base model to be parametrized and compressed. Each class is accompanied by a custom [`PreTrainedConfig`](https://huggingface.co/docs/transformers/main/en/main_classes/configuration), which fully configures the model, see [`ParametrizedModelConfig`](acip/core/parametrized_model.py) and [acip_config](config/model/model_factory/acip_config) for more details. We followed [🤗's custom model guide](https://huggingface.co/docs/transformers/en/custom_models) to make our ACIP models fully compatible with their API (`from_pretrained`, `save_pretrained`, `push_to_hub`, etc.). In particular, an ACIP model should behave exactly as the underlying base model and inherit its I/O interface. The (parametrized) base model can be accessed via `ACIPModel.model`.

Our training logic is based on [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/). [`BaseLitModule`](acip/training/pl_module.py) implements the central `LightningModule`, which provides a general training loop for any `PreTrainedModel` (not just Causal Language Models):
  - `BaseLitModule` requires an [`Objective`](acip/training/objective.py), which is responsible for performing the model forward pass and loss computation.
  - `BaseLitModule` builds its model and optimizer from a [model factory](acip/model/model_factory.py) and [optimizer factory](acip/training/optimizer.py), respectively. This follows Lightning's conventions to configure these objects lazily via the `configure_model` and `configure_optimizers` hooks, making our code extendable to more advanced Parallel Strategies like FSDP.
  - Similarly, [`BaseDataModule`](acip/training/pl_module.py) builds its datsets and dataloaders from a [dataset factory](acip/data/dataset.py).
  - All custom and ACIP-specific functionality is implemented via Lightning Callbacks, which can be divided into three groups:
    1. [ACIP](acip/training/acip.py): Implements training-related parts of the ACIP algorithm, like score map updates, regularization parameter scheduling, and post-tuning.
    2. [Monitoring](acip/training/monitoring.py): Implements model monitoring during training, involving regular calls of different [Model Evaluators](acip/eval/evaluator.py).
    3. [Benchmarking](acip/training/benchmarking.py): Implements benchmarking of a model before and after training, involving (more expensive) calls of Model Evaluators.

> ℹ️ To dive deeper into the code, we recommend starting with [`acip_entrypoint`](acip/entrypoints/acip_entrypoint.py), as it instantiates and manages all high level objects of an ACIP run.

# Updates

- [x] [2025-04-22] Released ACIP paper code.
- [x] [2025-04-15] Shared all ACIP models on [🤗 Hub](https://huggingface.co/collections/MerantixMomentum/acip-67fe8f7b9f3132468a117ea6). 

# Contact

Feel free to reach out to us via GH issues or email! <br>
`martin.genzel at merantix-momentum dot com` <br>
`patrick.putzky at merantix-momentum dot com`

# License

This project is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.

# Citation

When using or referring to this project, please cite our [paper](https://arxiv.org/abs/2502.01717):
```bibtex
@article{mxm2025acip,
  title={Choose Your Model Size: Any Compression by a Single Gradient Descent}, 
  author={M. Genzel, P. Putzky, P. Zhao, S. Schulze, M. Mollenhauer, R. Seidel, S. Dietzel, T. Wollmann},
  year={2025},
  journal={Preprint arXiv:2502.01717}
}
```
