import pytest
import rootutils

rootutils.setup_root(
    __file__,
    indicator=".project-root",
    pythonpath=True,
    dotenv=True,
    cwd=True,
    project_root_env_var=True,
)


import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM

from acip.core.acip_model import ACIPModelConfig
from acip.core.parametrized_model import AdapterConfig, BaseModelConfig, ParametrizationConfig, WeightQuantizationConfig


@pytest.fixture(scope="module")
def temp_dir(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("test_models")
    yield temp_dir
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


@pytest.fixture
def acip_model_config():
    """Config of a small ACIPModel suited for local testing."""
    base_model_config = BaseModelConfig(
        pretrained_model_cls=AutoModelForCausalLM,
        pretrained_model_kwargs={
            "pretrained_model_name_or_path": "facebook/opt-125m",
            "torch_dtype": "float32",
        },
    )
    parametrization_config = ParametrizationConfig(
        module_factory_cls="svd",
        module_factory_kwargs=dict(mask_scaling_factor=0.02),
        target_modules="all-linear",
    )
    adapter_config = AdapterConfig(
        peft_config=LoraConfig(
            r=32,
            lora_alpha=16,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules="all-linear",
            exclude_modules=["ortho", "parametrization", "base"],
        )
    )
    quantization_config = WeightQuantizationConfig(
        module_factory_cls="bnb4bit",
        module_factory_kwargs=dict(compute_dtype=torch.bfloat16, quant_type="fp4"),
        target_modules=["ortho", "base", "base_layer"],
    )

    return ACIPModelConfig(
        base_model_config=base_model_config,
        parametrization_config=parametrization_config,
        adapter_config=adapter_config,
        weight_quantization_config=quantization_config,
        model_mode="eval",
    )
