import math
from test.model.utils import deep_compare, forward_compare, model_compare

import torch
from accelerate import init_on_device
from transformers import AutoModel

from acip.core.acip_model import ACIPModel, ACIPModelConfig
from acip.core.parametrized_model import ParametrizedModel, ParametrizedModelConfig


def test_acip_model_config_save_and_load(acip_model_config, temp_dir):
    """Creates, saves, and loads an ACIP model config."""
    save_path = str(temp_dir / "test_config")
    config = acip_model_config
    config.save_pretrained(save_path)
    config_loaded = ACIPModelConfig.from_pretrained(save_path)
    deep_compare(config, config_loaded)


def test_parametrized_model_save_and_load(acip_model_config, temp_dir, test_device):
    """Tests reloading a `ParametrizedModel` from disk with and without the "with_init_empty_weights" flag."""
    save_path = str(temp_dir / "test_model")
    with init_on_device(test_device):
        config = ParametrizedModelConfig(
            base_model_config=acip_model_config.base_model_config,
            parametrization_config=acip_model_config.parametrization_config,
            adapter_config=acip_model_config.adapter_config,
            weight_quantization_config=acip_model_config.weight_quantization_config,
            model_mode=acip_model_config.model_mode,
        )
        parametrized_model = ParametrizedModel(config).to(test_device)

        parametrized_model.save_pretrained(save_path)
        parametrized_model_loaded = ParametrizedModel.from_pretrained(save_path, with_init_empty_weights=True).to(
            test_device
        )

        model_compare(parametrized_model, parametrized_model_loaded)
        forward_compare(parametrized_model, parametrized_model_loaded)

        parametrized_model.save_pretrained(
            save_path,
            include_filter=[
                "score",
                "lora_A.default.weight",
                "lora_B.default.weight",
                "lora_A.default.bias",
                "lora_b.default.bias",
            ],
        )
        parametrized_model_loaded = ParametrizedModel.from_pretrained(save_path, with_init_empty_weights=False).to(
            test_device
        )
        model_compare(parametrized_model, parametrized_model_loaded)
        forward_compare(parametrized_model, parametrized_model_loaded)


def test_acip_model_save_and_load(acip_model_config, temp_dir, test_device):
    """Tests reloading an `ACIPModel` from disk with and without the "with_init_empty_weights" flag."""
    save_path = str(temp_dir / "test_model")
    with init_on_device(test_device):
        config = acip_model_config
        acip_model = ACIPModel(config).to(test_device)

        acip_model.save_pretrained(save_path)
        acip_model_loaded = ACIPModel.from_pretrained(save_path, with_init_empty_weights=True).to(test_device)

        model_compare(acip_model, acip_model_loaded)
        forward_compare(acip_model, acip_model_loaded)

        acip_model.save_pretrained(
            save_path,
            include_filter=[
                "score",
                "lora_A.default.weight",
                "lora_B.default.weight",
                "lora_A.default.bias",
                "lora_b.default.bias",
            ],
        )
        acip_model_loaded = ACIPModel.from_pretrained(save_path, with_init_empty_weights=False).to(test_device)
        model_compare(acip_model, acip_model_loaded)
        forward_compare(acip_model, acip_model_loaded)


def test_acip_model_push_to_hub(acip_model_config, temp_dir, test_device):
    """Tests pushing an `ACIPModel` to the HuggingFace Hub and reloading it."""
    with init_on_device(test_device):
        config = acip_model_config
        acip_model = ACIPModel(config).to(test_device)
        acip_model.push_to_hub("acip_test")

        acip_model_loaded = AutoModel.from_pretrained(
            "martingenzel/acip_test", with_init_empty_weights=False, trust_remote_code=True
        ).to(test_device)
        model_compare(acip_model, acip_model_loaded)
        forward_compare(acip_model, acip_model_loaded)


def test_acip_model_quantization(acip_model_config, test_device):
    """Tests quantization of an `ACIPModel`."""
    with init_on_device(test_device):
        config = acip_model_config
        acip_model = ACIPModel(config).to(test_device)

    num_params = acip_model.get_num_params(full=True)
    acip_model.quantize()
    assert acip_model.get_num_params(full=True) == num_params, "Number of parameters do not match"


def test_base_vs_acip_model(acip_model_config, test_device):
    """Tests if an unparametrized model and a parametrized model have the same number of parameters and forward."""
    with init_on_device(test_device):
        config = acip_model_config
        acip_model = ACIPModel(config).to(test_device)

        config.parametrization_config = None
        config.weight_quantization_config = None
        base_model = ACIPModel(config).to(test_device)

        num_params = 0
        for param in base_model.parameters():
            num_params += param.numel()

        print(f"Number of parameters: {num_params}")
        assert num_params == acip_model.get_num_params(full=True), "Number of parameters do not match"

        forward_compare(acip_model, base_model, atol=1e-1)


def test_acip_model_compression(acip_model_config, temp_dir, test_device):
    """Tests pruning and compression of an `ACIPModel`."""
    save_path = str(temp_dir / "test_model")
    with init_on_device(test_device):
        config = acip_model_config
        acip_model = ACIPModel(config).to(test_device)

        # Random score map to achieve a unique parameter ranking
        score_map = acip_model.score_map
        for k, v in score_map.items():
            score_map[k] = torch.randn_like(v)
        acip_model.score_map = score_map

        acip_model.save_pretrained(save_path)
        acip_model_copy = ACIPModel.from_pretrained(save_path, with_init_empty_weights=True).to(test_device)
        model_compare(acip_model, acip_model_copy)

        compression_ratio = 0.75
        acip_model.prune_model_by_score(compression_ratio=compression_ratio)
        acip_model_copy.prune_model_by_score(compression_ratio=compression_ratio)
        model_compare(acip_model, acip_model_copy)

        print(f"Actual compression ratio: {acip_model.get_compression_ratio()}")
        assert (
            math.fabs(acip_model.get_compression_ratio() - compression_ratio) < 0.01
        ), f"Compression ratio should be close to {compression_ratio}"

        acip_model_copy.compress()
        forward_compare(acip_model, acip_model_copy, atol=1e-3)
