from collections import defaultdict
from typing import Any

import torch

from acip.core.parametrized_model import ParametrizedModel
from acip.core.projected_layer import mask_sparsity
from acip.eval.evaluator import ModelEvaluator
from acip.eval.utils import eval_mode


class SizeModelEvaluator(ModelEvaluator):
    """
    Evaluates the size of a `ParametrizedModel` in its compressed and original form.
    The returned dictionary has the following keys:
     - size/num_params: Number of parameters of the original model, where only the number of parameters in
       parametrized modules are counted
     - size/num_params_full: Number of parameters of the original model
     - size/num_params_compressed: Number of parameters of the compressed model, where only the number of parameters in
       parametrized modules are counted
     - size/num_params_compressed_full: Number of parameters of the compressed model
     - size/compression_ratio: Compression ratio of the compressed model, where only the number of parameters in
       parametrized modules are counted
     - size/compression_ratio_full: Compression ratio of the compressed model
     - size/parametrized_modules/<keys>: Optional keys to get information about individual parametrized modules,
       see `get_parametrized_modules_info` for details
    """

    def __init__(self, eval_parametrized_modules: bool = False):
        """
        Args:
            eval_parametrized_modules: If True, the size of the individual parametrized modules is analyzed as well.
                Beware that these are many, which can slow down the evaluation and logging process.
        """
        self.eval_parametrized_modules = eval_parametrized_modules

    def __call__(
        self,
        model: ParametrizedModel,
        prefix: str | None = "size",
        **kwargs: Any,
    ) -> dict[str, Any]:
        prefix = prefix + "/" if prefix is not None else ""
        results = {}
        with torch.no_grad(), eval_mode(model):
            results[prefix + "num_params"] = model.get_num_params()
            results[prefix + "num_params_full"] = model.get_num_params(full=True)
            results[prefix + "num_params_compressed"] = model.get_num_params(compressed=True)
            results[prefix + "num_params_compressed_full"] = model.get_num_params(compressed=True, full=True)
            results[prefix + "compression_ratio"] = model.get_compression_ratio()
            results[prefix + "compression_ratio_full"] = model.get_compression_ratio(full=True)

            if self.eval_parametrized_modules:
                parametrized_modules_info = get_parametrized_modules_info(model=model)
                for m_name, info in parametrized_modules_info.items():
                    for k, v in info.items():
                        results[prefix + f"parametrized_modules/{m_name}_{k}"] = v
        return results


def get_parametrized_modules_info(model: ParametrizedModel, lp_norm: float = 1.0) -> dict[str, dict[str, Any]]:
    """
    Helper function to get size info about the individual parametrized modules of a `ParametrizedModel`.
    The returned dictionary is nested and has the format [<module name>][<param name>.<info>],
    where <info> is one of the following:
     - "sparsity" (of the parameter)
     - "numel" (of the parameter)
     - "compression_ratio" (of the parameter)
     - "lp_norm" (of the parameter, with the given `lp_norm` >= 1.0)
    Moreover, the returned dictionary also collects global information about the parametrized modules:
     - [<module_name>]["num_params"] (number of parameters of the unparametrized module)
     - [<module_name>]["num_params_compressed"] (number of parameters of the compressed module)
     - [<module_name>]["compression_ratio"] (achievable compression ratio of the module)
    """
    info = defaultdict(dict)
    with torch.no_grad():
        for m_name, module in model.parametrized_modules.items():
            for p_name, param in module.parametrization.get_target_params().items():
                info[m_name][f"{p_name}.sparsity"] = mask_sparsity(param)
                info[m_name][f"{p_name}.numel"] = param.numel()
                info[m_name][f"{p_name}.compression_ratio"] = (
                    info[m_name][f"{p_name}.sparsity"] / info[m_name][f"{p_name}.numel"]
                )
                info[m_name][f"{p_name}.lp_norm"] = torch.norm(param.float(), p=lp_norm).item()
            info[m_name]["num_params"] = module.parametrization.get_num_params()
            info[m_name]["num_params_compressed"] = module.parametrization.get_num_params(compressed=True)
            info[m_name]["compression_ratio"] = info[m_name]["num_params_compressed"] / info[m_name]["num_params"]
    return info
