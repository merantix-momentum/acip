import logging
from typing import Any

import torch
from transformers import PreTrainedModel

from .parametrized_model import ParametrizedModel, ParametrizedModelConfig

logger = logging.getLogger(__name__)


class ACIPModelConfig(ParametrizedModelConfig):
    """
    Configuration for `ACIPModel`. Same functionality as `ParametrizedModelConfig`.

    See Also:
        - `ParametrizedModelConfig`
        - `ACIPModel`
    """

    model_type = "acip_model"


class ACIPModel(ParametrizedModel):
    """
    This class extends `ParametrizedModel` by additional functionality required for ACIP.
    It manages a `score_map` that stores the scores of the parametrized modules' target parameters,
    which are updated during tuning by the ACIP method.
    Moreover, it provides `prune_model_by_score` that prunes the target parameters of the model according to
    their scores to achieve any given size ratio.

    Notes: The `score_map` is managed in float32 internally because a lower precision may lead to unexpected numerical
        inaccuracies in the resulting parameter ranking. Fortunately, the memory consumption is negligible compared to
        the model weights itself.

    See Also:
        - `ParametrizedModel`
        - `ACIPModelConfig`
    """

    config_class = ACIPModelConfig

    def __init__(self, config: ACIPModelConfig, base_model: PreTrainedModel | None = None, **_: Any):
        super().__init__(config, base_model)
        self.config = config  # redundant but enables type hinting for ACIPModelConfig

        self._score_map: dict[str, torch.Tensor] | None = None
        # Register and initialize score map buffers
        # Important: don't run _update_score_map here because load_state_dict might still override the buffers
        self._init_score_map_buffers()

    def _init_score_map_buffers(self):
        """
        Register and initialize score map buffers in parametrized modules (with random numbers).
        Each target parameter "p_name" is associated with a buffer "p_name_score" that stores its score vector.
        """
        for m_name, module in self.parametrized_modules.items():
            for p_name, param in module.parametrization.get_target_params().items():
                module.parametrization.register_buffer(p_name + "_score", torch.ones_like(param.data).float())

    def _update_score_map(self):
        """Render `score_map` from the parametrized modules' score buffers."""
        self._score_map = {}
        for m_name, module in self.parametrized_modules.items():
            for p_name in module.parametrization.get_target_params().keys():
                self._score_map[f"{m_name}.parametrization.{p_name}"] = module.parametrization.get_buffer(
                    p_name + "_score"
                )

    @property
    def score_map(self) -> dict[str, torch.Tensor]:
        """Returns the score map as Tensor dictionary whose keys match those of `self.get_target_params`."""
        if self._score_map is None:
            self._update_score_map()
        return self._score_map

    @score_map.setter
    def score_map(self, score_map: dict[str, torch.Tensor]) -> None:
        """
        Updates `score_map` and the corresponding parametrized modules' score buffers.

        Args:
            score_map: Dictionary whose keys should match (a subset of) `self.get_target_params`.
        """
        if self._score_map is None:
            self._update_score_map()
        # score_map.keys() can be a subset of self.get_target_params().keys()
        for p_name, score in score_map.items():
            buffer = self.model.get_buffer(p_name + "_score")
            if buffer.shape != score.shape:
                raise ValueError(
                    f"Score map for '{p_name}' has incorrect shape: expected {buffer.shape}, got {score.shape}"
                )
            # cast to float32 to avoid numerical instabilities
            buffer.copy_(score.detach().float())
            self._score_map[p_name] = buffer

    def _predict_size_ratio_by_score(self, k: int, full: bool = False) -> tuple[float, dict[str, torch.Tensor]]:
        """
        Helper function that checks what would happen if the k smallest target parameters are pruned
        according to the global score map ranking. It returns the resulting size ratio
        and the corresponding parameter masks.

        Args:
            k: Number of target parameters to prune.
            full: Whether to count the number of parameters of the entire model or only the parametrized modules.
                See also `ParametrizedModel.get_num_params`.

        Returns: Tuple of size ratio and parameter masks. The masks indicate which parameters to keep.
        """
        # Find the threshold value for the k smallest entries according to the global score map ranking.
        score_map_cat = torch.cat([param.flatten() for param in self.score_map.values()])
        threshold = torch.kthvalue(score_map_cat, k).values.item()

        # Create a set of parameter masks marking which values to keep.
        param_masks = {}
        for p_name, score in self.score_map.items():
            param_masks[p_name] = (score > threshold).to(dtype=score.dtype)

        # Compute hypothetical size ratio if param_masks would be used as masks for the target parameters.
        size_ratio = self.get_size_ratio(full=full, target_params=param_masks)
        return size_ratio, param_masks

    def _get_param_masks(self, size_ratio: float, full: bool = False) -> dict[str, torch.Tensor]:
        """
        Helper function that determines which parameters to keep to reach a target size ratio.
        Instead of looping over `k -> _predict_size_ratio_by_score(k)`, a binary search can be used because
        the size ratio is monotonically increasing in k.

        Args:
            size_ratio: Target size ratio.
            full: Whether to count the number of parameters of the entire model or only the parametrized modules.
                See also `ParametrizedModel.get_num_params`.

        Returns: Parameter masks indicating which parameters to keep to reach the target size ratio.
        """
        if size_ratio == 1.0:
            return {p_name: torch.ones_like(score) for p_name, score in self.score_map.items()}

        # Perform a binary search to find the smallest k such that the size ratio is at least size_ratio.
        # Here, k_lo and k_hi are the lower and upper bound of the search interval.
        k_lo, k_hi = 1, sum(score.numel() for score in self.score_map.values())
        while k_lo < k_hi:
            k_mid = (k_lo + k_hi + 1) // 2  # round up to ensure low <= mid
            ratio, _ = self._predict_size_ratio_by_score(k=k_mid, full=full)
            if ratio > size_ratio:
                k_lo = k_mid
            else:
                k_hi = k_mid - 1
        k = k_lo
        # TODO: handle tie-breaks
        return self._predict_size_ratio_by_score(k=k, full=full)[1]

    def prune_model_by_score(
        self,
        size_ratio: float | None = None,
        compression_rate: float | None = None,
        full: bool = False,
    ) -> None:
        """
        This method prunes the target parameters of the model according to their scores to achieve
        a given size ratio.

        This can be efficiently implemented by a simple binary search strategy:
        We find the smallest number of parameters to be pruned according to the score map ranking
        such that the resulting size ratio is at least the target `size_ratio`.

        Args:
            size_ratio: The target size ratio, which is the ratio between the size of the compressed model and
                the original model (where size is measured in number of parameters).
                If not provided, `compression_rate` must be provided.
            compression_rate: This is a convenience parameter that allows you to set the target compression rate
                instead of `size_ratio`. It is equivalent to `size_ratio = 1.0 - compression_rate`.
                If both `size_ratio` and `compression_rate` are provided, `size_ratio` is used.
            full: Whether to count the number of parameters of the entire model or only the parametrized modules.
                See also `ParametrizedModel.get_num_params`.
        """
        if size_ratio is None and compression_rate is None:
            raise ValueError("Either `size_ratio` or `compression_rate` must be provided.")
        elif size_ratio is None and compression_rate is not None:
            size_ratio = 1.0 - compression_rate
        else:
            logger.warning("Both `size_ratio` and `compression_rate` are provided. Using `size_ratio`.")

        param_masks = self._get_param_masks(size_ratio=size_ratio, full=full)

        # Reset the target parameters according to the parameter masks
        for p_name, param in self.get_target_params().items():
            param.data[param_masks[p_name] > 0.0] = 1.0  # dummy value, will be rescaled by reset_target_params
            param.data[param_masks[p_name] == 0.0] = 0.0
        for m_name, module in self.parametrized_modules.items():
            if any(p_name.startswith(m_name) for p_name in param_masks.keys()):
                module.parametrization.reset_target_params(mode="nonzero")


# Register ACIPModelConfig and ACIPModel for AutoModel
# Required to push custom model to Huggingface Hub (see https://huggingface.co/docs/transformers/en/custom_models)
ACIPModelConfig.register_for_auto_class()
ACIPModel.register_for_auto_class("AutoModel")
