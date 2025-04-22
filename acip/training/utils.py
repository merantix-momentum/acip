from logging import getLogger
from typing import Any

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from plotly import graph_objects as go

logger = getLogger(__name__)


def stop_train(trainer: pl.Trainer) -> None:
    """Helper function to stop training of a Lightning `Trainer`."""
    trainer.should_stop = True
    logger.info(f"Training stop triggered at step {trainer.global_step}.")


def generate_params_plot(
    params: dict[str, torch.Tensor], zmin: float | None = 0.0, zmax: float | None = None
) -> go.Figure:
    """
    Generate a plotly heatmap figure from a dictionary of parameters.
    To this end, all parameters are vectorized, padded to the same length, and concatenated along the "dictionary axis".

    Args:
        params: Parameters to be visualized.
        zmin: If a float, truncate heatmap values below this value.
        zmax: If a float, truncates heatmap values above this value.

    Returns: Heatmap figure.
    """

    if len(params) == 0:
        return go.Figure()
    # Flatten
    params = {k: v.detach().flatten().float().cpu().numpy() for k, v in params.items()}
    # Pad
    max_len = max(len(param) for param in params.values())
    padded_params = [np.pad(param, (0, max_len - len(param)), constant_values=np.nan) for param in params.values()]
    # Concat
    heatmap_data = np.array(padded_params).T
    layer_names = list(params.keys())

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data,
            x=layer_names,
            y=list(range(max_len)),
            zmin=zmin if zmin is not None else np.nanmin(heatmap_data),
            zmax=zmax if zmax is not None else np.nanmax(heatmap_data),
        )
    )

    fig.update_layout(
        title=None,
        xaxis_title=None,
        xaxis=dict(showticklabels=False),  # hide x-axis tick labels due to space constraints
        yaxis_title="Index",
        yaxis=dict(autorange="reversed"),
    )

    return fig


def create_eval_dataframe(results_eval: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """
    Convert (evaluation) result dictionary into a dataframe.
    Here, the keys of the dictionary form the rows and the values are again dictionary-valued, representing
    the columns and their values for each row.
    The dataframe's columns are the union of all individual columns, where non-existent values are filled with NaN.

    This helper function is useful to create a wandb table.
    """
    # Collect all columns (metrics) of the dataframe
    all_metric_names = set()
    for metrics in results_eval.values():
        all_metric_names.update(metrics.keys())

    data = []
    for state, metrics in results_eval.items():
        # The keys of results_eval are stored in an extra column (serving as index)
        row = {"state": state}
        # Fill row with all metrics from results_eval[state]
        for metric_name in all_metric_names:
            row[metric_name] = metrics.get(metric_name, np.nan)
        data.append(row)

    return pd.DataFrame(data)
