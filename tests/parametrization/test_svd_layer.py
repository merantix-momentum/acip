import copy

import torch
from torch import allclose, randn

from acip.core.parametrized_layer import parametrize_module, unparametrize_module
from acip.core.projected_layer import SVDLinearParametrization, ThresholdCriterion


def test_svdlinear_parametrization():
    from torch.nn import Linear

    x = randn(4, 7)

    f1 = Linear(7, 10, bias=True)
    f2 = SVDLinearParametrization()
    f2.initialize(f1)
    assert allclose(f1.weight, f2.weight, atol=1e-6)
    assert allclose(f1.forward(x), f2.forward(x), atol=1e-6)

    f2 = parametrize_module(copy.deepcopy(f1), SVDLinearParametrization())
    assert allclose(f1.weight, f2.weight, atol=1e-6)
    assert allclose(f1.bias, f2.bias)

    assert allclose(f1.forward(x), f2.forward(x), atol=1e-6)

    f2 = unparametrize_module(f2)
    assert allclose(f1.weight, f2.weight, atol=1e-6)
    assert allclose(f1.bias, f2.bias)
    assert allclose(f1.forward(x), f2.forward(x), atol=1e-6)

    f2 = parametrize_module(copy.deepcopy(f1), SVDLinearParametrization(compression_criterion=ThresholdCriterion(2.0)))
    f2.parametrization.mask.data += torch.randn_like(f2.parametrization.mask.data)
    k = f2.parametrization.compression_criterion(f2.parametrization.mask).sum()
    f2.parametrization.reset_target_params(mode="compress")
    assert f2.parametrization.base.out_features == k
    assert f2.parametrization.ortho.in_features == k
    assert f2.parametrization.mask.numel() == k
