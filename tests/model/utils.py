import torch
from torch import nn


def deep_compare(a, b, path="root"):
    """
    Recursively compare two objects. If both have a to_dict() method, compare their dict representations.
    Otherwise, if they are dicts, lists/tuples, or objects with __dict__, recursively compare their contents.
    """
    # Check type equality
    if type(a) != type(b):
        raise AssertionError(f"Type mismatch at {path}: {type(a)} vs {type(b)}")

    # Use to_dict() if available
    if hasattr(a, "to_dict") and callable(a.to_dict) and hasattr(b, "to_dict") and callable(b.to_dict):
        return deep_compare(a.to_dict(), b.to_dict(), path + ".to_dict()")

    # Compare dictionaries
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            raise AssertionError(f"Dict keys mismatch at {path}: {set(a.keys())} vs {set(b.keys())}")
        for key in a:
            if key != "model_type":  # Ignore model_type because it might be just a class attribute
                deep_compare(a[key], b[key], f"{path}.{key}")
        return

    # Compare lists or tuples
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError(f"Length mismatch at {path}: {len(a)} vs {len(b)}")
        for idx, (item_a, item_b) in enumerate(zip(a, b)):
            deep_compare(item_a, item_b, f"{path}[{idx}]")
        return

    # Compare objects with __dict__
    if hasattr(a, "__dict__") and hasattr(b, "__dict__"):
        deep_compare(a.__dict__, b.__dict__, path + ".__dict__")
        return

    # Fallback to direct comparison
    if a != b:
        raise AssertionError(f"Mismatch at {path}: {a} != {b}")


@torch.no_grad()
def model_compare(
    model1: nn.Module,
    model2: nn.Module,
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> None:
    """Tests if the parameters and buffers of two models are the same."""
    # check if all weights are the same
    for name, param1 in model1.named_parameters():
        param2 = model2.get_parameter(name)
        assert torch.allclose(param1, param2, atol=atol, rtol=rtol), f"Parameter {name} is not the same"

    # check if all buffers are the same
    for name, param1 in model1.named_buffers():
        param2 = model2.get_buffer(name)
        assert torch.allclose(param1, param2, atol=atol, rtol=rtol), f"Buffer {name} is not the same"


@torch.no_grad()
def forward_compare(
    model1: nn.Module,
    model2: nn.Module,
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> None:
    """Tests if the forward pass of two models is the same."""
    # test forward pass for dummy input
    batch_size, sequence_length = 1, 20
    dummy_input = torch.randint(0, 50000, (batch_size, sequence_length)).to(model1.device)
    output1 = model1(dummy_input)
    output2 = model2(dummy_input)

    print(f"Max difference: {torch.max(torch.abs(output1.logits - output2.logits))}")
    print(f"Mean difference: {torch.mean(torch.abs(output1.logits - output2.logits))}")

    assert torch.allclose(output1.logits, output2.logits, atol=atol, rtol=rtol), "Output is not the same"
