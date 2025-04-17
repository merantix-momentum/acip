def test_new_linear_from_mask():
    from torch import BoolTensor, nn

    from acip.core.projected_layer import new_linear_from_mask

    f = nn.Linear(3, 7, bias=True)
    mask = BoolTensor([True, False, False])
    new_f = new_linear_from_mask(f, mask)
    assert new_f.in_features == 1

    mask = BoolTensor([True, False, False, True, True, True, False])
    new_f = new_linear_from_mask(f, mask, column_select=False)
    assert new_f.out_features == 4
