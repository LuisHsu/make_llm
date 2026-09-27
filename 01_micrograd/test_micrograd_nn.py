"""nn.py 的測試：只檢查外部行為，不限制內部寫法（例如用 tanh 還是 relu）。介面約定見 README.md。

測試內容：

| 測試                              | 檢查什麼                                                           |
|-----------------------------------|--------------------------------------------------------------------|
| test_neuron_parameters_and_output | Neuron(3)：4 個參數，都是 Value；輸出是 Value                      |
| test_layer_parameters_and_output  | Layer(3, 4)：16 個參數；輸出是長度 4 的 Value list                 |
| test_layer_single_output_is_value | Layer(3, 1) 的輸出是 Value，不是 list                              |
| test_mlp_parameters_and_output    | MLP(3, [4, 4, 1])：41 個參數且彼此是不同物件；輸出是 Value          |
| test_mlp_accepts_value_inputs     | 輸入 list 的元素是 Value 時也能呼叫                                |
| test_training_reduces_loss        | random.seed(0)、4 筆資料、平方誤差和、lr 0.01、50 步後 loss 下降  |
"""

import random

from engine import Value
from nn import MLP, Layer, Neuron


def test_neuron_parameters_and_output():
    n = Neuron(3)
    ps = n.parameters()
    assert len(ps) == 3 + 1
    assert all(isinstance(p, Value) for p in ps)
    assert isinstance(n([1.0, -2.0, 0.5]), Value)


def test_layer_parameters_and_output():
    layer = Layer(3, 4)
    assert len(layer.parameters()) == 4 * (3 + 1)
    out = layer([1.0, -2.0, 0.5])
    assert isinstance(out, list) and len(out) == 4
    assert all(isinstance(o, Value) for o in out)


def test_layer_single_output_is_value():
    assert isinstance(Layer(3, 1)([1.0, -2.0, 0.5]), Value)


def test_mlp_parameters_and_output():
    model = MLP(3, [4, 4, 1])
    assert len(model.parameters()) == 4 * (3 + 1) + 4 * (4 + 1) + 1 * (4 + 1)
    assert len({id(p) for p in model.parameters()}) == len(model.parameters())
    assert isinstance(model([1.0, -2.0, 0.5]), Value)


def test_mlp_accepts_value_inputs():
    model = MLP(2, [3, 1])
    assert isinstance(model([Value(0.5), Value(-1.0)]), Value)


def test_training_reduces_loss():
    random.seed(0)
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]
    model = MLP(3, [4, 4, 1])

    def loss_fn():
        return sum(((model(x) - y) ** 2 for x, y in zip(xs, ys)), Value(0.0))

    first = loss_fn().data
    for _ in range(50):
        for p in model.parameters():
            p.grad = 0.0
        loss = loss_fn()
        loss.backward()
        for p in model.parameters():
            p.data -= 0.01 * p.grad
    last = loss_fn().data

    assert last < first, f"loss 沒有下降：{first:.4f} -> {last:.4f}"
