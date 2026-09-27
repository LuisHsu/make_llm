"""engine.py 的測試：同一個運算式分別用 Value 和 PyTorch（float64）計算，
比對 forward 的值與每個輸入的梯度（rel_tol 1e-9、abs_tol 1e-12）。介面約定見 README.md。

測試內容：

| 測試                                   | 運算式與輸入                                                  |
|----------------------------------------|---------------------------------------------------------------|
| test_add／mul／sub／div                | a + b、a * b、a − b（2.0, −3.0）；a / b（2.0, −4.0）           |
| test_neg                               | −a（1.5）                                                     |
| test_pow[2／3／0.5／-1／-2.5]          | a ** k（1.7）                                                 |
| test_scalar_on_right                   | (a + 1) * 2 − 3 / 4（0.8）                                    |
| test_scalar_on_left                    | 1 + 2 * a（0.8）                                              |
| test_scalar_rsub／rtruediv             | 1 − a、3 / a（0.8）                                           |
| test_tanh[-1.2／0.3／2.0]              | a.tanh()                                                      |
| test_relu[-1.2／0.3]                   | a.relu()                                                      |
| test_exp[-1.2／0.0／1.5]               | a.exp()                                                       |
| test_same_value_added_to_itself        | a + a（3.0）                                                  |
| test_same_value_multiplied_by_itself   | a * a（3.0）                                                  |
| test_intermediate_used_by_several_nodes| 中間節點被好幾個節點使用（0.6, −1.1）                         |
| test_deep_composite_expression         | 多層組合，含 tanh、relu、exp、次方、與數字混用（0.4, −0.7, 1.3） |
| test_output_grad_is_one_after_backward | backward 之後輸出本身的 grad 為 1                             |
| test_operations_do_not_modify_input_data | 運算不改動輸入的 data                                       |
"""

import math

import pytest
import torch

from engine import Value


def check(f, *inputs):
    """f 只用 Value 與 torch.Tensor 共通的運算子／方法，所以同一個 f 兩邊都能跑。"""
    vs = [Value(x) for x in inputs]
    out = f(*vs)
    out.backward()

    ts = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in inputs]
    t_out = f(*ts)
    t_out.backward()

    assert isinstance(out, Value)
    assert math.isclose(out.data, t_out.item(), rel_tol=1e-9, abs_tol=1e-12)
    for i, (v, t) in enumerate(zip(vs, ts)):
        assert math.isclose(v.grad, t.grad.item(), rel_tol=1e-9, abs_tol=1e-12), (
            f"input {i}: got grad {v.grad}, expected {t.grad.item()}"
        )


# ---- 基本運算 ----

def test_add():
    check(lambda a, b: a + b, 2.0, -3.0)


def test_mul():
    check(lambda a, b: a * b, 2.0, -3.0)


def test_sub():
    check(lambda a, b: a - b, 2.0, -3.0)


def test_neg():
    check(lambda a: -a, 1.5)


def test_div():
    check(lambda a, b: a / b, 2.0, -4.0)


@pytest.mark.parametrize("k", [2, 3, 0.5, -1, -2.5])
def test_pow(k):
    check(lambda a: a**k, 1.7)


# ---- 與 Python 數字混用 ----

def test_scalar_on_right():
    check(lambda a: (a + 1) * 2 - 3 / 4, 0.8)


def test_scalar_on_left():
    check(lambda a: 1 + 2 * a, 0.8)


def test_scalar_rsub():
    check(lambda a: 1 - a, 0.8)


def test_scalar_rtruediv():
    check(lambda a: 3 / a, 0.8)


# ---- 非線性 ----

@pytest.mark.parametrize("x", [-1.2, 0.3, 2.0])
def test_tanh(x):
    check(lambda a: a.tanh(), x)


@pytest.mark.parametrize("x", [-1.2, 0.3])
def test_relu(x):
    check(lambda a: a.relu(), x)


@pytest.mark.parametrize("x", [-1.2, 0.0, 1.5])
def test_exp(x):
    check(lambda a: a.exp(), x)


# ---- 計算圖結構 ----

def test_same_value_added_to_itself():
    check(lambda a: a + a, 3.0)


def test_same_value_multiplied_by_itself():
    check(lambda a: a * a, 3.0)


def test_intermediate_used_by_several_nodes():
    def f(a, b):
        c = a * b
        d = c + a
        e = c * d
        return e + d.tanh() - c

    check(f, 0.6, -1.1)


def test_deep_composite_expression():
    def f(a, b, c):
        x = a * b + c
        y = x.tanh() * a - b**2
        z = (y + x).relu() + (x * y).exp() / (c**2 + 1)
        w = z * z + 2 * y - x / 3
        return w.tanh() + w * 0.1

    check(f, 0.4, -0.7, 1.3)


# ---- backward 本身 ----

def test_output_grad_is_one_after_backward():
    a = Value(2.0)
    b = Value(3.0)
    out = a * b + a
    out.backward()
    assert out.grad == 1.0


def test_operations_do_not_modify_input_data():
    a = Value(2.0)
    b = Value(-3.0)
    _ = (a * b + a**2).tanh()
    assert a.data == 2.0
    assert b.data == -3.0
