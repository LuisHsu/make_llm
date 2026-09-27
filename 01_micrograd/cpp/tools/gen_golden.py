"""產生 C++ 測試用的 golden 值：用 PyTorch float64 算出每個運算式的輸出與各輸入的梯度。

執行：uv run python 01_micrograd/cpp/tools/gen_golden.py

輸出：01_micrograd/cpp/tests/golden/engine_cases.json
每個 case 的 expr 欄只是給人看的說明，C++ 測試要自己把同一個運算式寫一次，
再拿 out 與 grads 比對。
"""

import json
from pathlib import Path

import torch


def shared_intermediate(a, b):
    c = a * b
    d = c + a
    e = c * d
    return e + d.tanh() - c


def deep_composite(a, b, c):
    x = a * b + c
    y = x.tanh() * a - b**2
    z = (y + x).relu() + (x * y).exp() / (c**2 + 1)
    w = z * z + 2 * y - x / 3
    return w.tanh() + w * 0.1


# (name, 說明用的運算式, 輸入, 用 torch tensor 計算的函式)
CASES = [
    ("add", "a + b", [2.0, -3.0], lambda a, b: a + b),
    ("mul", "a * b", [2.0, -3.0], lambda a, b: a * b),
    ("sub", "a - b", [2.0, -3.0], lambda a, b: a - b),
    ("div", "a / b", [2.0, -4.0], lambda a, b: a / b),
    ("neg", "-a", [1.5], lambda a: -a),
    ("pow_int", "a ** 3", [1.7], lambda a: a**3),
    ("pow_half", "a ** 0.5", [1.7], lambda a: a**0.5),
    ("pow_negative", "a ** -2.5", [1.7], lambda a: a**-2.5),
    ("scalar_right", "(a + 1) * 2 - 3 / 4", [0.8], lambda a: (a + 1) * 2 - 3 / 4),
    ("scalar_left", "1 + 2 * a", [0.8], lambda a: 1 + 2 * a),
    ("scalar_rsub", "1 - a", [0.8], lambda a: 1 - a),
    ("scalar_rtruediv", "3 / a", [0.8], lambda a: 3 / a),
    ("tanh_negative", "tanh(a)", [-1.2], lambda a: a.tanh()),
    ("tanh_small", "tanh(a)", [0.3], lambda a: a.tanh()),
    ("tanh_large", "tanh(a)", [2.0], lambda a: a.tanh()),
    ("relu_negative", "relu(a)", [-1.2], lambda a: a.relu()),
    ("relu_positive", "relu(a)", [0.3], lambda a: a.relu()),
    ("relu_zero", "relu(a)", [0.0], lambda a: a.relu()),
    ("exp_negative", "exp(a)", [-1.2], lambda a: a.exp()),
    ("exp_zero", "exp(a)", [0.0], lambda a: a.exp()),
    ("exp_positive", "exp(a)", [1.5], lambda a: a.exp()),
    ("reuse_add", "a + a", [3.0], lambda a: a + a),
    ("reuse_mul", "a * a", [3.0], lambda a: a * a),
    (
        "shared_intermediate",
        "c = a * b; d = c + a; e = c * d; e + tanh(d) - c",
        [0.6, -1.1],
        shared_intermediate,
    ),
    (
        "deep_composite",
        "x = a*b + c; y = tanh(x)*a - b**2; z = relu(y + x) + exp(x*y)/(c**2 + 1); w = z*z + 2*y - x/3; tanh(w) + w*0.1",
        [0.4, -0.7, 1.3],
        deep_composite,
    ),
]


def main() -> None:
    out_path = Path(__file__).resolve().parents[1] / "tests" / "golden" / "engine_cases.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cases = []
    for name, expr, inputs, fn in CASES:
        ts = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in inputs]
        out = fn(*ts)
        out.backward()
        grads = [t.grad.item() for t in ts]
        assert all(g is not None for g in grads), name
        cases.append(
            {
                "name": name,
                "expr": expr,
                "inputs": inputs,
                "out": out.item(),
                "grads": grads,
            }
        )

    names = [c["name"] for c in cases]
    assert len(set(names)) == len(names), "case 名稱重複"

    out_path.write_text(
        json.dumps({"dtype": "float64", "cases": cases}, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"{len(cases)} cases -> {out_path}")


if __name__ == "__main__":
    main()
