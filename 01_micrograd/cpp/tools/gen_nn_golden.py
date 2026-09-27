"""產生 C++ nn 測試用的 golden 值：用 PyTorch float64 算出固定權重下的輸出、loss 與每個參數的梯度。

執行：uv run python 01_micrograd/cpp/tools/gen_nn_golden.py

輸出：01_micrograd/cpp/tests/golden/nn_cases.json

因為權重是隨機初始化的，沒辦法直接比對；做法是先把 golden 檔裡的 params 寫進模型
（`parameters()` 回傳的 Value 共用節點，直接寫 `p.data()` 就會改到真正的參數），
再比對前向輸出與 backward 之後的參數梯度。

這裡固定假設每個 neuron 都套 tanh（和 01_micrograd/nn.py 一樣）；換成別的非線性
函數 golden 值就不會吻合。
"""

import json
import random
from pathlib import Path

import torch

# 每個 neuron 的非線性函數。改這個就要重新產生 golden 值。
NONLINEARITY = "tanh"


def layer_sizes(nin: int, nouts: list[int]) -> list[int]:
    return [nin] + list(nouts)


def param_count(nin: int, nouts: list[int]) -> int:
    sizes = layer_sizes(nin, nouts)
    return sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(nouts)))


def forward(params: list, nin: int, nouts: list[int], xs: list[float]) -> list:
    """參數的排列順序：層 → neuron → 該 neuron 的 nin 個權重 → bias。

    累加順序寫成 bias + w0*x0 + w1*x1 + ...，和 nn.py 的
    `sum((i * w for ...), self.bias)` 一致，避免浮點累加順序造成差異。
    """
    sizes = layer_sizes(nin, nouts)
    k = 0
    act = list(xs)
    for li in range(len(nouts)):
        n_in, n_out = sizes[li], sizes[li + 1]
        nxt = []
        for _ in range(n_out):
            ws = params[k : k + n_in]
            k += n_in
            acc = params[k]
            k += 1
            for w, a in zip(ws, act):
                acc = acc + w * a
            nxt.append(torch.tanh(acc))
        act = nxt
    assert k == len(params), "參數沒有用完，排列順序對不上"
    return act


# (name, kind, nin, nouts, inputs, targets)
# kind 決定 C++ 測試要用哪個類別：neuron / layer / mlp
CASES = [
    (
        "neuron_3",
        "neuron",
        3,
        [1],
        [[1.0, -2.0, 0.5], [-0.3, 0.8, 2.0]],
        [[1.0], [-1.0]],
    ),
    (
        "layer_3x4",
        "layer",
        3,
        [4],
        [[1.0, -2.0, 0.5], [0.25, 0.0, -1.5]],
        [[1.0, -1.0, 0.5, 0.0], [-0.5, 0.5, -1.0, 1.0]],
    ),
    (
        "layer_3x1",
        "layer",
        3,
        [1],
        [[0.7, -0.2, 1.1]],
        [[-1.0]],
    ),
    (
        "mlp_1_1",
        "mlp",
        1,
        [1],
        [[0.9], [-0.4]],
        [[0.5], [-0.5]],
    ),
    (
        "mlp_2_3_2",
        "mlp",
        2,
        [3, 2],
        [[0.5, -1.0], [-1.5, 2.0], [0.0, 0.25]],
        [[1.0, -1.0], [-1.0, 1.0], [0.5, 0.5]],
    ),
    (
        # 和 01_micrograd/training.py 同樣的形狀與資料
        "mlp_3_4_4_1",
        "mlp",
        3,
        [4, 4, 1],
        [
            [2.0, 3.5, 4.0],
            [-1.0, 3.0, 0.5],
            [1.0, -2.0, -1.0],
            [0.5, 1.5, -2.5],
        ],
        [[1.0], [0.5], [-1.0], [1.0]],
    ),
]


def main() -> None:
    out_path = Path(__file__).resolve().parents[1] / "tests" / "golden" / "nn_cases.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cases = []
    for name, kind, nin, nouts, inputs, targets in CASES:
        n_params = param_count(nin, nouts)
        # 用 case 名稱當 seed，每個 case 的權重固定且互不相同
        rng = random.Random(name)
        init = [rng.uniform(-1.0, 1.0) for _ in range(n_params)]
        params = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in init]

        outputs = []
        loss = torch.tensor(0.0, dtype=torch.float64)
        for x, t in zip(inputs, targets):
            out = forward(params, nin, nouts, x)
            assert len(out) == len(t), f"{name}：target 長度和輸出數不合"
            outputs.append([o.item() for o in out])
            for o, tv in zip(out, t):
                loss = loss + (o - tv) ** 2

        loss.backward()
        grads = [p.grad.item() for p in params]
        assert all(g is not None for g in grads), name

        cases.append(
            {
                "name": name,
                "kind": kind,
                "nin": nin,
                "nouts": nouts,
                "params": init,
                "inputs": inputs,
                "targets": targets,
                "outputs": outputs,
                "loss": loss.item(),
                "param_grads": grads,
            }
        )

    names = [c["name"] for c in cases]
    assert len(set(names)) == len(names), "case 名稱重複"

    out_path.write_text(
        json.dumps(
            {"dtype": "float64", "nonlinearity": NONLINEARITY, "cases": cases},
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"{len(cases)} cases -> {out_path}")


if __name__ == "__main__":
    main()
