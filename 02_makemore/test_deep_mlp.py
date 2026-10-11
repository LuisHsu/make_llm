"""deep_mlp.py 的測試（第 3 段）：只檢查外部行為，不限制內部寫法。介面約定見 README.md。

BatchNorm1d 以 torch.nn.BatchNorm1d 當正確答案比對。模型的訓練只有一項測試（test_train_deep_beats_bigram），
標記為 train，預設跳過，加 --run-train 才執行。

測試內容：

| 測試                                         | 檢查什麼                                                                |
|----------------------------------------------|-------------------------------------------------------------------------|
| test_linear_forward[bias／no_bias]           | 輸出 = x @ weight + bias；形狀；.out；parameters() 的內容與順序          |
| test_linear_same_generator_same_init         | 同 seed 的 generator 得到相同權重，不受全域亂數影響；不同 seed 不同       |
| test_tanh                                    | 輸出 = tanh(x)；.out；parameters() 是空 list                             |
| test_batchnorm_defaults                      | eps、momentum、training 的預設值；gamma／beta／running_* 的形狀與初始值  |
| test_batchnorm_train_forward                 | 訓練模式的輸出與 torch 相同（隨機 gamma／beta）；.out；不改動輸入         |
| test_batchnorm_running_stats                 | 連續 3 個 batch 後 running_mean／running_var 的形狀仍是 (dim,)、與 torch 相同 |
| test_batchnorm_custom_eps_momentum           | 自訂 eps、momentum 時輸出與 running stats 與 torch 相同                  |
| test_batchnorm_running_stats_no_grad         | 輸入與參數需要梯度時，running stats 不帶梯度                             |
| test_batchnorm_eval                          | eval 模式用 running stats、與 torch 相同、不更新 running stats、n=1 可用 |
| test_batchnorm_backward                      | 對 x、gamma、beta 的梯度與 torch 相同                                     |
| test_batchnorm_parameters                    | parameters() 是 [gamma, beta]，不含 running stats                         |
| test_init_structure[0／1]                    | model.layers 的型別；至少 3 個 Tanh，每個前面緊接 BatchNorm1d；輸出形狀   |
| test_init_same_seed_same_model               | 同 seed 輸出相同，不受全域亂數影響；不同 seed 不同                        |
| test_init_loss_near_log27[0／1]              | 訓練前 dev NLL 與 ln 27 相差 < 0.1                                        |
| test_init_tanh_not_saturated[0／1]           | 訓練前每個 Tanh 層輸出絕對值 > 0.97 的比例 < 10%                        |
| test_train_deep_beats_bigram（train，預設跳過）| eval 模式逐列獨立；dev NLL ≤ bigram 在 dev 上的 NLL − 0.25；可交給 sample |

門檻來源（Claude 用 scratchpad 的參考實作量，參考實作不進 repo）：參考實作照影片後半的做法（embedding 10、
5 個隱藏層、每層 100、BatchNorm、batch 32、有學習率衰減），split 用學習者的 split_words(words, 42)：
- 訓練前（seed 0、1、2）：dev NLL 3.294–3.307（ln 27 = 3.296），各 Tanh 層飽和比例 0.029–0.039
- 對照：權重 randn 不縮放且沒有 BatchNorm 時，dev NLL 3.66–3.71，飽和比例 0.69–0.84
- 訓練：2 萬步 19.4 s，dev 2.1719（bigram 2.4647，低 0.29）；5 萬步 48.4 s，dev 2.1260（低 0.34）
- 第 2 段學習者的 MLP（2 萬步、無學習率衰減）dev 2.2520，達不到這一段的門檻

負控（參考實作故意改錯，各自應失敗的測試都失敗）：正規化用 unbiased variance、running_var 用 biased、
不更新 running stats、eval 仍用 batch 統計量、Linear 用全域亂數、輸出層不縮小、BatchNorm gamma 放大 3 倍、
不用 BatchNorm；訓練測試另測 1000 步、不更新 running stats、eval 用 batch 統計量，三者皆失敗。
"""

import math

import pytest
import torch

from deep_mlp import BatchNorm1d, Linear, Tanh, init_deep
from mlp import build_dataset, evaluate, sample, split_words
from test_mlp import ALPHABET, _bigram_dev_nll

ATOL, RTOL = 1e-5, 1e-4
TRAIN_SPLIT_SEED = 42
MARGIN = 0.25
SATURATION = 0.97  # |tanh| 超過這個值算飽和
MAX_SATURATED = 0.10
MAX_INIT_GAP = 0.1


def _close(a, b):
    return torch.allclose(a, b, atol=ATOL, rtol=RTOL)


def _torch_bn(dim, gamma=None, beta=None, eps=1e-5, momentum=0.1):
    ref = torch.nn.BatchNorm1d(dim, eps=eps, momentum=momentum)
    with torch.no_grad():
        if gamma is not None:
            ref.weight.copy_(gamma)
        if beta is not None:
            ref.bias.copy_(beta)
    return ref


def _set_params(bn, gamma, beta):
    with torch.no_grad():
        bn.gamma.copy_(gamma)
        bn.beta.copy_(beta)


def _rand(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


@pytest.fixture(scope="module")
def split(words):
    train, dev, _ = split_words(words, TRAIN_SPLIT_SEED)
    return train, dev


def _tanh_layers(model):
    return [l for l in model.layers if isinstance(l, Tanh)]


def _set_training(model, flag):
    for l in model.layers:
        if isinstance(l, BatchNorm1d):
            l.training = flag


# ---------- Linear ----------


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
def test_linear_forward(bias):
    lin = Linear(4, 3, bias=bias, generator=torch.Generator().manual_seed(0))
    assert tuple(lin.weight.shape) == (4, 3), f"weight 形狀是 {tuple(lin.weight.shape)}，README 約定是 (fan_in, fan_out)"
    W, b = _rand(4, 3, seed=1), _rand(3, seed=2)
    with torch.no_grad():
        lin.weight.copy_(W)
        if bias:
            assert tuple(lin.bias.shape) == (3,)
            lin.bias.copy_(b)
        else:
            assert lin.bias is None, "bias=False 時 .bias 要是 None"
    x = _rand(5, 4, seed=3)
    out = lin(x)
    expected = x @ W + (b if bias else 0)
    assert tuple(out.shape) == (5, 3)
    assert _close(out, expected)
    assert lin.out is out, ".out 要是這次呼叫的輸出"
    params = lin.parameters()
    assert isinstance(params, list)
    expected_params = [lin.weight, lin.bias] if bias else [lin.weight]
    assert len(params) == len(expected_params) and all(p is q for p, q in zip(params, expected_params)), (
        "parameters() 要依序是 [weight, bias]（bias=False 時只有 [weight]）"
    )


def test_linear_same_generator_same_init():
    torch.manual_seed(1)
    a = Linear(6, 5, generator=torch.Generator().manual_seed(7))
    torch.manual_seed(2)
    b = Linear(6, 5, generator=torch.Generator().manual_seed(7))
    c = Linear(6, 5, generator=torch.Generator().manual_seed(8))
    assert torch.equal(a.weight, b.weight) and torch.equal(a.bias, b.bias), "同 seed 的 generator 要得到相同權重"
    assert not torch.equal(a.weight, c.weight), "不同 seed 的 generator 應該得到不同權重"


# ---------- Tanh ----------


def test_tanh():
    t = Tanh()
    x = _rand(5, 4)
    out = t(x)
    assert _close(out, torch.tanh(x))
    assert t.out is out
    assert t.parameters() == []


# ---------- BatchNorm1d ----------


def test_batchnorm_defaults():
    bn = BatchNorm1d(7)
    assert bn.eps == 1e-5 and bn.momentum == 0.1
    assert bn.training is True
    for name, value in [("gamma", 1.0), ("beta", 0.0), ("running_mean", 0.0), ("running_var", 1.0)]:
        t = getattr(bn, name)
        assert tuple(t.shape) == (7,), f"{name} 的形狀是 {tuple(t.shape)}，README 約定是 (dim,)"
        assert torch.equal(t, torch.full((7,), value)), f"{name} 的初始值要全部是 {value}"


def test_batchnorm_train_forward():
    gamma, beta = _rand(6, seed=1), _rand(6, seed=2)
    bn = BatchNorm1d(6)
    _set_params(bn, gamma, beta)
    ref = _torch_bn(6, gamma, beta)
    x = _rand(32, 6, seed=3) * 3 + 2
    x0 = x.clone()
    out = bn(x)
    assert _close(out, ref(x)), "訓練模式的輸出和 torch.nn.BatchNorm1d 不同（注意正規化用的是哪一種 variance）"
    assert bn.out is out
    assert torch.equal(x, x0), "不能改動輸入"


def test_batchnorm_running_stats():
    bn, ref = BatchNorm1d(6), _torch_bn(6)
    for s in range(3):
        x = _rand(32, 6, seed=10 + s) * (s + 1) + s
        bn(x)
        ref(x)
    # _close 會 broadcast，(1, dim) 和 (dim,) 也會被當成相同，所以形狀要另外檢查
    for name in ["running_mean", "running_var"]:
        shape = tuple(getattr(bn, name).shape)
        assert shape == (6,), f"更新後 {name} 的形狀是 {shape}，README 約定是 (dim,)"
    assert _close(bn.running_mean, ref.running_mean), "running_mean 和 torch 不同"
    assert _close(bn.running_var, ref.running_var), "running_var 和 torch 不同（注意更新時用的是哪一種 variance）"


def test_batchnorm_custom_eps_momentum():
    bn, ref = BatchNorm1d(4, eps=1e-3, momentum=0.3), _torch_bn(4, eps=1e-3, momentum=0.3)
    for s in range(2):
        x = _rand(8, 4, seed=20 + s) * 0.05  # variance 小，eps 的影響才看得出來
        assert _close(bn(x), ref(x))
    assert _close(bn.running_mean, ref.running_mean) and _close(bn.running_var, ref.running_var)


def test_batchnorm_running_stats_no_grad():
    bn = BatchNorm1d(5)
    bn.gamma.requires_grad_(True)
    bn.beta.requires_grad_(True)
    x = _rand(16, 5).requires_grad_(True)
    bn(x).sum().backward()
    assert not bn.running_mean.requires_grad and not bn.running_var.requires_grad, (
        "running stats 不能帶梯度（更新時不能被 autograd 追蹤）"
    )


def test_batchnorm_eval():
    gamma, beta = _rand(6, seed=1), _rand(6, seed=2)
    bn = BatchNorm1d(6)
    _set_params(bn, gamma, beta)
    ref = _torch_bn(6, gamma, beta)
    for s in range(3):
        x = _rand(32, 6, seed=30 + s) * 2 - 1
        bn(x)
        ref(x)
    bn.training = False
    ref.eval()
    mean0, var0 = bn.running_mean.clone(), bn.running_var.clone()
    x = _rand(10, 6, seed=40) * 5 + 3
    assert _close(bn(x), ref(x)), "eval 模式的輸出和 torch 不同（eval 要用 running stats，不是這個 batch 的統計量）"
    assert torch.equal(bn.running_mean, mean0) and torch.equal(bn.running_var, var0), "eval 模式不能更新 running stats"
    out1 = bn(x[:1])
    assert tuple(out1.shape) == (1, 6), "eval 模式要能處理 n = 1"
    assert _close(out1, ref(x[:1]))


def test_batchnorm_backward():
    gamma, beta = _rand(6, seed=1), _rand(6, seed=2)
    bn = BatchNorm1d(6)
    _set_params(bn, gamma, beta)
    bn.gamma.requires_grad_(True)
    bn.beta.requires_grad_(True)
    ref = _torch_bn(6, gamma, beta)
    x = _rand(32, 6, seed=3) * 3 + 2
    xa, xb = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    w = _rand(32, 6, seed=4)  # 讓每個輸出的上游梯度不同
    (bn(xa) * w).sum().backward()
    (ref(xb) * w).sum().backward()
    assert _close(xa.grad, xb.grad), "對 x 的梯度和 torch 不同"
    assert _close(bn.gamma.grad, ref.weight.grad), "對 gamma 的梯度和 torch 不同"
    assert _close(bn.beta.grad, ref.bias.grad), "對 beta 的梯度和 torch 不同"


def test_batchnorm_parameters():
    bn = BatchNorm1d(3)
    params = bn.parameters()
    assert isinstance(params, list) and len(params) == 2
    assert params[0] is bn.gamma and params[1] is bn.beta, "parameters() 要依序是 [gamma, beta]，不含 running stats"


# ---------- 模型：訓練前 ----------


@pytest.mark.parametrize("seed", [0, 1])
def test_init_structure(split, seed):
    model, block_size = init_deep(seed)
    assert type(block_size) is int and block_size >= 1, f"block_size 要是正整數，拿到 {block_size!r}"
    layers = model.layers
    assert isinstance(layers, list) and layers, "model.layers 要是非空的 list"
    bad = [type(l).__name__ for l in layers if not isinstance(l, (Linear, BatchNorm1d, Tanh))]
    assert not bad, f"model.layers 只能放 deep_mlp 的 Linear／BatchNorm1d／Tanh，拿到 {bad}"
    tanh_idx = [i for i, l in enumerate(layers) if isinstance(l, Tanh)]
    assert len(tanh_idx) >= 3, f"至少要 3 個 Tanh 層，只有 {len(tanh_idx)} 個"
    assert all(i > 0 and isinstance(layers[i - 1], BatchNorm1d) for i in tanh_idx), (
        "每個 Tanh 前面要緊接著一個 BatchNorm1d"
    )
    _, dev = split
    Xd, _ = build_dataset(dev[:20], block_size)
    with torch.no_grad():
        out = model(Xd)
    assert tuple(out.shape) == (Xd.shape[0], 27), f"model(X) 的形狀是 {tuple(out.shape)}，README 約定是 (n, 27)"
    assert out.device.type == "cpu"


def test_init_same_seed_same_model(split):
    _, dev = split
    torch.manual_seed(1)
    a, bs = init_deep(0)
    torch.manual_seed(2)
    b, _ = init_deep(0)
    c, _ = init_deep(1)
    Xd, _ = build_dataset(dev[:200], bs)
    with torch.no_grad():
        oa, ob, oc = a(Xd), b(Xd), c(Xd)
    assert torch.equal(oa, ob), "同 seed 要得到相同的模型，而且不受全域亂數狀態影響"
    assert not torch.equal(oa, oc), "不同 seed 應該得到不同的模型"


@pytest.mark.parametrize("seed", [0, 1])
def test_init_loss_near_log27(split, seed):
    _, dev = split
    model, bs = init_deep(seed)
    _set_training(model, True)
    Xd, Yd = build_dataset(dev, bs)
    nll = evaluate(model, Xd, Yd)
    print(f"\n訓練前 dev NLL {nll:.4f}；ln 27 = {math.log(27):.4f}")
    assert abs(nll - math.log(27)) < MAX_INIT_GAP, (
        f"訓練前 dev NLL {nll:.4f}，和 ln 27 = {math.log(27):.4f} 相差 {abs(nll - math.log(27)):.4f}，要 < {MAX_INIT_GAP}"
    )


@pytest.mark.parametrize("seed", [0, 1])
def test_init_tanh_not_saturated(split, seed):
    _, dev = split
    model, bs = init_deep(seed)
    _set_training(model, True)
    Xd, _ = build_dataset(dev, bs)
    with torch.no_grad():
        model(Xd)
    fracs = [(l.out.abs() > SATURATION).float().mean().item() for l in _tanh_layers(model)]
    print("\n各 Tanh 層飽和比例", [f"{f:.3f}" for f in fracs])
    assert all(f < MAX_SATURATED for f in fracs), (
        f"各 Tanh 層 |out| > {SATURATION} 的比例 {[round(f, 3) for f in fracs]}，每一層都要 < {MAX_SATURATED}"
    )


# ---------- 模型：訓練（預設跳過） ----------


@pytest.mark.train
def test_train_deep_beats_bigram(split):
    from deep_mlp import train_deep

    train, dev = split
    model, block_size = train_deep(train, 0)
    assert type(block_size) is int and block_size >= 1, f"block_size 要是正整數，拿到 {block_size!r}"
    _set_training(model, False)

    Xd, Yd = build_dataset(dev, block_size)
    with torch.no_grad():
        out5 = model(Xd[:5])
        out1 = model(Xd[:1])
    assert tuple(out5.shape) == (5, 27), f"model(X) 的形狀是 {tuple(out5.shape)}，README 約定是 (n, 27)"
    assert out5.device.type == "cpu", f"model(X) 回傳的 tensor 在 {out5.device}，README 約定要在 CPU 上"
    assert _close(out1, out5[:1]), (
        "eval 模式下，同一列的輸出不能因為同一批裡有其他列而改變（BatchNorm1d 的 eval 要用 running stats）"
    )

    dev_nll = evaluate(model, Xd, Yd)
    base = _bigram_dev_nll(train, dev)
    print(f"\n深層 MLP dev NLL {dev_nll:.4f}；bigram dev NLL {base:.4f}；門檻 {base - MARGIN:.4f}")
    assert dev_nll <= base - MARGIN, (
        f"dev NLL {dev_nll:.4f}，要 ≤ {base - MARGIN:.4f}（bigram 在 dev 上的 {base:.4f} − {MARGIN}）"
    )

    names = sample(model, block_size, 5, torch.Generator().manual_seed(0))
    assert len(names) == 5 and all(set(s) <= set(ALPHABET) for s in names)
