"""wavenet.py 與 deep_mlp.BatchNorm1d 3 維輸入的測試（第 5 段）：只檢查外部行為，不限制內部寫法。介面約定見 README.md。

BatchNorm1d 的 3 維輸入以 torch.nn.BatchNorm1d 當正確答案比對：torch 吃 (n, C, L)，所以把 (n, L, C) 轉置後丟給它，
輸出再轉置回來。模型的訓練只有一項測試（test_train_wavenet_beats_threshold），標記為 train，預設跳過，加 --run-train
才執行。

測試內容：

| 測試                                        | 檢查什麼                                                                   |
|---------------------------------------------|----------------------------------------------------------------------------|
| test_embedding_forward                      | 輸出 = weight[IX]，1 維與 2 維的 IX；形狀；.out；parameters() 是 [weight]   |
| test_embedding_same_generator_same_init     | 同 seed 的 generator 得到相同權重，不受全域亂數影響；不同 seed 不同          |
| test_flatten_consecutive[...]               | (B, T, C) → (B, T/n, n·C)，相鄰 n 個依時間順序接起來；T/n = 1 時是 (B, n·C)；.out；不改動輸入；parameters() 是 [] |
| test_sequential                             | 依序呼叫每一層；.layers 就是傳入的 list；parameters() 依層的順序串接；.out    |
| test_linear_3d                              | deep_mlp.Linear 吃 (n, L, C)：輸出 (n, L, fan_out) = x @ weight + bias       |
| test_batchnorm_3d_train_forward[...]        | 3 維訓練模式的輸出與 torch 相同；.out；不改動輸入                           |
| test_batchnorm_3d_running_stats             | 連續 3 個 batch 後 running stats 形狀仍是 (C,)、與 torch 相同                |
| test_batchnorm_3d_eval                      | 3 維 eval 用 running stats、與 torch 相同、不更新 running stats、n = 1 可用  |
| test_batchnorm_3d_backward                  | 3 維時對 x、gamma、beta 的梯度與 torch 相同                                  |
| test_batchnorm_2d_unchanged                 | 2 維輸入的行為不變（訓練、running stats、eval 與 torch 相同）                |
| test_init_structure[0／1]                   | model.layers 的型別；至少 2 個 FlattenConsecutive、2 個 Tanh，每個 Tanh 前面緊接 BatchNorm1d；輸出形狀 |
| test_init_hierarchical[0／1]                | 第一個 FlattenConsecutive 的輸出是 3 維且 L < block_size（不是一次攤平整個 context） |
| test_init_same_seed_same_model              | 同 seed 輸出相同，不受全域亂數影響；不同 seed 不同                            |
| test_init_loss_near_log27[0／1]             | 訓練前 dev NLL 與 ln 27 相差 < 0.1                                           |
| test_init_tanh_not_saturated[0／1]          | 訓練前每個 Tanh 層輸出絕對值 > 0.97 的比例 < 10%                             |
| test_train_wavenet_beats_threshold（train，預設跳過）| eval 逐列獨立；dev NLL ≤ bigram 在 dev 上的 NLL − MARGIN；可交給 sample |

門檻來源（Claude 用 scratchpad 的參考實作量，參考實作不進 repo）：參考實作照影片（embedding 24、3 層
FlattenConsecutive(2) → Linear(無 bias) → BatchNorm1d → Tanh、hidden 128、block_size 8、batch 32、lr 0.1，最後 1/4
步數 0.01），Linear、BatchNorm1d、Tanh 用學習者的 deep_mlp（BatchNorm1d 換成支援 3 維的參考版），split 用學習者的
split_words(words, 42)，bigram dev 2.4647：
- 訓練前（seed 0、1）：dev NLL 3.2973、3.3010；各 Tanh 層飽和比例 0.039–0.063
- 訓練：每步約 1.15 ms。6 萬步 53 s，dev 2.0041（seed 0）、2.0017（seed 1）；9 萬步 86 s，dev 1.9977；5000 步 dev 2.1420
- 門檻 MARGIN 0.40 → 2.0647：參考實作 6 萬步低 0.06；第 3 段 deep MLP（2.2102）、第 4 段（2.2447）都達不到

負控（參考實作故意改錯，各自應失敗的測試都失敗）：
- BatchNorm1d 3 維只對第 0 維取統計量：3 維的 6 項全部失敗（running stats 形狀變成 (L, C)）
- running_var 用 biased：3d_running_stats、3d_eval、2d_unchanged
- eval 仍用這一批的統計量：3d_eval、2d_unchanged；訓練測試的逐列獨立檢查
- FlattenConsecutive 在 T/n = 1 時不壓成 2 維：flatten 的 2 項、init_structure、init_loss（輸出變成 (n, 1, 27)）
- 先 FlattenConsecutive(1) 再 FlattenConsecutive(8)（一次攤平整個 context）：只有 init_hierarchical
- 最後一層不額外縮小：只有 init_loss
- 只訓練 5000 步：訓練測試的 dev NLL 門檻
"""

import math

import pytest
import torch

from deep_mlp import BatchNorm1d, Linear, Tanh
from mlp import build_dataset, evaluate, sample, split_words
from test_mlp import ALPHABET, _bigram_dev_nll
from wavenet import Embedding, FlattenConsecutive, Sequential, init_wavenet

ATOL, RTOL = 1e-5, 1e-4
TRAIN_SPLIT_SEED = 42
MARGIN = 0.40
SATURATION = 0.97
MAX_SATURATED = 0.10
MAX_INIT_GAP = 0.1
LAYER_TYPES = (Embedding, FlattenConsecutive, Linear, BatchNorm1d, Tanh)


def _close(a, b):
    return torch.allclose(a, b, atol=ATOL, rtol=RTOL)


def _rand(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _torch_bn(dim, gamma=None, beta=None, eps=1e-5, momentum=0.1):
    ref = torch.nn.BatchNorm1d(dim, eps=eps, momentum=momentum)
    with torch.no_grad():
        if gamma is not None:
            ref.weight.copy_(gamma)
        if beta is not None:
            ref.bias.copy_(beta)
    return ref


def _ref3(ref, x):
    """(n, L, C) 丟給 torch.nn.BatchNorm1d：torch 要 (n, C, L)。"""
    return ref(x.transpose(1, 2)).transpose(1, 2)


def _set_params(bn, gamma, beta):
    with torch.no_grad():
        bn.gamma.copy_(gamma)
        bn.beta.copy_(beta)


@pytest.fixture(scope="module")
def split(words):
    train, dev, _ = split_words(words, TRAIN_SPLIT_SEED)
    return train, dev


def _of_type(model, cls):
    return [l for l in model.layers if isinstance(l, cls)]


def _set_training(model, flag):
    for l in model.layers:
        if isinstance(l, BatchNorm1d):
            l.training = flag


# ---------- 新的層 ----------


def test_embedding_forward():
    emb = Embedding(27, 5, generator=torch.Generator().manual_seed(0))
    assert tuple(emb.weight.shape) == (27, 5), f"weight 形狀是 {tuple(emb.weight.shape)}，README 約定是 (num_embeddings, embedding_dim)"
    W = _rand(27, 5, seed=1)
    with torch.no_grad():
        emb.weight.copy_(W)
    IX = torch.tensor([[0, 3, 3], [26, 1, 0]])
    out = emb(IX)
    assert tuple(out.shape) == (2, 3, 5)
    assert torch.equal(out, W[IX])
    assert emb.out is out, ".out 要是這次呼叫的輸出"
    out1 = emb(torch.tensor([4, 4, 9]))
    assert tuple(out1.shape) == (3, 5) and torch.equal(out1, W[[4, 4, 9]]), "1 維的 IX 也要能用"
    params = emb.parameters()
    assert isinstance(params, list) and len(params) == 1 and params[0] is emb.weight, "parameters() 要是 [weight]"


def test_embedding_same_generator_same_init():
    torch.manual_seed(1)
    a = Embedding(27, 4, generator=torch.Generator().manual_seed(7))
    torch.manual_seed(2)
    b = Embedding(27, 4, generator=torch.Generator().manual_seed(7))
    c = Embedding(27, 4, generator=torch.Generator().manual_seed(8))
    assert torch.equal(a.weight, b.weight), "同 seed 的 generator 要得到相同權重"
    assert not torch.equal(a.weight, c.weight), "不同 seed 的 generator 應該得到不同權重"


@pytest.mark.parametrize("B,T,C,n", [(4, 8, 3, 2), (4, 8, 3, 4), (5, 6, 2, 3), (4, 8, 3, 8), (3, 2, 5, 2)])
def test_flatten_consecutive(B, T, C, n):
    x = _rand(B, T, C, seed=T * 10 + n)
    x0 = x.clone()
    fc = FlattenConsecutive(n)
    out = fc(x)
    if T // n == 1:
        expected = torch.cat([x[:, i] for i in range(T)], dim=1)
        assert tuple(out.shape) == (B, n * C), f"T/n = 1 時輸出要是 2 維 {(B, n * C)}，拿到 {tuple(out.shape)}"
    else:
        expected = torch.stack(
            [torch.cat([x[:, t * n + i] for i in range(n)], dim=1) for t in range(T // n)], dim=1
        )
        assert tuple(out.shape) == (B, T // n, n * C), f"輸出形狀要是 {(B, T // n, n * C)}，拿到 {tuple(out.shape)}"
    assert torch.equal(out, expected), "相鄰 n 個時間點的向量要依時間順序接起來"
    assert fc.out is out
    assert torch.equal(x, x0), "不能改動輸入"
    assert fc.parameters() == []


def test_sequential():
    g = torch.Generator().manual_seed(0)
    l1, t, l2 = Linear(4, 6, generator=g), Tanh(), Linear(6, 3, bias=False, generator=g)
    layers = [l1, t, l2]
    seq = Sequential(layers)
    assert isinstance(seq.layers, list) and len(seq.layers) == 3 and all(a is b for a, b in zip(seq.layers, layers)), (
        ".layers 要是傳入的那些層，順序不變"
    )
    x = _rand(5, 4, seed=1)
    out = seq(x)
    expected = l2(t(l1(x)))
    assert _close(out, expected), "要依序呼叫每一層"
    assert seq.out is out
    params = seq.parameters()
    expected_params = l1.parameters() + t.parameters() + l2.parameters()
    assert isinstance(params, list) and len(params) == len(expected_params) and all(
        p is q for p, q in zip(params, expected_params)
    ), "parameters() 要依層的順序把每一層的 parameters() 串起來"


def test_linear_3d():
    lin = Linear(3, 4, generator=torch.Generator().manual_seed(0))
    x = _rand(5, 2, 3, seed=1)
    out = lin(x)
    assert tuple(out.shape) == (5, 2, 4), f"(n, L, C) 輸入的輸出形狀要是 (n, L, fan_out)，拿到 {tuple(out.shape)}"
    assert _close(out, x @ lin.weight + lin.bias)


# ---------- BatchNorm1d：3 維輸入 ----------


@pytest.mark.parametrize("n,L,C", [(32, 4, 6), (2, 3, 5), (1, 4, 3)])
def test_batchnorm_3d_train_forward(n, L, C):
    gamma, beta = _rand(C, seed=1), _rand(C, seed=2)
    bn = BatchNorm1d(C)
    _set_params(bn, gamma, beta)
    ref = _torch_bn(C, gamma, beta)
    x = _rand(n, L, C, seed=3) * 3 + 2
    x0 = x.clone()
    out = bn(x)
    assert tuple(out.shape) == (n, L, C)
    assert _close(out, _ref3(ref, x)), (
        "3 維訓練模式的輸出和 torch 不同（每個 channel 要對前兩維的全部 n × L 個值一起算統計量）"
    )
    assert bn.out is out
    assert torch.equal(x, x0), "不能改動輸入"


def test_batchnorm_3d_running_stats():
    bn, ref = BatchNorm1d(6), _torch_bn(6)
    for s in range(3):
        x = _rand(16, 4, 6, seed=10 + s) * (s + 1) + s
        bn(x)
        _ref3(ref, x)
    for name in ["running_mean", "running_var"]:
        shape = tuple(getattr(bn, name).shape)
        assert shape == (6,), f"3 維輸入更新後 {name} 的形狀是 {shape}，README 約定是 (C,)"
    assert _close(bn.running_mean, ref.running_mean), "running_mean 和 torch 不同"
    assert _close(bn.running_var, ref.running_var), "running_var 和 torch 不同（注意 unbiased 時除以的是幾）"


def test_batchnorm_3d_eval():
    gamma, beta = _rand(6, seed=1), _rand(6, seed=2)
    bn = BatchNorm1d(6)
    _set_params(bn, gamma, beta)
    ref = _torch_bn(6, gamma, beta)
    for s in range(3):
        x = _rand(16, 4, 6, seed=30 + s) * 2 - 1
        bn(x)
        _ref3(ref, x)
    bn.training = False
    ref.eval()
    mean0, var0 = bn.running_mean.clone(), bn.running_var.clone()
    x = _rand(10, 4, 6, seed=40) * 5 + 3
    assert _close(bn(x), _ref3(ref, x)), "3 維 eval 的輸出和 torch 不同（eval 要用 running stats）"
    assert torch.equal(bn.running_mean, mean0) and torch.equal(bn.running_var, var0), "eval 模式不能更新 running stats"
    out1 = bn(x[:1])
    assert tuple(out1.shape) == (1, 4, 6), "eval 模式要能處理 n = 1"
    assert _close(out1, _ref3(ref, x[:1]))


def test_batchnorm_3d_backward():
    gamma, beta = _rand(6, seed=1), _rand(6, seed=2)
    bn = BatchNorm1d(6)
    _set_params(bn, gamma, beta)
    bn.gamma.requires_grad_(True)
    bn.beta.requires_grad_(True)
    ref = _torch_bn(6, gamma, beta)
    x = _rand(8, 4, 6, seed=3) * 3 + 2
    xa, xb = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    w = _rand(8, 4, 6, seed=4)
    (bn(xa) * w).sum().backward()
    (_ref3(ref, xb) * w).sum().backward()
    assert _close(xa.grad, xb.grad), "對 x 的梯度和 torch 不同"
    assert _close(bn.gamma.grad, ref.weight.grad), "對 gamma 的梯度和 torch 不同"
    assert _close(bn.beta.grad, ref.bias.grad), "對 beta 的梯度和 torch 不同"


def test_batchnorm_2d_unchanged():
    gamma, beta = _rand(6, seed=1), _rand(6, seed=2)
    bn = BatchNorm1d(6)
    _set_params(bn, gamma, beta)
    ref = _torch_bn(6, gamma, beta)
    for s in range(3):
        x = _rand(32, 6, seed=50 + s) * 3 + 2
        assert _close(bn(x), ref(x)), "2 維訓練模式的輸出和 torch 不同"
    assert tuple(bn.running_mean.shape) == (6,) and tuple(bn.running_var.shape) == (6,)
    assert _close(bn.running_mean, ref.running_mean) and _close(bn.running_var, ref.running_var)
    bn.training = False
    ref.eval()
    x = _rand(10, 6, seed=60)
    assert _close(bn(x), ref(x)), "2 維 eval 的輸出和 torch 不同"


# ---------- 模型：訓練前 ----------


def _first_flatten_out(model, Xd):
    with torch.no_grad():
        model(Xd)
    return _of_type(model, FlattenConsecutive)[0].out


@pytest.mark.parametrize("seed", [0, 1])
def test_init_structure(split, seed):
    model, block_size = init_wavenet(seed)
    assert type(block_size) is int and block_size >= 1, f"block_size 要是正整數，拿到 {block_size!r}"
    layers = model.layers
    assert isinstance(layers, list) and layers, "model.layers 要是非空的 list"
    bad = [type(l).__name__ for l in layers if not isinstance(l, LAYER_TYPES)]
    assert not bad, f"model.layers 只能放 Embedding／FlattenConsecutive／Linear／BatchNorm1d／Tanh，拿到 {bad}"
    assert len(_of_type(model, FlattenConsecutive)) >= 2, "至少要 2 個 FlattenConsecutive（階層式）"
    tanh_idx = [i for i, l in enumerate(layers) if isinstance(l, Tanh)]
    assert len(tanh_idx) >= 2, f"至少要 2 個 Tanh 層，只有 {len(tanh_idx)} 個"
    assert all(i > 0 and isinstance(layers[i - 1], BatchNorm1d) for i in tanh_idx), "每個 Tanh 前面要緊接著一個 BatchNorm1d"
    _, dev = split
    Xd, _ = build_dataset(dev[:20], block_size)
    with torch.no_grad():
        out = model(Xd)
    assert tuple(out.shape) == (Xd.shape[0], 27), f"model(X) 的形狀是 {tuple(out.shape)}，README 約定是 (n, 27)"
    assert out.device.type == "cpu"


@pytest.mark.parametrize("seed", [0, 1])
def test_init_hierarchical(split, seed):
    model, block_size = init_wavenet(seed)
    _, dev = split
    Xd, _ = build_dataset(dev[:20], block_size)
    out = _first_flatten_out(model, Xd)
    assert out is not None and out.ndim == 3 and out.shape[1] < block_size, (
        f"第一個 FlattenConsecutive 的輸出要是 3 維 (n, L, C)、而且 L < block_size = {block_size}，拿到形狀 "
        f"{None if out is None else tuple(out.shape)}：一次把整個 context 攤平就是第 2 段的 MLP，不是階層式"
    )


def test_init_same_seed_same_model(split):
    _, dev = split
    torch.manual_seed(1)
    a, bs = init_wavenet(0)
    torch.manual_seed(2)
    b, _ = init_wavenet(0)
    c, _ = init_wavenet(1)
    Xd, _ = build_dataset(dev[:200], bs)
    with torch.no_grad():
        oa, ob, oc = a(Xd), b(Xd), c(Xd)
    assert torch.equal(oa, ob), "同 seed 要得到相同的模型，而且不受全域亂數狀態影響"
    assert not torch.equal(oa, oc), "不同 seed 應該得到不同的模型"


@pytest.mark.parametrize("seed", [0, 1])
def test_init_loss_near_log27(split, seed):
    _, dev = split
    model, bs = init_wavenet(seed)
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
    model, bs = init_wavenet(seed)
    _set_training(model, True)
    Xd, _ = build_dataset(dev, bs)
    with torch.no_grad():
        model(Xd)
    fracs = [(l.out.abs() > SATURATION).float().mean().item() for l in _of_type(model, Tanh)]
    print("\n各 Tanh 層飽和比例", [f"{f:.3f}" for f in fracs])
    assert all(f < MAX_SATURATED for f in fracs), (
        f"各 Tanh 層 |out| > {SATURATION} 的比例 {[round(f, 3) for f in fracs]}，每一層都要 < {MAX_SATURATED}"
    )


# ---------- 模型：訓練（預設跳過） ----------


@pytest.mark.train
def test_train_wavenet_beats_threshold(split):
    from wavenet import train_wavenet

    train, dev = split
    model, block_size = train_wavenet(train, 0)
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
    print(f"\nWaveNet dev NLL {dev_nll:.4f}；bigram dev NLL {base:.4f}；門檻 {base - MARGIN:.4f}")
    assert math.isfinite(dev_nll), f"dev NLL 是 {dev_nll}，訓練發散了"
    assert dev_nll <= base - MARGIN, (
        f"dev NLL {dev_nll:.4f}，要 ≤ {base - MARGIN:.4f}（bigram 在 dev 上的 {base:.4f} − {MARGIN}）"
    )

    names = sample(model, block_size, 5, torch.Generator().manual_seed(0))
    assert len(names) == 5 and all(set(s) <= set(ALPHABET) for s in names)
