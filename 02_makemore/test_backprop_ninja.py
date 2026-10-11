"""backprop_ninja.py 的測試（第 4 段）：只檢查外部行為，不限制內部寫法。介面約定見 README.md。

所有手推梯度都以 PyTorch autograd 當正確答案比對（影片的 cmp()）。學習者的函式一律在 torch.no_grad() 裡呼叫，
傳入的 tensor 也都不帶 requires_grad，所以用 autograd 算梯度會直接報錯。模型的訓練只有一項測試
（test_train_manual_beats_bigram），標記為 train，預設跳過，加 --run-train 才執行。

練習 1 的 26 個 d_<名稱> 各自獨立測：傳入的 fwd 是測試自己用 README 的 forward 算的值（不是學習者的 forward），
上游梯度（直接用到這個變數的那幾個中間變數的梯度）是 autograd 算出的正確值，以關鍵字參數傳入。所以一個函式寫錯，
只有它自己那一項失敗，不會連帶讓更上游的失敗。

測試內容：

| 測試                                          | 檢查什麼                                                                  |
|-----------------------------------------------|---------------------------------------------------------------------------|
| test_forward_loss[video／small]               | 回傳 (float, Forward)；loss 與 autograd 的 forward 相同；不改動輸入        |
| test_forward_value[video／small-<名稱>]       | 19 個中間變數：形狀與數值和 README 的定義相同                              |
| test_grad[video／small-<名稱>]                | d_<名稱> 存在；梯度的形狀與該變數相同、數值與 autograd 相同；不改動傳入的 tensor |
| test_cross_entropy_backward[...]              | dlogits 與 autograd（F.cross_entropy）相同；形狀；不改動輸入               |
| test_cross_entropy_backward_large_logits      | logits 加上 200 時不出現 inf／nan，結果與 autograd 相同                    |
| test_batchnorm_backward[...]                  | dhprebn 與 autograd（unbiased variance 的 BatchNorm）相同；形狀；不改動輸入 |
| test_loss_and_grads[video／small]             | 回傳 (float, list)；loss 與 7 個參數的梯度與 autograd 相同；不改動輸入      |
| test_train_manual_beats_bigram（train，預設跳過）| 在 no_grad 裡訓練；eval 逐列獨立；dev NLL ≤ bigram 在 dev 上的 NLL − 0.2；可交給 sample |

三組設定：video 是影片的大小（n 32、block_size 3、n_embd 10、n_hidden 64）；small 是 n 7、block_size 4、n_embd 5、
n_hidden 13，抓寫死的維度或 batch 大小；tiny_var 和 video 同大小，但 W1、b1 縮小 300 倍，讓 bnvar 約 1e-5、和 eps 同一個
數量級（video 的 bnvar 約 1，漏掉 eps 的差別小於容許誤差，抓不到）。b1 也要跟著縮小：hprebn 的平均遠大於它的變化
時，減平均會相消，float32 的捨入誤差就超過容許誤差（參考實作在 float64 下和 autograd 差 1e-13，float32 差 1e-3）。參數照影片用非標準的初始化（b1、bnbias 不是 0，bngain 不是 1），避免全 0
或全 1 掩蓋錯誤；Xb 的前兩列全部是 0，保證同一個字元在 batch 裡出現多次（C 的梯度要累加）。

鑑別力有限的項目：logit_maxes 與 b1 的梯度理論上是 0，autograd 算出的只是浮點誤差，容許誤差（atol 1e-6）蓋過它，
這兩項通過不代表推導正確；d_logits 裡經過 logit_maxes 的那一份也一樣小，漏掉它測試看不出來。

門檻來源（Claude 用 scratchpad 的參考實作量，參考實作不進 repo）：參考實作照影片練習 4（n_embd 10、n_hidden 200、
block_size 3、batch 32、lr 0.1，一半步數後 0.01；訓練後用整份 train 校正 BatchNorm 的 mean／variance），
split 用學習者的 split_words(words, 42)：
- 2 萬步 6.0 s，dev 2.2102（bigram 2.4647，門檻 2.2647）；1000 步 dev 2.4382，不通過
- 第 2 段學習者的 MLP（dev 2.2520）也會過這個門檻：這一段的重點是梯度，門檻只確認手推梯度真的訓練得動

負控（參考實作故意改錯，各自應失敗的測試都失敗，而且只有出錯的那個函式失敗）：
- d_bndiff2 用 n 而不是 n − 1：只有 d_bndiff2 ×2 組；batchnorm_backward 用 biased：batchnorm_backward 3 項、loss_and_grads 2 項
- d_bnvar 與 batchnorm_backward 漏掉 eps：d_bnvar 只有 tiny_var 失敗（video、small 的 bnvar 約 1，抓不到）；
  batchnorm_backward 的 n=5、n=2 與 scale 1.5e-3 三組、loss_and_grads[tiny_var]
- forward 的 bnvar 用 n：test_forward_loss 與 bnvar 之後的 13 個 test_forward_value ×2 組
- d_C 用指定（=）而不是累加：只有 d_C ×2 組
- d_counts 漏掉經過 counts_sum 的那一支：只有 d_counts ×2 組
- 把 batch 大小寫死成 32：small 那一組用到 n 的 d_logprobs、d_bndiff2、d_hprebn
- forward 改動傳入的 C：test_forward_loss ×2（另外 emb 到 bnmeani 的值也跟著不同）
- d_counts 改動傳入的 dprobs：只有 d_counts ×2 組
- d_probs 裡呼叫 .backward()：只有 d_probs ×2 組（RuntimeError）；訓練時用 loss.backward()：訓練測試失敗
- cross-entropy 先 exp 不減最大值：只有 large_logits；漏掉 ÷n：n > 1 的 cross-entropy 與 loss_and_grads
- 訓練：評估時 BatchNorm 用這一批的統計量 → 逐列獨立的檢查失敗；只訓練 1000 步 → dev NLL 門檻失敗
"""

import math

import pytest
import torch
import torch.nn.functional as F

import backprop_ninja
from backprop_ninja import batchnorm_backward, cross_entropy_backward, forward, loss_and_grads

ATOL, RTOL = 1e-6, 1e-4
EPS = 1e-5
TRAIN_SPLIT_SEED = 42
MARGIN = 0.2

PARAM_NAMES = ["C", "W1", "b1", "W2", "b2", "bngain", "bnbias"]
# README forward 表的順序
FWD_NAMES = [
    "emb", "embcat", "hprebn", "bnmeani", "bndiff", "bndiff2", "bnvar", "bnvar_inv", "bnraw", "hpreact", "h",
    "logits", "logit_maxes", "norm_logits", "counts", "counts_sum", "counts_sum_inv", "probs", "logprobs",
]
# 直接用到每個變數的中間變數（依 forward 表的順序）：d_<名稱> 收到的上游梯度。README 的 d_<名稱> 表
CONSUMERS = {
    "logprobs": [], "probs": ["logprobs"], "counts_sum_inv": ["probs"], "counts_sum": ["counts_sum_inv"],
    "counts": ["counts_sum", "probs"], "norm_logits": ["counts"], "logit_maxes": ["norm_logits"],
    "logits": ["logit_maxes", "norm_logits"], "h": ["logits"], "W2": ["logits"], "b2": ["logits"],
    "hpreact": ["h"], "bngain": ["hpreact"], "bnbias": ["hpreact"], "bnraw": ["hpreact"], "bnvar_inv": ["bnraw"],
    "bnvar": ["bnvar_inv"], "bndiff2": ["bnvar"], "bndiff": ["bndiff2", "bnraw"], "bnmeani": ["bndiff"],
    "hprebn": ["bnmeani", "bndiff"], "embcat": ["hprebn"], "W1": ["hprebn"], "b1": ["hprebn"], "emb": ["embcat"],
    "C": ["emb"],
}
# 影片 cmp() 的順序：從 loss 往回
GRAD_NAMES = [
    "logprobs", "probs", "counts_sum_inv", "counts_sum", "counts", "norm_logits", "logit_maxes", "logits",
    "h", "W2", "b2", "hpreact", "bngain", "bnbias", "bnraw", "bnvar_inv", "bnvar", "bndiff2", "bndiff",
    "bnmeani", "hprebn", "embcat", "W1", "b1", "emb", "C",
]
CONFIGS = {
    "video": dict(n=32, block_size=3, n_embd=10, n_hidden=64, seed=0),
    "small": dict(n=7, block_size=4, n_embd=5, n_hidden=13, seed=1),
    "tiny_var": dict(n=32, block_size=3, n_embd=10, n_hidden=64, seed=2, pre_scale=1 / 300),
}


def _close(a, b, atol=ATOL):
    return torch.allclose(a, b, atol=atol, rtol=RTOL)


def _atol(name, ref_grads):
    """b1 的正確梯度是 0，算出來的只有加總 dhprebn 每一欄的捨入誤差，大小和每欄絕對值的和成正比，atol 跟著放大。"""
    if name == "b1":
        return ATOL * max(1.0, ref_grads["hprebn"].abs().sum(0).max().item())
    return ATOL


def _maxdiff(a, b):
    return (a - b).abs().max().item()


def _make_case(n, block_size, n_embd, n_hidden, seed, pre_scale=1.0):
    g = torch.Generator().manual_seed(seed)
    fan_in = block_size * n_embd
    params = [
        torch.randn(27, n_embd, generator=g),
        torch.randn(fan_in, n_hidden, generator=g) * (5 / 3) / fan_in**0.5 * pre_scale,
        torch.randn(n_hidden, generator=g) * 0.1 * pre_scale,
        torch.randn(n_hidden, 27, generator=g) * 0.1,
        torch.randn(27, generator=g) * 0.1,
        torch.randn(1, n_hidden, generator=g) * 0.1 + 1.0,
        torch.randn(1, n_hidden, generator=g) * 0.1,
    ]
    Xb = torch.randint(0, 27, (n, block_size), generator=g)
    Xb[:2] = 0
    Yb = torch.randint(0, 27, (n,), generator=g)
    return params, Xb, Yb


def _reference(params, Xb, Yb):
    """README 定義的 forward，用 autograd 算每個中間變數與參數的梯度。回傳 (loss, 中間變數的值, 梯度)。"""
    p = {k: t.clone().requires_grad_(True) for k, t in zip(PARAM_NAMES, params)}
    n = Xb.shape[0]
    v = {}
    v["emb"] = p["C"][Xb]
    v["embcat"] = v["emb"].view(n, -1)
    v["hprebn"] = v["embcat"] @ p["W1"] + p["b1"]
    v["bnmeani"] = 1 / n * v["hprebn"].sum(0, keepdim=True)
    v["bndiff"] = v["hprebn"] - v["bnmeani"]
    v["bndiff2"] = v["bndiff"] ** 2
    v["bnvar"] = 1 / (n - 1) * v["bndiff2"].sum(0, keepdim=True)
    v["bnvar_inv"] = (v["bnvar"] + EPS) ** -0.5
    v["bnraw"] = v["bndiff"] * v["bnvar_inv"]
    v["hpreact"] = p["bngain"] * v["bnraw"] + p["bnbias"]
    v["h"] = torch.tanh(v["hpreact"])
    v["logits"] = v["h"] @ p["W2"] + p["b2"]
    v["logit_maxes"] = v["logits"].max(1, keepdim=True).values
    v["norm_logits"] = v["logits"] - v["logit_maxes"]
    v["counts"] = v["norm_logits"].exp()
    v["counts_sum"] = v["counts"].sum(1, keepdim=True)
    v["counts_sum_inv"] = v["counts_sum"] ** -1
    v["probs"] = v["counts"] * v["counts_sum_inv"]
    v["logprobs"] = v["probs"].log()
    loss = -v["logprobs"][torch.arange(n), Yb].mean()
    for t in v.values():
        t.retain_grad()
    loss.backward()
    values = {k: t.detach().clone() for k, t in v.items()}
    grads = {k: t.grad for k, t in v.items()} | {k: t.grad for k, t in p.items()}
    return loss.item(), values, grads


def _clones(params, Xb, Yb):
    return [t.clone() for t in params], Xb.clone(), Yb.clone()


def _unchanged(params, Xb, Yb, originals):
    p0, X0, Y0 = originals
    return all(torch.equal(a, b) for a, b in zip(params, p0)) and torch.equal(Xb, X0) and torch.equal(Yb, Y0)


FWD_FIELDS = FWD_NAMES + PARAM_NAMES + ["Xb", "Yb"]


def _make_forward(case):
    return backprop_ninja.Forward(**{k: case["fwd"][k].clone() for k in FWD_FIELDS})


def _same_forward(fwd, ref):
    return all(torch.equal(getattr(fwd, k), ref[k]) for k in FWD_FIELDS)


# ---------- 練習 1：forward 與逐個中間變數的梯度 ----------


@pytest.fixture(scope="module", params=list(CONFIGS))
def case(request):
    params, Xb, Yb = _make_case(**CONFIGS[request.param])
    loss, values, grads = _reference(params, Xb, Yb)
    fwd = values | dict(zip(PARAM_NAMES, params)) | {"Xb": Xb, "Yb": Yb}
    return dict(params=params, Xb=Xb, Yb=Yb, loss=loss, values=values, grads=grads, fwd=fwd)


@pytest.fixture(scope="module")
def forward_result(case):
    params, Xb, Yb = _clones(case["params"], case["Xb"], case["Yb"])
    originals = _clones(params, Xb, Yb)
    with torch.no_grad():
        result = forward(params, Xb, Yb)
    return dict(result=result, mutated=not _unchanged(params, Xb, Yb, originals))


def test_forward_loss(case, forward_result):
    result = forward_result["result"]
    assert isinstance(result, tuple) and len(result) == 2, "forward 要回傳 (loss, fwd)"
    loss, fwd = result
    assert type(loss) is float, f"loss 要是 Python float，拿到 {type(loss).__name__}"
    assert isinstance(fwd, backprop_ninja.Forward), f"fwd 要是 Forward，拿到 {type(fwd).__name__}"
    assert abs(loss - case["loss"]) < 1e-5, f"loss {loss:.6f}，README 定義的 forward 是 {case['loss']:.6f}"
    assert not forward_result["mutated"], "forward 不能改動傳入的 params、Xb、Yb"


@pytest.mark.parametrize("name", FWD_NAMES)
def test_forward_value(case, forward_result, name):
    _, fwd = forward_result["result"]
    got, ref = getattr(fwd, name), case["values"][name]
    assert tuple(got.shape) == tuple(ref.shape), f"fwd.{name} 的形狀是 {tuple(got.shape)}，README 定義是 {tuple(ref.shape)}"
    assert _close(got, ref), f"fwd.{name} 和 README 的定義不同，最大差 {_maxdiff(got, ref):.3e}"


@pytest.mark.parametrize("name", GRAD_NAMES)
def test_grad(case, name):
    fn = getattr(backprop_ninja, f"d_{name}", None)
    assert callable(fn), f"backprop_ninja 裡沒有 d_{name}"
    fwd = _make_forward(case)
    upstream = {f"d{k}": case["grads"][k].clone() for k in CONSUMERS[name]}
    with torch.no_grad():
        got = fn(fwd, **upstream)
    ref = case["grads"][name]
    assert isinstance(got, torch.Tensor), f"d_{name} 要回傳 tensor，拿到 {type(got).__name__}"
    assert tuple(got.shape) == tuple(ref.shape), f"d_{name} 的形狀是 {tuple(got.shape)}，要和 {name} 相同：{tuple(ref.shape)}"
    assert _close(got, ref, _atol(name, case["grads"])), f"d_{name} 和 autograd 不同，最大差 {_maxdiff(got, ref):.3e}"
    assert _same_forward(fwd, case["fwd"]), f"d_{name} 不能改動 fwd"
    assert all(torch.equal(t, case["grads"][k[1:]]) for k, t in upstream.items()), f"d_{name} 不能改動傳入的上游梯度"


# ---------- 練習 2：cross-entropy 一式 ----------


@pytest.mark.parametrize("n,scale", [(32, 1.0), (5, 3.0), (1, 1.0)])
def test_cross_entropy_backward(n, scale):
    g = torch.Generator().manual_seed(n)
    logits = torch.randn(n, 27, generator=g) * scale
    Yb = torch.randint(0, 27, (n,), generator=g)
    x = logits.clone().requires_grad_(True)
    F.cross_entropy(x, Yb).backward()
    logits0, Y0 = logits.clone(), Yb.clone()
    with torch.no_grad():
        got = cross_entropy_backward(logits, Yb)
    assert tuple(got.shape) == (n, 27), f"dlogits 的形狀是 {tuple(got.shape)}，要是 {(n, 27)}"
    assert _close(got, x.grad), f"dlogits 和 autograd 不同，最大差 {_maxdiff(got, x.grad):.3e}"
    assert torch.equal(logits, logits0) and torch.equal(Yb, Y0), "不能改動傳入的 logits、Yb"


def test_cross_entropy_backward_large_logits():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(8, 27, generator=g) * 3 + 200
    Yb = torch.randint(0, 27, (8,), generator=g)
    x = logits.clone().requires_grad_(True)
    F.cross_entropy(x, Yb).backward()
    with torch.no_grad():
        got = cross_entropy_backward(logits, Yb)
    assert torch.isfinite(got).all(), "logits 很大時 dlogits 出現 inf／nan"
    assert _close(got, x.grad), f"dlogits 和 autograd 不同，最大差 {_maxdiff(got, x.grad):.3e}"


# ---------- 練習 3：BatchNorm 一式 ----------


# scale 1.5e-3：variance 約 1e-5，和 eps 同一個數量級。平均也一起縮小，避免減平均時相消（見檔案開頭）
@pytest.mark.parametrize("n,dim,scale", [(32, 64, 1.0), (5, 7, 1.0), (2, 3, 1.0), (32, 64, 1.5e-3)])
def test_batchnorm_backward(n, dim, scale):
    g = torch.Generator().manual_seed(100 + n)
    hprebn = (torch.randn(n, dim, generator=g) * 2 + 0.5) * scale
    bngain = torch.randn(1, dim, generator=g) * 0.1 + 1.0
    bnbias = torch.randn(1, dim, generator=g) * 0.1
    dhpreact = torch.randn(n, dim, generator=g)
    x = hprebn.clone().requires_grad_(True)
    mean = x.mean(0, keepdim=True)
    var = ((x - mean) ** 2).sum(0, keepdim=True) / (n - 1)
    hpreact = bngain * (x - mean) * (var + EPS) ** -0.5 + bnbias
    (hpreact * dhpreact).sum().backward()
    inputs0 = [t.clone() for t in (dhpreact, hprebn, bngain)]
    with torch.no_grad():
        got = batchnorm_backward(dhpreact, hprebn, bngain)
    assert tuple(got.shape) == (n, dim), f"dhprebn 的形狀是 {tuple(got.shape)}，要是 {(n, dim)}"
    assert _close(got, x.grad), (
        f"dhprebn 和 autograd 不同，最大差 {_maxdiff(got, x.grad):.3e}（variance 是 unbiased，除以 n − 1）"
    )
    assert all(torch.equal(a, b) for a, b in zip((dhpreact, hprebn, bngain), inputs0)), "不能改動傳入的 tensor"


# ---------- 練習 4：完整梯度與訓練 ----------


@pytest.mark.parametrize("config", list(CONFIGS))
def test_loss_and_grads(config):
    params, Xb, Yb = _make_case(**CONFIGS[config])
    originals = _clones(params, Xb, Yb)
    ref_loss, _, ref_grads = _reference(params, Xb, Yb)
    with torch.no_grad():
        result = loss_and_grads(params, Xb, Yb)
    assert isinstance(result, tuple) and len(result) == 2, "loss_and_grads 要回傳 (loss, grads)"
    loss, grads = result
    assert type(loss) is float, f"loss 要是 Python float，拿到 {type(loss).__name__}"
    assert abs(loss - ref_loss) < 1e-5, f"loss {loss:.6f}，README 定義的 forward 是 {ref_loss:.6f}"
    assert isinstance(grads, list) and len(grads) == 7, "grads 要是 7 個 tensor 的 list，順序同 params"
    for name, got in zip(PARAM_NAMES, grads):
        ref = ref_grads[name]
        assert tuple(got.shape) == tuple(ref.shape), f"{name} 的梯度形狀是 {tuple(got.shape)}，要是 {tuple(ref.shape)}"
        assert _close(got, ref, _atol(name, ref_grads)), f"{name} 的梯度和 autograd 不同，最大差 {_maxdiff(got, ref):.3e}"
    assert _unchanged(params, Xb, Yb, originals), "loss_and_grads 不能改動傳入的 params、Xb、Yb"


@pytest.mark.train
def test_train_manual_beats_bigram(words):
    from backprop_ninja import train_manual
    from mlp import build_dataset, evaluate, sample, split_words
    from test_mlp import ALPHABET, _bigram_dev_nll

    train, dev, _ = split_words(words, TRAIN_SPLIT_SEED)
    try:
        with torch.no_grad():
            model, block_size = train_manual(train, 0)
    except RuntimeError as e:
        pytest.fail(f"train_manual 在 torch.no_grad() 裡失敗（訓練時用了 autograd？）：{e}")
    assert type(block_size) is int and block_size >= 1, f"block_size 要是正整數，拿到 {block_size!r}"

    Xd, Yd = build_dataset(dev, block_size)
    with torch.no_grad():
        out5 = model(Xd[:5])
        out1 = model(Xd[:1])
    assert tuple(out5.shape) == (5, 27), f"model(X) 的形狀是 {tuple(out5.shape)}，README 約定是 (n, 27)"
    assert out5.device.type == "cpu", f"model(X) 回傳的 tensor 在 {out5.device}，README 約定要在 CPU 上"
    assert torch.allclose(out1, out5[:1], atol=1e-5, rtol=1e-4), (
        "同一列的輸出不能因為同一批裡有其他列而改變（評估時 BatchNorm 要用固定的統計量，不是這一批的）"
    )

    dev_nll = evaluate(model, Xd, Yd)
    base = _bigram_dev_nll(train, dev)
    print(f"\n手推梯度訓練 dev NLL {dev_nll:.4f}；bigram dev NLL {base:.4f}；門檻 {base - MARGIN:.4f}")
    assert math.isfinite(dev_nll), f"dev NLL 是 {dev_nll}，訓練發散了"
    assert dev_nll <= base - MARGIN, (
        f"dev NLL {dev_nll:.4f}，要 ≤ {base - MARGIN:.4f}（bigram 在 dev 上的 {base:.4f} − {MARGIN}）"
    )

    names = sample(model, block_size, 5, torch.Generator().manual_seed(0))
    assert len(names) == 5 and all(set(s) <= set(ALPHABET) for s in names)
