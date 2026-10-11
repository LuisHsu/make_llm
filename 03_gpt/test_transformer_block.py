"""attention.py 第 3 段（LayerNorm、Head 的 dropout、MultiHeadAttention、FeedForward、Block）的測試：只檢查外部行為。
介面約定見 README.md。

正確答案：LayerNorm 以 F.layer_norm 比對；MultiHeadAttention 用各 head 自己的 query／key／value 加 SDPA(is_causal=True)
組出來；FeedForward 用 fc1、fc2 組出來；Block 用 F.layer_norm（ln1、ln2 的參數）加上已經測過的 sa、ffwd 組出來。
dropout 只測外部行為：eval 模式等於沒有 dropout 的算法；train 模式 p > 0 時，兩次呼叫的輸出不同、也和 eval 不同；
p = 0 時 train 和 eval 相同。

測試內容（n_embd 32、n_head 4、block_size 16；模組建立前先 torch.manual_seed(0)）：

| 測試                                   | 檢查什麼                                                                      |
|----------------------------------------|-------------------------------------------------------------------------------|
| test_layernorm_defaults                | weight 全 1、bias 全 0、eps 1e-5；要訓練的參數剛好是 weight、bias               |
| test_layernorm_matches_torch[2d／3d]   | 隨機 weight／bias，輸出和 F.layer_norm 相同；不改動輸入                         |
| test_layernorm_custom_eps              | eps = 0.01、輸入 variance 和 eps 同數量級時仍相同                              |
| test_layernorm_backward                | 對 x、weight、bias 的梯度和 F.layer_norm 相同                                  |
| test_layernorm_handwritten             | forward 裡不能呼叫 F.layer_norm／torch.layer_norm（測試把它們換成會報錯的函式）  |
| test_head_dropout                      | Head(…, dropout) 的 dropout 行為；eval 時等於 SDPA                             |
| test_mha_attributes                    | heads 是 4 個 Head（32 → 8）的 ModuleList；proj 是 Linear(32, 32)（有 bias）；參數剛好是這些 |
| test_mha_matches_reference[T=16／5／1] | eval 輸出 = proj(各 head 的 causal attention 依序接起來)                       |
| test_mha_causal                        | 改動 t0 之後的輸入，t0 之前的輸出不變                                          |
| test_mha_dropout                       | dropout 行為                                                                   |
| test_ffwd_attributes                   | fc1 是 Linear(32, 128)、fc2 是 Linear(128, 32)（都有 bias）；參數剛好是這些      |
| test_ffwd_matches_reference            | eval 輸出 = fc2(relu(fc1(x)))，3 維輸入                                         |
| test_ffwd_dropout                      | dropout 行為                                                                   |
| test_block_attributes                  | ln1、ln2 是 attention.LayerNorm(32)；sa、ffwd 的型別；參數剛好是四個子模組的     |
| test_block_matches_reference[T=16／5]  | eval 輸出 = pre-norm + residual（ln1、ln2 用不同的隨機參數，順序錯會被抓到）    |
| test_block_causal                      | 改動 t0 之後的輸入，t0 之前的輸出不變                                          |
| test_block_dropout                     | dropout 行為                                                                   |
| test_block_backward                    | 每個參數都有梯度                                                              |

負控（Claude 用 scratchpad 的參考實作故意改錯，各自應失敗的測試都失敗）：
- LayerNorm 用 unbiased variance：layernorm 的 matches／custom_eps／backward，block_matches_reference，block_dropout
- LayerNorm 對第 0 維取統計量：同上（test_gpt.py 的 test_uses_position 也會失敗）
- eps 加在 sqrt 外面、或忽略傳入的 eps：只有 test_layernorm_custom_eps（eps 1e-5 時差異小於容許誤差）
- LayerNorm 呼叫 F.layer_norm：test_layernorm_handwritten
- Head 沒有 dropout：test_head_dropout
- MultiHeadAttention 沒有 proj：mha_matches_reference、mha_dropout、block_backward；proj 沒有 bias：mha_attributes
- 各 head 反序接起來：mha_matches_reference、mha_dropout
- MultiHeadAttention 完全沒有 dropout（head 與 proj 之後都沒有）：test_mha_dropout
  （只拿掉 head 的 dropout、proj 之後還有：全部通過。dropout 的位置照學習者的選擇不檢查）
- FeedForward 沒有 ReLU、改用 GELU：ffwd_matches_reference、ffwd_dropout；內層 1 倍：ffwd_attributes
- Block 改成 post-norm、沒有 residual、ln1／ln2 用反：block_matches_reference、block_dropout
- Block 的 residual 寫成 x += …（in-place）：block_matches_reference、block_dropout、block_backward（autograd 報 inplace 錯誤）

參考實作 61 passed（含第 2 段的測試與訓練測試）。
"""

import pytest
import torch
import torch.nn.functional as F

from attention import Block, FeedForward, Head, LayerNorm, MultiHeadAttention

N_EMBD, N_HEAD, BLOCK = 32, 4, 16
HEAD_SIZE = N_EMBD // N_HEAD
ATOL = 1e-5
P = 0.5


def _randn(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _build(cls, *args, **kwargs):
    torch.manual_seed(0)
    return cls(*args, **kwargs)


def _sdpa(head, x):
    return F.scaled_dot_product_attention(head.query(x), head.key(x), head.value(x), is_causal=True)


def _ids(module):
    return {id(p) for p in module.parameters()}


def _check_dropout(make, reference, x):
    """make(p) 建立模組；reference(module, x) 是沒有 dropout 的算法。"""
    m = make(P)
    m.eval()
    with torch.no_grad():
        ev = m(x)
        assert torch.allclose(ev, reference(m, x), atol=ATOL), "eval 模式下 dropout 不應該起作用"
        assert torch.equal(m(x), ev), "eval 模式下兩次呼叫應該相同"
        m.train()
        a, b = m(x), m(x)
    assert not torch.allclose(a, ev, atol=ATOL), f"train 模式、dropout = {P} 時輸出應該和 eval 不同"
    assert not torch.allclose(a, b, atol=ATOL), f"train 模式、dropout = {P} 時兩次呼叫應該不同"

    m0 = make(0.0)
    m0.train()
    with torch.no_grad():
        assert torch.allclose(m0(x), reference(m0, x), atol=ATOL), "dropout = 0 時 train 模式應該和沒有 dropout 相同"


# ---------- LayerNorm ----------

def test_layernorm_defaults():
    ln = LayerNorm(N_EMBD)
    assert not isinstance(ln, torch.nn.LayerNorm), "LayerNorm 要自己寫，不能繼承 nn.LayerNorm"
    assert torch.equal(ln.weight.detach(), torch.ones(N_EMBD))
    assert torch.equal(ln.bias.detach(), torch.zeros(N_EMBD))
    assert ln.eps == 1e-5
    assert _ids(ln) == {id(ln.weight), id(ln.bias)}


def _random_ln(dim, eps=1e-5, seed=7):
    ln = LayerNorm(dim, eps=eps)
    with torch.no_grad():
        ln.weight.copy_(_randn(dim, seed=seed))
        ln.bias.copy_(_randn(dim, seed=seed + 1))
    return ln


@pytest.mark.parametrize("shape", [(6, N_EMBD), (3, 5, N_EMBD)], ids=["2d", "3d"])
def test_layernorm_matches_torch(shape):
    ln = _random_ln(N_EMBD)
    x = _randn(*shape, seed=1) * 3 + 1
    before = x.clone()
    with torch.no_grad():
        out = ln(x)
        expected = F.layer_norm(x, (N_EMBD,), ln.weight, ln.bias, 1e-5)
    assert out.shape == x.shape
    assert torch.allclose(out, expected, atol=ATOL), f"最大差 {(out - expected).abs().max().item():.3g}"
    assert torch.equal(x, before), "LayerNorm 不能改動輸入"


def test_layernorm_custom_eps():
    ln = _random_ln(8, eps=1e-2)
    assert ln.eps == 1e-2
    x = _randn(4, 8, seed=2) * 0.1
    with torch.no_grad():
        out = ln(x)
        expected = F.layer_norm(x, (8,), ln.weight, ln.bias, 1e-2)
    assert torch.allclose(out, expected, atol=ATOL), f"最大差 {(out - expected).abs().max().item():.3g}"


def test_layernorm_backward():
    ln = _random_ln(N_EMBD)
    x1 = (_randn(3, 5, N_EMBD, seed=3) * 2).requires_grad_()
    x2 = x1.detach().clone().requires_grad_()
    r = _randn(3, 5, N_EMBD, seed=4)
    g = torch.autograd.grad((ln(x1) * r).sum(), [x1, ln.weight, ln.bias])
    g_ref = torch.autograd.grad((F.layer_norm(x2, (N_EMBD,), ln.weight, ln.bias, 1e-5) * r).sum(), [x2, ln.weight, ln.bias])
    for name, a, b in zip(["x", "weight", "bias"], g, g_ref):
        assert torch.allclose(a, b, atol=1e-4), f"對 {name} 的梯度不同，最大差 {(a - b).abs().max().item():.3g}"


def test_layernorm_handwritten(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("LayerNorm 要自己寫，不能呼叫 F.layer_norm／torch.layer_norm／nn.LayerNorm")

    ln = _random_ln(N_EMBD)
    monkeypatch.setattr(torch.nn.functional, "layer_norm", forbidden)
    monkeypatch.setattr(torch, "layer_norm", forbidden)
    ln(_randn(2, N_EMBD, seed=5))


# ---------- Head 的 dropout ----------

def test_head_dropout():
    x = _randn(2, BLOCK, N_EMBD, seed=6)
    _check_dropout(lambda p: _build(Head, N_EMBD, HEAD_SIZE, BLOCK, dropout=p), _sdpa, x)


# ---------- MultiHeadAttention ----------

def _mha_reference(mha, x):
    return mha.proj(torch.cat([_sdpa(h, x) for h in mha.heads], dim=-1))


def test_mha_attributes():
    mha = _build(MultiHeadAttention, N_EMBD, N_HEAD, BLOCK)
    assert isinstance(mha.heads, torch.nn.ModuleList) and len(mha.heads) == N_HEAD
    for h in mha.heads:
        assert isinstance(h, Head)
        assert (h.query.in_features, h.query.out_features) == (N_EMBD, HEAD_SIZE)
    assert isinstance(mha.proj, torch.nn.Linear)
    assert (mha.proj.in_features, mha.proj.out_features) == (N_EMBD, N_EMBD) and mha.proj.bias is not None
    expected = set().union(*(_ids(h) for h in mha.heads)) | _ids(mha.proj)
    assert _ids(mha) == expected, "要訓練的參數應該剛好是各 head 與 proj 的參數"


@pytest.mark.parametrize("T", [BLOCK, 5, 1], ids=["T=16", "T=5", "T=1"])
def test_mha_matches_reference(T):
    mha = _build(MultiHeadAttention, N_EMBD, N_HEAD, BLOCK)
    mha.eval()
    x = _randn(3, T, N_EMBD, seed=8)
    with torch.no_grad():
        out = mha(x)
        expected = _mha_reference(mha, x)
    assert out.shape == (3, T, N_EMBD)
    assert torch.allclose(out, expected, atol=ATOL), f"最大差 {(out - expected).abs().max().item():.3g}"


def test_mha_causal():
    mha = _build(MultiHeadAttention, N_EMBD, N_HEAD, BLOCK)
    mha.eval()
    x = _randn(2, BLOCK, N_EMBD, seed=9)
    changed = x.clone()
    changed[:, 6:] = _randn(2, BLOCK - 6, N_EMBD, seed=10)
    with torch.no_grad():
        assert torch.allclose(mha(x)[:, :6], mha(changed)[:, :6], atol=ATOL), "位置 t 的輸出不能看到 t 之後的輸入"


def test_mha_dropout():
    x = _randn(2, BLOCK, N_EMBD, seed=11)
    _check_dropout(lambda p: _build(MultiHeadAttention, N_EMBD, N_HEAD, BLOCK, dropout=p), _mha_reference, x)


# ---------- FeedForward ----------

def _ffwd_reference(ff, x):
    return ff.fc2(F.relu(ff.fc1(x)))


def test_ffwd_attributes():
    ff = _build(FeedForward, N_EMBD)
    for name, shape in (("fc1", (N_EMBD, 4 * N_EMBD)), ("fc2", (4 * N_EMBD, N_EMBD))):
        lin = getattr(ff, name)
        assert isinstance(lin, torch.nn.Linear), f"ffwd.{name} 應該是 nn.Linear"
        assert (lin.in_features, lin.out_features) == shape and lin.bias is not None
    assert _ids(ff) == _ids(ff.fc1) | _ids(ff.fc2)


def test_ffwd_matches_reference():
    ff = _build(FeedForward, N_EMBD)
    ff.eval()
    x = _randn(3, 7, N_EMBD, seed=12)
    with torch.no_grad():
        out = ff(x)
        expected = _ffwd_reference(ff, x)
    assert out.shape == x.shape
    assert torch.allclose(out, expected, atol=ATOL), f"最大差 {(out - expected).abs().max().item():.3g}"


def test_ffwd_dropout():
    x = _randn(2, 7, N_EMBD, seed=13)
    _check_dropout(lambda p: _build(FeedForward, N_EMBD, dropout=p), _ffwd_reference, x)


# ---------- Block ----------

def _make_block(p=0.0):
    blk = _build(Block, N_EMBD, N_HEAD, BLOCK, dropout=p)
    with torch.no_grad():
        for i, ln in enumerate((blk.ln1, blk.ln2)):
            ln.weight.copy_(_randn(N_EMBD, seed=20 + i) * 0.5 + 1)
            ln.bias.copy_(_randn(N_EMBD, seed=30 + i) * 0.5)
    return blk


def _block_reference(blk, x):
    h = x + blk.sa(F.layer_norm(x, (N_EMBD,), blk.ln1.weight, blk.ln1.bias, blk.ln1.eps))
    return h + blk.ffwd(F.layer_norm(h, (N_EMBD,), blk.ln2.weight, blk.ln2.bias, blk.ln2.eps))


def test_block_attributes():
    blk = _build(Block, N_EMBD, N_HEAD, BLOCK)
    for name in ("ln1", "ln2"):
        ln = getattr(blk, name)
        assert isinstance(ln, LayerNorm) and ln.weight.shape == (N_EMBD,), f"block.{name} 應該是 attention.LayerNorm({N_EMBD})"
    assert isinstance(blk.sa, MultiHeadAttention) and len(blk.sa.heads) == N_HEAD
    assert isinstance(blk.ffwd, FeedForward)
    expected = _ids(blk.ln1) | _ids(blk.ln2) | _ids(blk.sa) | _ids(blk.ffwd)
    assert _ids(blk) == expected, "要訓練的參數應該剛好是 ln1、sa、ln2、ffwd 的參數"


@pytest.mark.parametrize("T", [BLOCK, 5], ids=["T=16", "T=5"])
def test_block_matches_reference(T):
    blk = _make_block()
    blk.eval()
    x = _randn(3, T, N_EMBD, seed=14)
    before = x.clone()
    with torch.no_grad():
        out = blk(x)
        expected = _block_reference(blk, x)
    assert out.shape == x.shape
    assert torch.allclose(out, expected, atol=1e-4), (
        f"最大差 {(out - expected).abs().max().item():.3g}；應該是 x = x + sa(ln1(x))，再 x = x + ffwd(ln2(x))")
    assert torch.equal(x, before), "Block 不能改動輸入"


def test_block_causal():
    blk = _make_block()
    blk.eval()
    x = _randn(2, BLOCK, N_EMBD, seed=15)
    changed = x.clone()
    changed[:, 6:] = _randn(2, BLOCK - 6, N_EMBD, seed=16)
    with torch.no_grad():
        assert torch.allclose(blk(x)[:, :6], blk(changed)[:, :6], atol=ATOL), "位置 t 的輸出不能看到 t 之後的輸入"


def test_block_dropout():
    x = _randn(2, BLOCK, N_EMBD, seed=17)
    _check_dropout(_make_block, _block_reference, x)


def test_block_backward():
    blk = _make_block()
    out = blk(_randn(2, BLOCK, N_EMBD, seed=18))
    (out * _randn(2, BLOCK, N_EMBD, seed=19)).sum().backward()
    missing = [n for n, p in blk.named_parameters() if p.grad is None]
    assert not missing, f"這些參數沒有梯度：{missing}"
