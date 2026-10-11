"""attention.py 的測試（第 2 段）：只檢查外部行為，不限制內部寫法。介面約定見 README.md。

Head 以 F.scaled_dot_product_attention(q, k, v, is_causal=True) 當正確答案比對（q、k、v 用 Head 自己的
query／key／value 算），它的預設縮放就是 1/√head_size。測試的 n_embd（32）和 head_size（8）不同，所以用
1/√n_embd 縮放會被抓到。

測試內容：

| 測試                              | 檢查什麼                                                                      |
|-----------------------------------|-------------------------------------------------------------------------------|
| test_causal_mean[...]             | (4, 8, 2)、(3, 5, 4)、(2, 1, 3)：out[b, t] = x[b, :t+1] 的平均；形狀不變；不改動輸入 |
| test_head_attributes              | key／query／value 是 nn.Linear(32, 8, bias=False)；要訓練的參數剛好是這三個 weight |
| test_head_matches_sdpa[T=16／5／1] | 輸出 (3, T, 8)，和 SDPA(is_causal=True) 相同；T < block_size 也能算            |
| test_head_causal                  | 改動 t0 之後的輸入，t0 之前的輸出不變、之後的輸出會變                          |
| test_head_backward                | 對 x 與三個 weight 的梯度和 SDPA 路徑相同                                      |

測資：x 都是固定 seed 的 torch.randn；Head 建立前先 torch.manual_seed(0)。

負控（Claude 用 scratchpad 的參考實作故意改錯，各自應失敗的測試都失敗）：
- causal_mean 每一列都除以 T／沒有 tril（含未來）：test_causal_mean 的 4x8x2、3x5x4（T = 1 時兩者剛好相同）
- 縮放用 1/√n_embd、不縮放：test_head_matches_sdpa 的 T=16、T=5，test_head_backward（T = 1 時 softmax 只有一項，縮放沒影響）
- 沒有 causal mask、softmax 對 dim=1、先 softmax 再把未來位置設 0（沒有重新正規化）：test_head_matches_sdpa
  的 T=16、T=5，test_head_causal，test_head_backward（test_single_head_lm.py 的 test_causal 也會失敗）
- mask 沒有切成 [:T, :T]：T < block_size 的測試報形狀錯誤（T=16 剛好等於 block_size 所以通過）
- key 有 bias：test_head_attributes
- mask 註冊成 nn.Parameter：test_head_attributes（test_single_head_lm.py 的 test_backward 也會失敗）

不同但正確的寫法也要能過（已確認通過）：causal_mean 用 for 迴圈；mask 不存 buffer、每次 forward 現做。
"""

import pytest
import torch
import torch.nn.functional as F

from attention import Head, causal_mean

N_EMBD, HEAD_SIZE, BLOCK = 32, 8, 16
ATOL = 1e-5


def _randn(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _head():
    torch.manual_seed(0)
    return Head(N_EMBD, HEAD_SIZE, BLOCK)


def _sdpa(head, x):
    return F.scaled_dot_product_attention(head.query(x), head.key(x), head.value(x), is_causal=True)


@pytest.mark.parametrize("shape", [(4, 8, 2), (3, 5, 4), (2, 1, 3)], ids=["4x8x2", "3x5x4", "2x1x3"])
def test_causal_mean(shape):
    x = _randn(*shape, seed=1)
    before = x.clone()
    out = causal_mean(x)
    expected = x.cumsum(1) / torch.arange(1, shape[1] + 1).view(1, -1, 1)
    assert out.shape == x.shape
    assert torch.allclose(out, expected, atol=ATOL), "out[b, t] 應該是 x[b, 0..t]（含 t）的平均"
    assert torch.equal(x, before), "causal_mean 不能改動輸入"


def test_head_attributes():
    head = _head()
    for name in ("key", "query", "value"):
        lin = getattr(head, name)
        assert isinstance(lin, torch.nn.Linear), f"head.{name} 應該是 nn.Linear"
        assert (lin.in_features, lin.out_features) == (N_EMBD, HEAD_SIZE)
        assert lin.bias is None, f"head.{name} 不應該有 bias"
    params = {id(p) for p in head.parameters()}
    expected = {id(head.key.weight), id(head.query.weight), id(head.value.weight)}
    assert params == expected, "要訓練的參數應該剛好是 key、query、value 的 weight（mask 不是參數）"


@pytest.mark.parametrize("T", [BLOCK, 5, 1], ids=["T=16", "T=5", "T=1"])
def test_head_matches_sdpa(T):
    head = _head()
    x = _randn(3, T, N_EMBD, seed=2)
    before = x.clone()
    with torch.no_grad():
        out = head(x)
        expected = _sdpa(head, x)
    assert out.shape == (3, T, HEAD_SIZE)
    diff = (out - expected).abs().max().item()
    assert torch.allclose(out, expected, atol=ATOL), f"和 causal attention 的正確答案最大差 {diff:.3g}"
    assert torch.equal(x, before), "Head 不能改動輸入"


def test_head_causal():
    head = _head()
    x = _randn(2, BLOCK, N_EMBD, seed=3)
    changed = x.clone()
    t0 = 6
    changed[:, t0:] = _randn(2, BLOCK - t0, N_EMBD, seed=4)
    with torch.no_grad():
        a, b = head(x), head(changed)
    assert torch.allclose(a[:, :t0], b[:, :t0], atol=ATOL), "位置 t 的輸出不能看到 t 之後的輸入"
    assert not torch.allclose(a[:, t0:], b[:, t0:], atol=ATOL)


def test_head_backward():
    head = _head()
    weights = [head.query.weight, head.key.weight, head.value.weight]
    x1 = _randn(2, 10, N_EMBD, seed=5).requires_grad_()
    x2 = x1.detach().clone().requires_grad_()
    r = _randn(2, 10, HEAD_SIZE, seed=6)
    g_out = torch.autograd.grad((head(x1) * r).sum(), [x1] + weights)
    g_ref = torch.autograd.grad((_sdpa(head, x2) * r).sum(), [x2] + weights)
    for name, a, b in zip(["x", "query.weight", "key.weight", "value.weight"], g_out, g_ref):
        assert torch.allclose(a, b, atol=1e-4), f"對 {name} 的梯度不同，最大差 {(a - b).abs().max().item():.3g}"
