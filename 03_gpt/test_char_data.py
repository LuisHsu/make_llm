"""char_data.py 的測試（第 1 段）：只檢查外部行為，不限制內部寫法。介面約定見 README.md。

測試內容：

| 測試                                  | 檢查什麼                                                                    |
|---------------------------------------|-----------------------------------------------------------------------------|
| test_build_vocab_small                | "hello world" 的 stoi／itos：依字元排序編號、型別、互為反函數               |
| test_build_vocab_shakespeare          | 整份資料：65 種字元；'\n' → 0、' ' → 1、'A' → 13、'a' → 39、'z' → 64        |
| test_encode_decode_small              | encode 回傳 list[int]、值正確；decode 回傳 str；decode 也吃 tensor.tolist() |
| test_encode_decode_roundtrip          | 整份資料 decode(encode(text)) == text，長度 1115394                         |
| test_split_data[...]                  | 長度 25、15（frac 0.9）與 10（frac 0.5）：前 int(frac·n) 個是 train，其餘 val，不打亂；不改動輸入 |
| test_get_batch_values                 | 形狀 (batch, block)、int64；每列是 data 的一段連續切片；y 是 x 往後移一格   |
| test_get_batch_offsets                | 起點只會落在 0..len−block−1，且這個範圍的每個起點都抽得到；len = block+1 也能用 |
| test_get_batch_generator              | 同 seed 結果相同、不受全域亂數影響；不同 seed 不同；不改動輸入              |

測資：get_batch 用 data = 5 + 3·arange(n)，所以從值就能算回起點，也能分辨「回傳 data 的值」和「回傳 index」。
test_get_batch_offsets 用長度 50、block 8、抽 5000 列：42 個合法起點各自漏抽的機率約 (41/42)^5000 ≈ 1e-52。

負控（Claude 用 scratchpad 的參考實作故意改錯，各自應失敗的測試都失敗）：
- build_vocab 依字元第一次出現的順序編號：test_build_vocab_small、test_build_vocab_shakespeare、test_encode_decode_small
- split_data 用 round 取代 int：test_split_data[15-0.9]（13.5 → 14）
- get_batch 起點用 randint(len − block − 1)：test_get_batch_offsets（最後一個起點抽不到；len = block+1 時報錯）
- get_batch 用全域亂數：test_get_batch_generator（test_bigram_lm.py 的 test_estimate_loss 也會失敗）
- get_batch 的 y 沒有往後移：test_get_batch_values、test_get_batch_offsets
- get_batch 回傳 index 而不是 data 的值：test_get_batch_values、test_get_batch_offsets
"""

import pytest
import torch

from char_data import build_vocab, decode, encode, get_batch, split_data


def _affine(n):
    return 5 + 3 * torch.arange(n, dtype=torch.long)


def test_build_vocab_small():
    stoi, itos = build_vocab("hello world")
    chars = [" ", "d", "e", "h", "l", "o", "r", "w"]
    assert stoi == {c: i for i, c in enumerate(chars)}
    assert itos == {i: c for i, c in enumerate(chars)}
    assert all(type(k) is str and type(v) is int for k, v in stoi.items())
    assert all(type(k) is int and type(v) is str for k, v in itos.items())


def test_build_vocab_shakespeare(text):
    stoi, itos = build_vocab(text)
    assert len(stoi) == len(itos) == 65
    assert (stoi["\n"], stoi[" "], stoi["A"], stoi["a"], stoi["z"]) == (0, 1, 13, 39, 64)
    assert all(itos[stoi[c]] == c for c in stoi)


def test_encode_decode_small():
    stoi, itos = build_vocab("hello world")
    ids = encode("hold", stoi)
    assert ids == [3, 5, 4, 1]
    assert type(ids) is list and all(type(i) is int for i in ids)
    s = decode([7, 5, 6, 4, 1], itos)
    assert s == "world" and type(s) is str
    assert decode(torch.tensor([3, 2, 4, 4, 5]).tolist(), itos) == "hello"
    assert encode("", stoi) == [] and decode([], itos) == ""


def test_encode_decode_roundtrip(text):
    stoi, itos = build_vocab(text)
    ids = encode(text, stoi)
    assert len(ids) == len(text) == 1115394
    assert decode(ids, itos) == text


@pytest.mark.parametrize("n, frac", [(25, 0.9), (15, 0.9), (10, 0.5)], ids=["25-0.9", "15-0.9", "10-0.5"])
def test_split_data(n, frac):
    data = _affine(n)
    before = data.clone()
    train, val = split_data(data, frac) if frac != 0.9 else split_data(data)
    k = int(frac * n)
    assert torch.equal(train, before[:k]), f"train 應該是前 int({frac}·{n}) = {k} 個"
    assert torch.equal(val, before[k:])
    assert torch.equal(data, before), "split_data 不能改動輸入"


def test_get_batch_values():
    data = _affine(1000)
    block, batch = 8, 64
    x, y = get_batch(data, block, batch, torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (batch, block)
    assert x.dtype == y.dtype == torch.int64
    start = (x[:, 0] - 5) // 3
    pos = start[:, None] + torch.arange(block)
    assert torch.equal(x, data[pos]), "x 的每一列應該是 data 的一段連續切片 data[i : i+block]"
    assert torch.equal(y, data[pos + 1]), "y 應該是 x 往後移一格，也就是 data[i+1 : i+block+1]"


def test_get_batch_offsets():
    block = 8
    data = _affine(50)
    x, _ = get_batch(data, block, 5000, torch.Generator().manual_seed(1))
    starts = set(((x[:, 0] - 5) // 3).tolist())
    assert starts == set(range(50 - block)), (
        f"起點應該剛好涵蓋 0..{50 - block - 1}；少了 {sorted(set(range(50 - block)) - starts)}，"
        f"多了 {sorted(starts - set(range(50 - block)))}")

    data = _affine(block + 1)
    x, y = get_batch(data, block, 10, torch.Generator().manual_seed(2))
    assert torch.equal(x, data[:block].expand(10, block))
    assert torch.equal(y, data[1:].expand(10, block))


def test_get_batch_generator():
    data = _affine(500)
    before = data.clone()
    torch.manual_seed(123)
    x1, y1 = get_batch(data, 16, 32, torch.Generator().manual_seed(7))
    torch.manual_seed(456)
    x2, y2 = get_batch(data, 16, 32, torch.Generator().manual_seed(7))
    assert torch.equal(x1, x2) and torch.equal(y1, y2), "同 seed 的 generator 要得到相同結果，不受全域亂數影響"
    x3, _ = get_batch(data, 16, 32, torch.Generator().manual_seed(8))
    assert not torch.equal(x1, x3), "不同 seed 應該得到不同的 batch"
    assert torch.equal(data, before), "get_batch 不能改動輸入"
