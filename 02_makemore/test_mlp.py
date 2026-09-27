"""mlp.py 的測試：只檢查外部行為，不限制內部寫法。介面約定見 README.md。

evaluate 與 sample 用測試自己做的假 model 測，假 model 只提供 README 約定的 `model(X) -> (n, 27) logits`。
模型本身與訓練只有一項測試（test_train_mlp_beats_bigram），標記為 train，預設跳過，加 --run-train 才執行。

測試內容：

| 測試                                           | 檢查什麼                                                              |
|------------------------------------------------|-----------------------------------------------------------------------|
| test_build_dataset_emma                        | README 的 "emma"、block_size 3 例子，逐筆比對                         |
| test_build_dataset_order_and_duplicates        | 依 words 順序、名字內依位置；重複的名字各算一次                       |
| test_build_dataset_short_name_long_context     | 長度 1 的名字配 block_size 5，前面用 '.' 補滿                         |
| test_build_dataset_shapes[1／3／5]             | 整份資料：形狀、dtype、CPU、筆數 = 所有名字長度 + 1 的總和（228146）  |
| test_build_dataset_block1_matches_bigrams      | block_size 1 時，(X[:, 0], Y) 的計數與第 1 段 golden 的 27×27 計數相同 |
| test_split_sizes[...]                          | 三份的長度符合 README 的公式（n = 1、7、10、32033）                   |
| test_split_is_a_partition                      | 三份合起來（多重集合）等於原本的 words                                |
| test_split_is_shuffled                         | train 不是原本 words 的前 80%                                         |
| test_split_same_seed_same_result               | 同 seed 結果相同，而且不受全域亂數狀態（random、torch）影響           |
| test_split_different_seed_differs              | 不同 seed 結果不同                                                    |
| test_split_does_not_mutate_input               | 不改動傳入的 words                                                    |
| test_evaluate_uniform_is_log_27                | 全 0 logits → ln 27                                                   |
| test_evaluate_hand_computed                    | 手算的小例子：正確答案 logit 為 L、其他為 0 → −ln(e^L / (e^L + 26))   |
| test_evaluate_uses_each_row                    | 每一列的 logits 依 X 而不同時，逐列對應到正確的 Y                     |
| test_evaluate_returns_float_and_keeps_inputs   | 回傳 Python float；不改動 X、Y                                        |
| test_evaluate_large_input                      | 228146 列一次處理                                                     |
| test_sample_follows_context[1／3／5]           | 假 model 依完整 context 決定下一個字元，檢查 context 的初始值與移動方向 |
| test_sample_empty_name                         | 第一步就取到 '.' → 空字串                                             |
| test_sample_same_seed_same_result              | 同 seed 結果相同；不受全域亂數影響；只含 a–z                          |
| test_sample_distribution                       | P('.')=0.25、P('a')=0.5、P('b')=0.25：平均長度約 3，a:b 約 2:1        |
| test_train_mlp_beats_bigram（train，預設跳過） | split_words(words, 42) 的 train 訓練；dev NLL ≤ bigram 在 dev 上的 NLL − 0.2 |

test_train_mlp_beats_bigram 的門檻來源（Claude 用 scratchpad 的參考 MLP 量，參考實作不進 repo）：
split seed 42 時，bigram（train 上計數、平滑 1）的 dev NLL 為 2.4533；參考 MLP（block_size 3、embedding 10、
隱藏層 200、batch 32）訓練 2 萬步（約 5 s）dev 比 bigram 低 0.27–0.28，5 萬步（約 12 s）低 0.32，20 萬步
（約 46 s）低 0.35。門檻定為低 0.2。

evaluate 與 sample 的假 model 會檢查收到的 X 形狀與 dtype（README：(n, block_size)、int64），
不符合時測試會以 AssertionError 失敗；被呼叫超過 MAX_CALLS 次也會失敗（sample 停不下來時）。
"""

import json
import math
import random
from collections import Counter
from pathlib import Path

import pytest
import torch

from mlp import build_dataset, evaluate, sample, split_words

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
IX = {".": 0} | {c: i + 1 for i, c in enumerate(ALPHABET)}  # README 約定的字元表
GOLDEN_BIGRAM = json.loads((Path(__file__).parent / "golden" / "bigram.json").read_text())
NEG = -1e9  # 當作「機率 0」的 logit


def _ctx(s):
    return [IX[c] for c in s]


def _logits_for(probs):
    """{字元: 機率} → 長度 27 的 logits（log 機率；沒列到的字元是 NEG）。"""
    row = [NEG] * 27
    for c, p in probs.items():
        row[IX[c]] = math.log(p)
    return row


MAX_CALLS = 100_000  # 測試裡最多約 1.2 萬次；超過代表 sample 停不下來


class FakeModel:
    """依每一列 context 查表決定 logits；查不到的 context 一律輸出 '.'。

    被呼叫超過 MAX_CALLS 次就讓測試失敗，避免 sample 停不下來時整個 pytest 卡住。
    """

    def __init__(self, block_size, table, default=None):
        self.block_size = block_size
        self.table = {tuple(_ctx(k)): _logits_for(v) for k, v in table.items()}
        self.default = _logits_for(default or {".": 1.0})
        self.calls = 0

    def __call__(self, X):
        assert isinstance(X, torch.Tensor), f"model 收到的 X 不是 tensor：{type(X)}"
        assert X.dtype == torch.int64, f"model 收到的 X dtype 是 {X.dtype}，README 約定是 torch.int64"
        assert X.dim() == 2 and X.shape[1] == self.block_size, (
            f"model 收到的 X 形狀是 {tuple(X.shape)}，README 約定是 (n, {self.block_size})"
        )
        self.calls += 1
        assert self.calls <= MAX_CALLS, (
            f"model 已被呼叫超過 {MAX_CALLS} 次，sample 可能停不下來（取到 '.' 有沒有結束？有沒有依機率取樣？）"
        )
        rows = [self.table.get(tuple(r), self.default) for r in X.tolist()]
        return torch.tensor(rows, dtype=torch.float32)


# ---------- build_dataset ----------


def test_build_dataset_emma():
    X, Y = build_dataset(["emma"], 3)
    assert X.tolist() == [_ctx("..."), _ctx("..e"), _ctx(".em"), _ctx("emm"), _ctx("mma")]
    assert Y.tolist() == _ctx("emma.")


def test_build_dataset_order_and_duplicates():
    X, Y = build_dataset(["ab", "c", "ab"], 2)
    assert X.tolist() == [
        _ctx(".."), _ctx(".a"), _ctx("ab"),  # "ab"
        _ctx(".."), _ctx(".c"),              # "c"
        _ctx(".."), _ctx(".a"), _ctx("ab"),  # "ab"（重複，再算一次）
    ]
    assert Y.tolist() == _ctx("ab.") + _ctx("c.") + _ctx("ab.")


def test_build_dataset_short_name_long_context():
    X, Y = build_dataset(["a"], 5)
    assert X.tolist() == [_ctx("....."), _ctx("....a")]
    assert Y.tolist() == _ctx("a.")


@pytest.mark.parametrize("block_size", [1, 3, 5])
def test_build_dataset_shapes(words, block_size):
    X, Y = build_dataset(words, block_size)
    m = sum(len(w) + 1 for w in words)
    assert isinstance(X, torch.Tensor) and isinstance(Y, torch.Tensor)
    assert tuple(X.shape) == (m, block_size)
    assert tuple(Y.shape) == (m,)
    assert X.dtype == torch.int64 and Y.dtype == torch.int64
    assert X.device.type == "cpu" and Y.device.type == "cpu"
    assert int(X.min()) >= 0 and int(X.max()) <= 26
    assert int(Y.min()) >= 0 and int(Y.max()) <= 26


def test_build_dataset_block1_matches_bigrams(words):
    X, Y = build_dataset(words, 1)
    N = torch.zeros((27, 27), dtype=torch.int64)
    N.index_put_((X[:, 0], Y), torch.ones_like(Y), accumulate=True)
    assert torch.equal(N, torch.tensor(GOLDEN_BIGRAM["counts"], dtype=torch.int64))


# ---------- split_words ----------


@pytest.mark.parametrize("n", [1, 7, 10, 32033])
def test_split_sizes(words, n):
    ws = words[:n]
    train, dev, test = split_words(ws, 0)
    assert all(isinstance(part, list) for part in (train, dev, test))
    assert len(train) == int(0.8 * n)
    assert len(dev) == int(0.9 * n) - int(0.8 * n)
    assert len(test) == n - int(0.9 * n)


def test_split_is_a_partition(words):
    train, dev, test = split_words(words, 0)
    assert Counter(train) + Counter(dev) + Counter(test) == Counter(words)


def test_split_is_shuffled(words):
    train, _, _ = split_words(words, 0)
    assert train != words[: len(train)]


def test_split_same_seed_same_result(words):
    random.seed(1)
    torch.manual_seed(1)
    a = split_words(words, 42)
    random.seed(2)
    torch.manual_seed(2)
    b = split_words(words, 42)
    assert a == b


def test_split_different_seed_differs(words):
    assert split_words(words, 1)[0] != split_words(words, 2)[0]


def test_split_does_not_mutate_input(words):
    ws = list(words)
    split_words(ws, 0)
    assert ws == words


# ---------- evaluate ----------


def test_evaluate_uniform_is_log_27():
    X = torch.zeros((5, 3), dtype=torch.int64)
    Y = torch.tensor([0, 1, 2, 3, 26])

    def uniform(X):
        return torch.zeros((X.shape[0], 27))

    assert evaluate(uniform, X, Y) == pytest.approx(math.log(27), abs=1e-5)


def test_evaluate_hand_computed():
    # 每一列：正確答案的 logit 是 L，其他 26 個是 0 → p = e^L / (e^L + 26)
    L = 2.0
    X = torch.zeros((4, 2), dtype=torch.int64)
    Y = torch.tensor([5, 0, 26, 5])

    def model(X):
        logits = torch.zeros((X.shape[0], 27))
        logits[torch.arange(4), Y] = L
        return logits

    expected = -math.log(math.exp(L) / (math.exp(L) + 26))
    assert evaluate(model, X, Y) == pytest.approx(expected, abs=1e-5)


def test_evaluate_uses_each_row():
    # 假 model 依 context 給出不同的分佈；逐列手算 −ln p 再平均
    model = FakeModel(2, {
        "..": {"a": 0.5, "b": 0.5},
        ".a": {"a": 0.1, ".": 0.9},
        "ab": {"c": 0.25, ".": 0.75},
    })
    X = torch.tensor([_ctx(".."), _ctx(".a"), _ctx("ab"), _ctx("..")])
    Y = torch.tensor(_ctx("b.ca"))
    expected = -(math.log(0.5) + math.log(0.9) + math.log(0.25) + math.log(0.5)) / 4
    assert evaluate(model, X, Y) == pytest.approx(expected, abs=1e-5)


def test_evaluate_returns_float_and_keeps_inputs():
    X = torch.tensor([_ctx("..."), _ctx("..a")])
    Y = torch.tensor([1, 0])
    X0, Y0 = X.clone(), Y.clone()

    def uniform(X):
        return torch.zeros((X.shape[0], 27))

    result = evaluate(uniform, X, Y)
    assert type(result) is float
    assert torch.equal(X, X0) and torch.equal(Y, Y0)


def test_evaluate_large_input():
    g = torch.Generator().manual_seed(0)
    X = torch.randint(0, 27, (228146, 3), generator=g)
    Y = torch.randint(0, 27, (228146,), generator=g)

    def uniform(X):
        return torch.zeros((X.shape[0], 27))

    assert evaluate(uniform, X, Y) == pytest.approx(math.log(27), abs=1e-4)


# ---------- sample ----------


@pytest.mark.parametrize("block_size", [1, 3, 5])
def test_sample_follows_context(block_size):
    # 每一步的下一個字元完全由整個 context 決定；context 的初始值或移動方向錯了，
    # 就會查不到表而輸出 '.'，得到比 "bcdef" 短的名字
    name = "bcdef"
    table = {}
    ctx = "." * block_size
    for ch in name + ".":
        table[ctx] = {ch: 1.0}
        ctx = (ctx + ch)[-block_size:]
    model = FakeModel(block_size, table)
    assert sample(model, block_size, 3, torch.Generator().manual_seed(0)) == [name] * 3


def test_sample_empty_name():
    model = FakeModel(3, {"...": {".": 1.0}})
    assert sample(model, 3, 2, torch.Generator().manual_seed(0)) == ["", ""]


def _mixed_model():
    # 不論 context，下一個字元：'.' 0.25、'a' 0.5、'b' 0.25
    return FakeModel(3, {}, default={".": 0.25, "a": 0.5, "b": 0.25})


def test_sample_same_seed_same_result():
    torch.manual_seed(1)
    a = sample(_mixed_model(), 3, 50, torch.Generator().manual_seed(7))
    torch.manual_seed(2)
    b = sample(_mixed_model(), 3, 50, torch.Generator().manual_seed(7))
    assert a == b
    assert isinstance(a, list) and len(a) == 50
    assert all(isinstance(s, str) and set(s) <= set(ALPHABET) for s in a)


def test_sample_distribution():
    names = sample(_mixed_model(), 3, 3000, torch.Generator().manual_seed(0))
    # 每一步結束的機率是 0.25 → 平均長度 (1 − 0.25) / 0.25 = 3
    mean_len = sum(map(len, names)) / len(names)
    assert mean_len == pytest.approx(3.0, abs=0.3)
    counts = Counter("".join(names))
    assert set(counts) <= {"a", "b"}
    assert counts["a"] / counts["b"] == pytest.approx(2.0, rel=0.15)


# ---------- 模型與訓練（預設跳過，加 --run-train 才執行） ----------

TRAIN_SPLIT_SEED = 42
MARGIN = 0.2


def _bigram_dev_nll(train, dev):
    """第 1 段的計數法：只在 train 上計數（平滑 1），再算 dev 上的平均 NLL。"""
    counts = Counter()
    for w in train:
        s = "." + w + "."
        counts.update(zip(s, s[1:]))
    N = torch.ones((27, 27), dtype=torch.float64)
    for (a, b), c in counts.items():
        N[IX[a], IX[b]] += c
    P = (N / N.sum(1, keepdim=True)).tolist()
    total, m = 0.0, 0
    for w in dev:
        s = "." + w + "."
        for a, b in zip(s, s[1:]):
            total -= math.log(P[IX[a]][IX[b]])
            m += 1
    return total / m


@pytest.mark.train
def test_train_mlp_beats_bigram(words):
    from mlp import train_mlp

    train, dev, _ = split_words(words, TRAIN_SPLIT_SEED)
    model, block_size = train_mlp(train, 0)
    assert type(block_size) is int and block_size >= 1, f"block_size 要是正整數，拿到 {block_size!r}"

    Xd, Yd = build_dataset(dev, block_size)
    with torch.no_grad():
        out = model(Xd[:5])
    assert tuple(out.shape) == (5, 27), f"model(X) 的形狀是 {tuple(out.shape)}，README 約定是 (n, 27)"
    assert out.device.type == "cpu", f"model(X) 回傳的 tensor 在 {out.device}，README 約定要在 CPU 上"

    dev_nll = evaluate(model, Xd, Yd)
    base = _bigram_dev_nll(train, dev)
    print(f"\nMLP dev NLL {dev_nll:.4f}；bigram dev NLL {base:.4f}；門檻 {base - MARGIN:.4f}")
    assert dev_nll <= base - MARGIN, (
        f"dev NLL {dev_nll:.4f}，要 ≤ {base - MARGIN:.4f}（bigram 在 dev 上的 {base:.4f} − {MARGIN}）"
    )

    names = sample(model, block_size, 5, torch.Generator().manual_seed(0))
    assert len(names) == 5 and all(set(s) <= set(ALPHABET) for s in names)
