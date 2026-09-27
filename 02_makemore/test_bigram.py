"""bigram.py 的測試：只檢查外部行為，不限制內部寫法。介面約定見 README.md。

測試內容：

| 測試                                         | 檢查什麼                                                          |
|----------------------------------------------|-------------------------------------------------------------------|
| test_vocab                                   | 字元表的編號與大小                                                |
| test_count_bigrams_single_word／_repeated_pairs | 手算得出的小例子（["ab"]、["aa", "a"]）                        |
| test_count_bigrams_matches_golden            | 整份資料的 27×27 計數與 golden 完全相同                           |
| test_count_probs_rows_sum_to_one             | 平滑 0 和 1 時，每一列加總都是 1                                  |
| test_count_probs_smoothing                   | 平滑 0 時沒出現過的組合機率是 0；平滑 1 時沒有 0；預設平滑是 1    |
| test_count_probs_small_case                  | ["ab"] 的機率值                                                   |
| test_nll_uniform_is_log_27                   | 均勻分佈下 NLL = ln 27 ≈ 3.2958                                   |
| test_nll_hand_computed                       | 手算的小例子                                                      |
| test_nll_is_a_python_float                   | 回傳型別是 float，不是 tensor                                     |
| test_nll_matches_golden[0]／[1]              | 整份資料上的 NLL：平滑 0 ≈ 2.4540，平滑 1 ≈ 2.4546                |
| test_sample_deterministic_chain              | 機率只有 0 和 1 時，取樣結果是固定的                              |
| test_sample_same_seed_same_result            | 同一個 seed 結果相同；只含 a–z                                    |
| test_sample_mean_length_matches_data         | 取 3000 個名字，平均長度接近資料的平均長度（約 6.12）             |
| test_train_nn_matches_counting               | 網路輸出是機率分佈；NLL 不低於計數法的最小值，也不高於平滑 1 的 NLL + 0.05 |

golden/bigram.json 的來源：data/names.txt，由 Claude 用一份只存在於 scratchpad 的參考實作
算出來，參考實作不進 repo（它就是這一段的答案）。其中幾個數字可以和 Karpathy 影片獨立對照，
都一致：32033 個名字、228146 個 bigram、'n.' 6763 次、'a.' 6640 次、'an' 5438 次、
'.a' 4410 次、未平滑的 NLL 2.4540（影片 2.4541）。

測試本身的驗證（Claude 執行）：參考實作 16/16 通過；對參考實作做 11 種錯誤改動，每一種都
至少讓一項測試失敗。
"""

import json
import math
from pathlib import Path

import pytest
import torch

from bigram import count_bigrams, count_probs, itos, nll, sample, stoi, train_nn

GOLDEN = json.loads((Path(__file__).parent / "golden" / "bigram.json").read_text())
# 取樣測試用 golden 計數建 P，不依賴 count_bigrams：計數寫錯時（例如名字結尾沒補 '.'），
# 某些字元永遠轉移不到 '.'，取樣會停不下來
GOLDEN_N = torch.tensor(GOLDEN["counts"], dtype=torch.int64)


def _probs(entries):
    """依 {(前一個字元, 下一個字元): 機率} 建出 27×27 的 P；沒列到的列用均勻分佈。"""
    P = torch.full((27, 27), 1 / 27)
    for a in {a for a, _ in entries}:
        P[stoi[a]] = 0.0
    for (a, b), p in entries.items():
        P[stoi[a], stoi[b]] = p
    return P


# ---------- 字元表 ----------


def test_vocab():
    assert stoi["."] == 0
    assert [stoi[c] for c in "abcdefghijklmnopqrstuvwxyz"] == list(range(1, 27))
    assert all(itos[i] == c for c, i in stoi.items())
    assert len(stoi) == len(itos) == 27


# ---------- 計數 ----------


def test_count_bigrams_single_word():
    N = count_bigrams(["ab"])
    assert tuple(N.shape) == (27, 27)
    # .a、ab、b.
    assert N[0, 1] == 1 and N[1, 2] == 1 and N[2, 0] == 1
    assert N.sum() == 3


def test_count_bigrams_repeated_pairs():
    N = count_bigrams(["aa", "a"])
    # "aa"：.a、aa、a.；"a"：.a、a.
    assert N[0, 1] == 2 and N[1, 1] == 1 and N[1, 0] == 2
    assert N.sum() == 5


def test_count_bigrams_matches_golden(words):
    assert len(words) == GOLDEN["num_words"]
    N = count_bigrams(words)
    assert N.sum() == GOLDEN["num_bigrams"]
    assert torch.equal(N.to(torch.int64), GOLDEN_N)


# ---------- 機率 ----------


def test_count_probs_rows_sum_to_one(words):
    N = count_bigrams(words)
    for s in (0, 1):
        P = count_probs(N, s)
        assert tuple(P.shape) == (27, 27)
        assert torch.allclose(P.sum(1).double(), torch.ones(27, dtype=torch.float64), atol=1e-5)


def test_count_probs_smoothing(words):
    N = count_bigrams(words)
    P0 = count_probs(N, 0)
    P1 = count_probs(N, 1)
    assert (P0[N == 0] == 0).all()
    assert (P1 > 0).all()
    assert (P1 == count_probs(N)).all()  # smoothing 預設為 1


def test_count_probs_small_case():
    N = count_bigrams(["ab"])
    P = count_probs(N, 0)
    assert P[0, 1] == pytest.approx(1.0)
    P = count_probs(N, 1)
    # '.' 那一列：.a 出現 1 次，加 1 後是 2；其他 26 格各是 1；總共 28
    assert P[0, 1] == pytest.approx(2 / 28)
    assert P[0, 2] == pytest.approx(1 / 28)


# ---------- NLL ----------


def test_nll_uniform_is_log_27(words):
    P = torch.full((27, 27), 1 / 27)
    assert nll(P, words) == pytest.approx(math.log(27), abs=1e-5)


def test_nll_hand_computed():
    # "ab" 的 bigram：.a、ab、b.，機率 0.5、0.25、1.0
    P = _probs({(".", "a"): 0.5, (".", "b"): 0.5, ("a", "b"): 0.25, ("a", "c"): 0.75, ("b", "."): 1.0})
    expected = -(math.log(0.5) + math.log(0.25) + math.log(1.0)) / 3
    assert nll(P, ["ab"]) == pytest.approx(expected, abs=1e-6)


def test_nll_is_a_python_float():
    assert isinstance(nll(torch.full((27, 27), 1 / 27), ["ab"]), float)


@pytest.mark.parametrize("smoothing", [0, 1])
def test_nll_matches_golden(words, smoothing):
    P = count_probs(count_bigrams(words), smoothing)
    assert nll(P, words) == pytest.approx(GOLDEN[f"nll_smoothing_{smoothing}"], abs=1e-4)


# ---------- 取樣 ----------


def test_sample_deterministic_chain():
    # . → a → . 的機率都是 1，所以每次都只會得到 "a"
    P = _probs({(".", "a"): 1.0, ("a", "."): 1.0})
    assert sample(P, 5, torch.Generator().manual_seed(0)) == ["a"] * 5
    # . → b → c → .
    P = _probs({(".", "b"): 1.0, ("b", "c"): 1.0, ("c", "."): 1.0})
    assert sample(P, 3, torch.Generator().manual_seed(0)) == ["bc"] * 3


def test_sample_same_seed_same_result():
    P = count_probs(GOLDEN_N)
    a = sample(P, 20, torch.Generator().manual_seed(123))
    b = sample(P, 20, torch.Generator().manual_seed(123))
    assert a == b
    assert len(a) == 20
    assert all(isinstance(s, str) and set(s) <= set("abcdefghijklmnopqrstuvwxyz") for s in a)


def test_sample_mean_length_matches_data(words):
    # 用未平滑的計數機率取樣時，名字的期望長度等於資料的平均長度（約 6.12）
    P = count_probs(GOLDEN_N, 0)
    names = sample(P, 3000, torch.Generator().manual_seed(0))
    data_mean = sum(map(len, words)) / len(words)
    sample_mean = sum(map(len, names)) / len(names)
    assert sample_mean == pytest.approx(data_mean, abs=0.4)


# ---------- 神經網路版 ----------


def test_train_nn_matches_counting(words):
    P = train_nn(words)
    assert tuple(P.shape) == (27, 27)
    P = P.detach()
    assert (P >= 0).all()
    assert torch.allclose(P.sum(1).double(), torch.ones(27, dtype=torch.float64), atol=1e-4)
    loss = nll(P, words)
    # 計數法（未平滑）就是訓練資料上 NLL 的最小值，神經網路版不可能比它更低
    assert loss >= GOLDEN["nll_smoothing_0"] - 1e-3
    # 訓練好的網路應該接近計數法
    assert loss <= GOLDEN["nll_smoothing_1"] + 0.05
