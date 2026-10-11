"""bigram_lm.py 的測試（第 1 段）：只檢查外部行為，不限制內部寫法。介面約定見 README.md。

loss 以 F.cross_entropy 當正確答案比對。模型的訓練有兩項測試（test_train_bigram、test_train_bigram_same_seed），標記為 train，預設跳過，
加 --run-train 才執行。兩項共用同一次 seed 0 的訓練，第二項再多訓練一次，共訓練兩次。

測試內容：

| 測試                                   | 檢查什麼                                                                     |
|----------------------------------------|------------------------------------------------------------------------------|
| test_forward_no_targets                | 沒有 targets：logits 形狀 (B, T, V)、loss 是 None                             |
| test_forward_loss                      | 有 targets：loss 是 0 維 tensor，等於所有 B·T 個位置的 cross-entropy 平均；logits 可以是 (B, T, V) 或 (B·T, V) |
| test_forward_is_bigram                 | 位置 t 的 logits 只由 idx[:, t] 決定：同一個 token 在哪裡都一樣，改前面的 token 不影響 |
| test_backward                          | 有參數、全部 requires_grad；loss.backward() 之後每個參數都有梯度              |
| test_generate_shape                    | 輸出 (B, T + max_new_tokens)、int64、前 T 欄就是輸入、值在 0..V−1；不改動輸入 |
| test_generate_generator                | 同 seed 結果相同、不受全域亂數影響；不同 seed 不同                            |
| test_generate_feeds_back               | 用覆寫 forward 的子類別：每一步只看最後一個位置，新 token 接在後面再丟回去    |
| test_generate_samples_softmax          | 同上：20000 列抽一次，各 token 的比例和 softmax 機率相差 < 0.02，機率 0 的不出現 |
| test_estimate_loss                     | 記錄呼叫的 model：= 依序用 get_batch 抽 eval_iters 批的 loss 平均；回傳 float；eval 模式、不追蹤梯度；結束後回到 train 模式 |
| test_train_bigram（train，預設跳過）    | 整份 val 的 loss ≤ 計數法 bigram 的 val NLL + 0.02；generate＋decode 可產生文字 |
| test_train_bigram_same_seed（train）    | 呼叫前全域亂數設成 1、2，各用 seed 0 訓練一次：每個 token 的 logits 相差 ≤ 1e-4 |

門檻來源（Claude 用 scratchpad 的參考實作量，參考實作不進 repo）：資料用學習者的 split_data(整份資料)，
train 1003854、val 111540 個 token。
- 計數法 bigram（用 train 計數，smoothing 1）：train 2.4546、val 2.4819；smoothing 0.001：val 2.4913
- 參考實作照影片（nn.Embedding(V, V)、block 8、batch 32、AdamW lr 1e-2、3000 步，CPU）：seed 0／1 的整份 val loss
  2.4874／2.4867，各 1–2 s；5000 步 2.4885／2.4841；lr 1e-3、10000 步 2.4918／2.4874
- 不夠的設定：lr 1e-2 只跑 1000 步 2.5139／2.5130；lr 1e-3、3000 步 2.7983／2.8017
- 門檻 = 2.4819 + 0.02 ≈ 2.502：影片的設定低 0.015；1000 步高 0.012
- 訓練前（nn.Embedding 預設初始化）的 val loss 4.56（ln 65 = 4.17），只記錄、不檢查

生成的測試用子類別覆寫 forward：最後一個位置的 logits 由最後一個 token 決定，其他位置的 logits 一律指向 token V−1，
所以用錯位置一定會被抓到。test_generate_samples_softmax 的容許誤差：p = 0.4 時標準差 √(0.24/20000) ≈ 0.0035，
0.02 約 5.8 個標準差。

負控（Claude 用 scratchpad 的參考實作故意改錯，各自應失敗的測試都失敗）：
- loss 只算最後一個位置：test_forward_loss
- generate 用第 0 個位置的 logits：test_generate_feeds_back、test_generate_samples_softmax
- generate 用 argmax 取代抽樣：test_generate_samples_softmax、test_generate_generator（不同 seed 結果相同）
- generate 用全域亂數：test_generate_generator
- estimate_loss 沒有切 eval 模式／沒有關梯度／結束後沒有切回 train 模式：test_estimate_loss
- 只訓練 1000 步：test_train_bigram
- train_bigram 不呼叫 torch.manual_seed（參數初始化用呼叫前的全域亂數）：test_train_bigram_same_seed
- char_data.get_batch 的 y 沒有往後移、回傳 index：test_train_bigram 也會失敗

不同但正確的寫法也要能過（已確認全部通過）：有 targets 時 logits 保持 (B, T, V)；generate 只把最後一個 token 丟給 model。
"""

import math

import pytest
import torch
import torch.nn.functional as F

from bigram_lm import BigramLanguageModel, estimate_loss
from char_data import build_vocab, decode, encode, get_batch, split_data

V = 10
MARGIN = 0.02


def _rand_idx(B, T, seed=0):
    return torch.randint(V, (B, T), generator=torch.Generator().manual_seed(seed))


def _model(seed=0):
    torch.manual_seed(seed)
    return BigramLanguageModel(V)


def test_forward_no_targets():
    model = _model()
    logits, loss = model(_rand_idx(4, 6))
    assert logits.shape == (4, 6, V)
    assert loss is None


def test_forward_loss():
    model = _model()
    idx, targets = _rand_idx(4, 6, seed=1), _rand_idx(4, 6, seed=2)
    with torch.no_grad():
        ref_logits, _ = model(idx)
        logits, loss = model(idx, targets)
    assert logits.shape in ((4, 6, V), (24, V)), f"有 targets 時 logits 應該是 (4, 6, {V}) 或 (24, {V})，得到 {tuple(logits.shape)}"
    assert torch.allclose(logits.reshape(4, 6, V), ref_logits)
    assert isinstance(loss, torch.Tensor) and loss.dim() == 0
    expected = F.cross_entropy(ref_logits.reshape(-1, V), targets.reshape(-1))
    assert torch.allclose(loss, expected, atol=1e-6), f"loss {loss.item():.6f}，預期 {expected.item():.6f}"


def test_forward_is_bigram():
    model = _model()
    idx = _rand_idx(5, 7, seed=3)
    idx[0, 0] = idx[2, 5] = idx[4, 3] = 7
    with torch.no_grad():
        logits, _ = model(idx)
        assert torch.equal(logits[0, 0], logits[2, 5]) and torch.equal(logits[0, 0], logits[4, 3])
        changed = idx.clone()
        changed[:, :4] = (changed[:, :4] + 1) % V
        logits2, _ = model(changed)
    assert torch.equal(logits[:, 4:], logits2[:, 4:]), "改動前面的 token 不應該影響後面位置的 logits"


def test_backward():
    model = _model()
    params = list(model.parameters())
    assert params and all(p.requires_grad for p in params)
    _, loss = model(_rand_idx(4, 6, seed=1), _rand_idx(4, 6, seed=2))
    loss.backward()
    assert all(p.grad is not None for p in params)


def test_generate_shape():
    model = _model()
    idx = _rand_idx(3, 4, seed=5)
    before = idx.clone()
    out = model.generate(idx, 7, torch.Generator().manual_seed(0))
    assert out.shape == (3, 11) and out.dtype == torch.int64
    assert torch.equal(out[:, :4], before), "輸出的前 T 欄應該就是輸入"
    assert out.min() >= 0 and out.max() < V
    assert torch.equal(idx, before), "generate 不能改動輸入"


def test_generate_generator():
    model = _model()
    idx = torch.zeros(4, 1, dtype=torch.long)
    torch.manual_seed(1)
    a = model.generate(idx, 50, torch.Generator().manual_seed(9))
    torch.manual_seed(2)
    b = model.generate(idx, 50, torch.Generator().manual_seed(9))
    assert torch.equal(a, b), "同 seed 的 generator 要得到相同結果，不受全域亂數影響"
    c = model.generate(idx, 50, torch.Generator().manual_seed(10))
    assert not torch.equal(a, c), "不同 seed 應該抽到不同的結果"


class _NextTokenStub(BigramLanguageModel):
    """最後一個位置：下一個 token 幾乎一定是 (最後一個 token + 1) % V；其他位置一律指向 V−1。"""

    def forward(self, idx, targets=None):
        B, T = idx.shape
        logits = torch.zeros(B, T, V)
        logits[:, :, V - 1] = 50.0
        logits[:, -1, :] = 0.0
        logits[torch.arange(B), -1, (idx[:, -1] + 1) % V] = 50.0
        return logits, None


PROBS = torch.tensor([0.1, 0.2, 0.3, 0.4] + [0.0] * (V - 4))


class _FixedDistStub(BigramLanguageModel):
    """最後一個位置的機率分佈固定是 PROBS；其他位置一律指向 V−1。"""

    def forward(self, idx, targets=None):
        B, T = idx.shape
        logits = torch.zeros(B, T, V)
        logits[:, :, V - 1] = 50.0
        logits[:, -1, :] = PROBS.log()
        return logits, None


def test_generate_feeds_back():
    model = _NextTokenStub(V)
    idx = torch.tensor([[0, 0, 0], [5, 5, 2]])
    out = model.generate(idx, 4, torch.Generator().manual_seed(0))
    assert out.tolist() == [[0, 0, 0, 1, 2, 3, 4], [5, 5, 2, 3, 4, 5, 6]], (
        "每一步應該只用最後一個位置的 logits，抽到的 token 接在後面，下一步再丟回 model")


def test_generate_samples_softmax():
    model = _FixedDistStub(V)
    n = 20000
    out = model.generate(torch.zeros(n, 3, dtype=torch.long), 1, torch.Generator().manual_seed(0))
    freq = torch.bincount(out[:, 3], minlength=V).float() / n
    assert freq[4:].sum() == 0, "機率是 0 的 token 不應該被抽到"
    assert torch.allclose(freq, PROBS, atol=0.02), f"抽樣比例 {freq[:4].tolist()}，應接近 {PROBS[:4].tolist()}"


class _Recorder(torch.nn.Module):
    """loss = x 的平均（值由抽到的 batch 決定）；記錄每次呼叫時的模式與梯度狀態。"""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(()))
        self.calls = []

    def forward(self, idx, targets=None):
        self.calls.append((self.training, torch.is_grad_enabled(), tuple(idx.shape)))
        assert targets is not None and targets.shape == idx.shape
        return None, self.w * idx.float().mean()


def test_estimate_loss():
    data = 5 + 3 * torch.arange(300, dtype=torch.long)
    model = _Recorder()
    model.train()
    got = estimate_loss(model, data, 8, 4, 10, torch.Generator().manual_seed(3))
    g = torch.Generator().manual_seed(3)
    expected = sum(get_batch(data, 8, 4, g)[0].float().mean().item() for _ in range(10)) / 10
    assert type(got) is float, f"estimate_loss 應該回傳 Python float，得到 {type(got)}"
    assert len(model.calls) == 10, f"應該呼叫 model 10 次（eval_iters），實際 {len(model.calls)} 次"
    assert all(shape == (4, 8) for _, _, shape in model.calls)
    assert not any(training for training, _, _ in model.calls), "呼叫 model 時應該是 eval 模式（model.eval()）"
    assert not any(grad for _, grad, _ in model.calls), "estimate_loss 裡不應該追蹤梯度"
    assert model.training, "estimate_loss 結束後應該切回 train 模式"
    assert math.isclose(got, expected, rel_tol=1e-5), (
        f"得到 {got:.6f}，預期 {expected:.6f}（依序用 get_batch 抽 10 批、loss 取平均）")


def _full_loss(model, seq):
    with torch.no_grad():
        _, loss = model(seq[:-1][None], seq[1:][None])
    return loss.item()


@pytest.fixture(scope="module")
def shakespeare(text):
    stoi, itos = build_vocab(text)
    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    train, val = split_data(data)
    return stoi, itos, train, val


def _train_seed0(train, vocab, global_seed):
    """先把全域亂數設成 global_seed，再用 seed 0 訓練；train_bigram 的結果不應該受 global_seed 影響。"""
    from bigram_lm import train_bigram

    torch.manual_seed(global_seed)
    model = train_bigram(train, vocab, 0).to("cpu")
    model.eval()
    return model


@pytest.fixture(scope="module")
def trained(shakespeare):
    stoi, _, train, _ = shakespeare
    return _train_seed0(train, len(stoi), global_seed=1)


@pytest.mark.train
def test_train_bigram(shakespeare, trained):
    stoi, itos, train, val = shakespeare
    vocab = len(stoi)

    N = torch.zeros(vocab, vocab, dtype=torch.float64)
    N.index_put_((train[:-1], train[1:]), torch.ones(len(train) - 1, dtype=torch.float64), accumulate=True)
    P = (N + 1) / (N + 1).sum(1, keepdim=True)
    count_val = -P[val[:-1], val[1:]].log().mean().item()

    model = trained
    val_loss = _full_loss(model, val)
    print(f"\n計數法 bigram val NLL {count_val:.4f}；門檻 {count_val + MARGIN:.4f}；"
          f"train loss {_full_loss(model, train):.4f}；val loss {val_loss:.4f}")
    assert val_loss <= count_val + MARGIN, f"整份 val 的 loss {val_loss:.4f}，門檻 {count_val + MARGIN:.4f}"

    out = model.generate(torch.zeros(1, 1, dtype=torch.long), 200, torch.Generator().manual_seed(0))
    sample = decode(out[0].tolist(), itos)
    print(sample)
    assert len(sample) == 201 and set(sample) <= set(stoi)


@pytest.mark.train
def test_train_bigram_same_seed(shakespeare, trained):
    stoi, _, train, val = shakespeare
    vocab = len(stoi)
    again = _train_seed0(train, vocab, global_seed=2)
    tokens = torch.arange(vocab)[None]
    with torch.no_grad():
        a, _ = trained(tokens)
        b, _ = again(tokens)
    diff = (a - b).abs().max().item()
    print(f"\n兩次訓練的 logits 最大差 {diff:.3g}；val loss {_full_loss(trained, val):.6f} vs {_full_loss(again, val):.6f}")
    assert torch.allclose(a, b, atol=1e-4), (
        f"同 seed 訓練兩次（呼叫前的全域亂數狀態不同）得到不同的模型：logits 最大差 {diff:.3g}。"
        "所有隨機性（包括參數初始化）都要來自 seed")
