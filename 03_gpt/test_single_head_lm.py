"""single_head_lm.py 的測試（第 2 段）：只檢查外部行為，不限制內部寫法。介面約定見 README.md。

loss 以 F.cross_entropy 當正確答案比對。訓練有兩項測試，標記為 train，預設跳過，加 --run-train 才執行；兩項共用
同一次 seed 0 的訓練，第二項再多訓練一次，共訓練兩次。

測試內容（模型用 SingleHeadLM(10, 16, 8)，建立前先 torch.manual_seed(0)）：

| 測試                                       | 檢查什麼                                                                  |
|--------------------------------------------|---------------------------------------------------------------------------|
| test_forward_shapes[T=8／3]                 | 沒有 targets：logits (B, T, V)、loss 是 None；T < block_size 也能算；model.block_size |
| test_forward_loss                          | loss = 全部 B·T 個位置的 cross-entropy 平均；logits 可以是 (B, T, V) 或 (B·T, V) |
| test_causal                                | 改動 t0 之後的 token，t0 之前的 logits 不變                                |
| test_uses_position                         | 全部是同一個 token 時，不同位置的 logits 不同（有 position embedding）      |
| test_uses_context                          | 改動第 0 個 token，最後一個位置的 logits 會變（有 attention，不是 bigram）   |
| test_backward                              | 每個參數都有梯度                                                          |
| test_generate_shape                        | 從長度 2 生成 20 個（超過 block_size）：形狀、int64、前綴不變、值在 0..V−1；不改動輸入 |
| test_generate_long_prompt                  | 輸入本身就比 block_size 長也能生成                                         |
| test_generate_generator                    | 同 seed 結果相同、不受全域亂數影響；不同 seed 不同                         |
| test_generate_crops_context                | 覆寫 forward 的子類別：每一步丟進 forward 的剛好是最後 min(長度, block_size) 個 token；新 token 接回去 |
| test_generate_samples_softmax              | 同上：20000 列抽一次，比例和 softmax 機率相差 < 0.02，機率 0 的不出現       |
| test_train_single_head（train）             | val 切成長度 block_size 的段落算 loss ≤ 計數法 bigram 的 val NLL − 0.05；可生成文字 |
| test_train_single_head_same_seed（train）   | 呼叫前全域亂數設成 1、2，各用 seed 0 訓練一次：每個位置的 logits 相差 ≤ 1e-4 |

門檻來源（Claude 用 scratchpad 的參考實作量，參考實作不進 repo）：模型照影片（token + position embedding、
Head(n_embd, n_embd, block_size)、lm_head），split 用 split_data(整份資料)。val loss 的算法：val 切成不重疊、長度
block_size 的段落（最後不足一段的丟掉），所有位置一起平均（每段的前幾個位置 context 比較短）。
- 計數法 bigram（train 計數、每格加 1）val NLL 2.4819；bigram 和位置無關，切段落算也一樣
- 影片的設定（n_embd 32、block 8、batch 32、AdamW lr 1e-3、5000 步，CPU）：seed 0／1 val 2.4009／2.3982，各 3–4 s
- 10000 步 2.3763／2.3731；lr 1e-2 5000 步 2.4244／2.4139；block 32、lr 1e-2 5000 步 2.3648／2.3766
- 不夠的設定：1000 步 2.5123／2.5147
- 門檻 = 2.4819 − 0.05 ≈ 2.432：影片的設定低 0.03；只能做到 bigram 程度的模型（約 2.48）過不了

負控（Claude 用 scratchpad 的參考實作故意改錯，各自應失敗的測試都失敗）：
- 沒有加 position embedding（表還在但沒用到）：test_uses_position、test_backward、test_train_single_head
- 沒有經過 Head（embedding 直接接 lm_head）：test_uses_context、test_backward、test_train_single_head
- generate 沒有截斷 context：test_generate_shape、_long_prompt、_generator 報 index 錯誤，test_generate_crops_context，
  test_train_single_head
- generate 只丟最後一個 token：test_generate_crops_context
- generate 用全域亂數：test_generate_generator；用 argmax：test_generate_generator、test_generate_samples_softmax
- 只訓練 1000 步：test_train_single_head
- train_single_head 不呼叫 torch.manual_seed：test_train_single_head_same_seed

不同但正確的寫法也要能過（已確認通過）：有 targets 時把 logits reshape 成 (B·T, V)。
"""

import pytest
import torch
import torch.nn.functional as F

from char_data import build_vocab, decode, encode, split_data
from single_head_lm import SingleHeadLM

V, N_EMBD, BLOCK = 10, 16, 8
MARGIN = 0.05


def _rand_idx(B, T, seed=0):
    return torch.randint(V, (B, T), generator=torch.Generator().manual_seed(seed))


def _model():
    torch.manual_seed(0)
    return SingleHeadLM(V, N_EMBD, BLOCK)


@pytest.mark.parametrize("T", [BLOCK, 3], ids=["T=8", "T=3"])
def test_forward_shapes(T):
    model = _model()
    assert model.block_size == BLOCK
    logits, loss = model(_rand_idx(4, T))
    assert logits.shape == (4, T, V)
    assert loss is None


def test_forward_loss():
    model = _model()
    idx, targets = _rand_idx(4, BLOCK, seed=1), _rand_idx(4, BLOCK, seed=2)
    with torch.no_grad():
        ref_logits, _ = model(idx)
        logits, loss = model(idx, targets)
    assert logits.shape in ((4, BLOCK, V), (4 * BLOCK, V))
    assert torch.allclose(logits.reshape(4, BLOCK, V), ref_logits, atol=1e-6)
    assert isinstance(loss, torch.Tensor) and loss.dim() == 0
    expected = F.cross_entropy(ref_logits.reshape(-1, V), targets.reshape(-1))
    assert torch.allclose(loss, expected, atol=1e-6), f"loss {loss.item():.6f}，預期 {expected.item():.6f}"


def test_causal():
    model = _model()
    idx = _rand_idx(3, BLOCK, seed=3)
    changed = idx.clone()
    changed[:, 5:] = (changed[:, 5:] + 1) % V
    with torch.no_grad():
        a, _ = model(idx)
        b, _ = model(changed)
    assert torch.allclose(a[:, :5], b[:, :5], atol=1e-6), "位置 t 的 logits 不能看到 t 之後的 token"


def test_uses_position():
    model = _model()
    with torch.no_grad():
        logits, _ = model(torch.full((1, BLOCK), 3, dtype=torch.long))
    assert not torch.allclose(logits[0, 0], logits[0, BLOCK - 1], atol=1e-4), (
        "輸入全是同一個 token 時，不同位置的 logits 應該不同：模型要知道每個 token 的位置")


def test_uses_context():
    model = _model()
    idx = _rand_idx(2, BLOCK, seed=4)
    changed = idx.clone()
    changed[:, 0] = (changed[:, 0] + 1) % V
    with torch.no_grad():
        a, _ = model(idx)
        b, _ = model(changed)
    assert not torch.allclose(a[:, -1], b[:, -1], atol=1e-6), (
        "改動第 0 個 token，最後一個位置的 logits 應該會變：每個位置要能看到前面的 token")


def test_backward():
    model = _model()
    params = list(model.parameters())
    assert params and all(p.requires_grad for p in params)
    _, loss = model(_rand_idx(4, BLOCK, seed=1), _rand_idx(4, BLOCK, seed=2))
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"這些參數沒有梯度：{missing}"


def test_generate_shape():
    model = _model()
    idx = _rand_idx(3, 2, seed=5)
    before = idx.clone()
    out = model.generate(idx, 20, torch.Generator().manual_seed(0))
    assert out.shape == (3, 22) and out.dtype == torch.int64
    assert torch.equal(out[:, :2], before)
    assert out.min() >= 0 and out.max() < V
    assert torch.equal(idx, before), "generate 不能改動輸入"


def test_generate_long_prompt():
    model = _model()
    out = model.generate(_rand_idx(2, BLOCK + 5, seed=6), 5, torch.Generator().manual_seed(0))
    assert out.shape == (2, BLOCK + 10)


def test_generate_generator():
    model = _model()
    idx = torch.zeros(4, 1, dtype=torch.long)
    torch.manual_seed(1)
    a = model.generate(idx, 30, torch.Generator().manual_seed(9))
    torch.manual_seed(2)
    b = model.generate(idx, 30, torch.Generator().manual_seed(9))
    assert torch.equal(a, b), "同 seed 的 generator 要得到相同結果，不受全域亂數影響"
    c = model.generate(idx, 30, torch.Generator().manual_seed(10))
    assert not torch.equal(a, c), "不同 seed 應該抽到不同的結果"


class _CropStub(SingleHeadLM):
    """記錄每次丟進 forward 的 idx；最後一個位置的下一個 token 幾乎一定是 (最後一個 token + 1) % V，其他位置指向 V−1。"""

    def forward(self, idx, targets=None):
        self.seen.append(idx.clone())
        B, T = idx.shape
        logits = torch.zeros(B, T, V)
        logits[:, :, V - 1] = 50.0
        logits[:, -1, :] = 0.0
        logits[torch.arange(B), -1, (idx[:, -1] + 1) % V] = 50.0
        return logits, None


def test_generate_crops_context():
    torch.manual_seed(0)
    model = _CropStub(V, N_EMBD, 4)
    model.seen = []
    idx = torch.tensor([[0, 1, 2], [5, 5, 5]])
    out = model.generate(idx, 6, torch.Generator().manual_seed(0))
    assert out.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8], [5, 5, 5, 6, 7, 8, 9, 0, 1]], (
        "每一步應該只用最後一個位置的 logits，抽到的 token 接在後面，下一步再丟回 model")
    assert len(model.seen) == 6
    for k, seen in enumerate(model.seen):
        expected = out[:, :3 + k][:, -4:]
        assert torch.equal(seen, expected), (
            f"第 {k} 步丟進 forward 的應該是目前序列的最後 min(長度, block_size) = {expected.shape[1]} 個 token，"
            f"實際形狀 {tuple(seen.shape)}")


PROBS = torch.tensor([0.1, 0.2, 0.3, 0.4] + [0.0] * (V - 4))


class _FixedDistStub(SingleHeadLM):
    def forward(self, idx, targets=None):
        B, T = idx.shape
        logits = torch.zeros(B, T, V)
        logits[:, :, V - 1] = 50.0
        logits[:, -1, :] = PROBS.log()
        return logits, None


def test_generate_samples_softmax():
    torch.manual_seed(0)
    model = _FixedDistStub(V, N_EMBD, BLOCK)
    n = 20000
    out = model.generate(torch.zeros(n, 3, dtype=torch.long), 1, torch.Generator().manual_seed(0))
    freq = torch.bincount(out[:, 3], minlength=V).float() / n
    assert freq[4:].sum() == 0, "機率是 0 的 token 不應該被抽到"
    assert torch.allclose(freq, PROBS, atol=0.02), f"抽樣比例 {freq[:4].tolist()}，應接近 {PROBS[:4].tolist()}"


def _chunk_loss(model, seq):
    bs = model.block_size
    n = (len(seq) - 1) // bs
    x, y = seq[:n * bs].view(n, bs), seq[1:n * bs + 1].view(n, bs)
    with torch.no_grad():
        _, loss = model(x, y)
    return loss.item()


@pytest.fixture(scope="module")
def shakespeare(text):
    stoi, itos = build_vocab(text)
    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    train, val = split_data(data)
    return stoi, itos, train, val


def _train_seed0(train, vocab, global_seed):
    """先把全域亂數設成 global_seed，再用 seed 0 訓練；train_single_head 的結果不應該受 global_seed 影響。"""
    from single_head_lm import train_single_head

    torch.manual_seed(global_seed)
    model = train_single_head(train, vocab, 0).to("cpu")
    model.eval()
    return model


@pytest.fixture(scope="module")
def trained(shakespeare):
    stoi, _, train, _ = shakespeare
    return _train_seed0(train, len(stoi), global_seed=1)


@pytest.mark.train
def test_train_single_head(shakespeare, trained):
    stoi, itos, train, val = shakespeare
    vocab = len(stoi)
    N = torch.zeros(vocab, vocab, dtype=torch.float64)
    N.index_put_((train[:-1], train[1:]), torch.ones(len(train) - 1, dtype=torch.float64), accumulate=True)
    P = (N + 1) / (N + 1).sum(1, keepdim=True)
    count_val = -P[val[:-1], val[1:]].log().mean().item()

    model = trained
    val_loss = _chunk_loss(model, val)
    print(f"\nblock_size {model.block_size}；計數法 bigram val NLL {count_val:.4f}；門檻 {count_val - MARGIN:.4f}；"
          f"train loss {_chunk_loss(model, train):.4f}；val loss {val_loss:.4f}")
    assert val_loss <= count_val - MARGIN, f"val loss {val_loss:.4f}，門檻 {count_val - MARGIN:.4f}"

    out = model.generate(torch.zeros(1, 1, dtype=torch.long), 200, torch.Generator().manual_seed(0))
    sample = decode(out[0].tolist(), itos)
    print(sample)
    assert len(sample) == 201 and set(sample) <= set(stoi)


@pytest.mark.train
def test_train_single_head_same_seed(shakespeare, trained):
    stoi, _, train, val = shakespeare
    again = _train_seed0(train, len(stoi), global_seed=2)
    x = val[:4 * trained.block_size].view(4, trained.block_size)
    with torch.no_grad():
        a, _ = trained(x)
        b, _ = again(x)
    diff = (a - b).abs().max().item()
    print(f"\n兩次訓練的 logits 最大差 {diff:.3g}")
    assert torch.allclose(a, b, atol=1e-4), (
        f"同 seed 訓練兩次（呼叫前的全域亂數狀態不同）得到不同的模型：logits 最大差 {diff:.3g}。"
        "所有隨機性（包括參數初始化）都要來自 seed")
