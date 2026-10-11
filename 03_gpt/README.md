# 階段 3：Transformer 與 GPT

教材、筆記與檢核在 `NOTES.md`。這份檔案說明怎麼開始、測試怎麼跑、失敗時看哪裡。
目前寫到第 3 段（LayerNorm、multi-head、FeedForward、Transformer block 與小 GPT），之後每開始一段就往下補。

## 分段

| 段 | 內容 | 學習者寫的檔案 |
|---|---|---|
| 1 | 資料（字元表、encode／decode、切分、batch）與 bigram baseline | `char_data.py`、`bigram_lm.py` |
| 2 | 加權平均的矩陣技巧、單一 self-attention head、單一 head 的語言模型 | `attention.py`、`single_head_lm.py` |
| 3 | LayerNorm（手寫）、multi-head、FeedForward、Transformer block、小 GPT 與訓練 | `attention.py`（接著寫）、`gpt.py` |
| 4 | 放大到影片的設定訓練、生成 | `gpt.py`（接著寫） |

## 分工

| 誰 | 寫什麼 |
|---|---|
| 學習者 | `char_data.py`：字元表、encode／decode、train／val 切分、`get_batch` |
| 學習者 | `bigram_lm.py`：bigram 模型、`generate`、`estimate_loss`、訓練 |
| 學習者 | `attention.py`：`causal_mean`、`Head`（第 3 段接著寫） |
| 學習者 | `single_head_lm.py`：單一 head 的語言模型、截斷 context 的 `generate`、訓練 |
| 學習者 | `attention.py`（第 3 段）：`Head` 加 dropout、`MultiHeadAttention`、`FeedForward`、`LayerNorm`、`Block` |
| 學習者 | `gpt.py`：`GPTLanguageModel`、`train_small_gpt` |
| Claude | `shakespeare_data.py`（資料下載與讀取）、`conftest.py`、`test_char_data.py`、`test_bigram_lm.py`、`test_attention.py`、`test_single_head_lm.py`、`test_transformer_block.py`、`test_gpt.py` |

依專案 CLAUDE.md §1.1，核心實作由學習者親手寫，Claude 不提供骨架或範例程式碼。

## 這個階段可以用哪些 torch 功能

和階段 2 不同，這個階段照影片用 `torch.nn`：

- **可以用**：`nn.Module`、`nn.Linear`、`nn.Embedding`、`nn.Dropout`、`F.softmax`、`F.cross_entropy`、
  `torch.optim.AdamW` 等
- **不能用**（要自己寫）：`F.scaled_dot_product_attention`、`nn.MultiheadAttention`、`nn.Transformer*`、
  `nn.LayerNorm`、`F.layer_norm`。測試會用它們當正確答案
- device：訓練時照 CLAUDE.md §3 的順序選（`"cuda"` → `"mps"` → `"cpu"`），計時前後呼叫對應的 `synchronize()`。
  單元測試都在 CPU 上跑

## 目錄

```
03_gpt/
├── README.md
├── NOTES.md
├── char_data.py          # ← 學習者寫（第 1 段）
├── bigram_lm.py          # ← 學習者寫（第 1 段）
├── attention.py          # ← 學習者寫（第 2 段起）
├── single_head_lm.py     # ← 學習者寫（第 2 段）
├── gpt.py                # ← 學習者寫（第 3 段起）
├── shakespeare_data.py   # 下載 tiny shakespeare、load_text()
├── conftest.py           # 讓測試能 import 這個資料夾的模組；提供 text fixture；--run-train 選項
├── test_char_data.py     # 第 1 段：char_data 的測試
├── test_bigram_lm.py     # 第 1 段：bigram_lm 的測試
├── test_attention.py     # 第 2 段：attention 的測試
├── test_single_head_lm.py  # 第 2 段：single_head_lm 的測試
├── test_transformer_block.py  # 第 3 段：attention.py 新增部分的測試
└── test_gpt.py           # 第 3 段：gpt 的測試
```

## 資料

```bash
uv run python 03_gpt/shakespeare_data.py
```

會把 tiny shakespeare（1115394 個字元、65 種字元）下載到 repo 根目錄的 `data/tinyshakespeare.txt`，並核對
sha256。檔案已存在且內容正確就不會重新下載。`data/` 不進 git。

在自己的程式裡讀資料：

```python
from shakespeare_data import load_text
text = load_text()   # str，整份文字
```

從 repo 根目錄用 `uv run python 03_gpt/xxx.py` 執行時，`03_gpt/` 會在 `sys.path` 上，可以直接
`import shakespeare_data`、`import char_data`。

## 第 1 段：資料與 bigram baseline

### 這一段要做的事

影片的前半：把文字變成整數序列、切成 train／val、隨機抽出 `(x, y)` 的 batch，再用最簡單的語言模型（bigram：
只看目前這一個字元，預測下一個字元）把訓練、評估、生成的流程走一遍。之後的 GPT 會沿用這一套資料處理與
model 介面（`forward`、`generate`、`estimate_loss`）。

### 介面約定

檔案：`03_gpt/char_data.py`、`03_gpt/bigram_lm.py`。測試會
`from char_data import build_vocab, encode, decode, split_data, get_batch`、
`from bigram_lm import BigramLanguageModel, estimate_loss`，訓練測試另外會 `from bigram_lm import train_bigram`。
名稱、參數順序或回傳型別不同，測試就會失敗。

#### 共通約定

- **所有隨機性只能來自傳入的 `generator`／`seed`**：同一個 seed 要得到完全相同的結果，不受全域亂數狀態影響
  （影片用的是全域亂數，這裡不同）。測試傳入的都是 CPU 的 `torch.Generator`
- **不能改動輸入的 tensor**
- **import 時不能跑重的工作**：訓練、印結果這類程式放在 `if __name__ == "__main__":` 底下

#### `build_vocab(text) -> tuple[dict[str, int], dict[int, str]]`（`char_data.py`）

- 輸入：`text`，任意 `str`
- 字元表：`text` 裡出現過的所有不同字元，用 Python 的 `sorted` 排序（依 Unicode 碼位），依序編號 0, 1, 2, …
  - 例：`"hello world"` → `' '` 是 0、`'d'` 是 1、`'e'` 是 2、`'h'` 是 3、`'l'` 是 4、`'o'` 是 5、`'r'` 是 6、`'w'` 是 7
  - 整份 tiny shakespeare：65 個字元，`'\n'` 是 0、`' '` 是 1、`'A'` 是 13、`'a'` 是 39、`'z'` 是 64
- 回傳 `(stoi, itos)`：
  - `stoi`：鍵是單一字元的 `str`，值是編號（Python `int`）
  - `itos`：`stoi` 反過來，鍵是編號（Python `int`），值是單一字元的 `str`

#### `encode(s, stoi) -> list[int]`、`decode(ids, itos) -> str`（`char_data.py`）

- `encode`：把字串的每個字元換成編號，回傳 Python 的 `list`，元素是 Python `int`（不是 tensor）。空字串 → `[]`
- `decode`：`ids` 是編號的 `list`（測試也會傳 `tensor.tolist()` 的結果），回傳接起來的 `str`。`[]` → `""`
- 整份資料：`decode(encode(text, stoi), itos) == text`

#### `split_data(data, frac=0.9) -> tuple[torch.Tensor, torch.Tensor]`（`char_data.py`）

- 輸入：`data` 是 1 維的 int64 tensor（整份文字 encode 之後的序列）；`frac` 是 train 的比例，預設 0.9。
  測試會省略 `frac`，也會用位置參數傳入（`split_data(data, 0.5)`）
- 回傳 `(train, val)`：`n = int(frac × len(data))`（**無條件捨去**，不是四捨五入），`train` 是前 `n` 個，`val` 是
  其餘的。**不打亂順序**（文字的前 90% 當 train、後 10% 當 val，和影片一樣）
- 例：長度 15、`frac = 0.9` → `n = int(13.5) = 13`

#### `get_batch(data, block_size, batch_size, generator) -> tuple[torch.Tensor, torch.Tensor]`（`char_data.py`）

- 輸入：`data` 是 1 維的 int64 tensor（測試傳 CPU tensor）；`block_size`、`batch_size` 是正整數；`generator` 見共通約定
- 隨機選 `batch_size` 個起點 `i`（可以重複），每個起點從 `0..len(data) − block_size − 1` 均勻抽出：
  **這個範圍裡的每個起點都要有機會被抽到**，範圍外的不能出現。`len(data) = block_size + 1` 時只有起點 0
- 回傳 `(x, y)`，形狀都是 `(batch_size, block_size)`、int64：
  - `x[b] = data[i : i + block_size]`
  - `y[b] = data[i + 1 : i + block_size + 1]`（`x` 往後移一格，也就是每個位置的「下一個字元」）
  - 放的是 `data` 裡的**值**，不是 index

#### `BigramLanguageModel(vocab_size)`（`bigram_lm.py`）

- 繼承 `torch.nn.Module`；建構子只有 `vocab_size` 一個參數（測試會寫子類別並用 `子類別(vocab_size)` 建立）
- 參數的初始化可以用全域亂數（例如 `nn.Embedding` 的預設初始化）；測試建立模型前會自己設 `torch.manual_seed`
- 要有參數，而且都 `requires_grad`；`loss.backward()` 之後每個參數都要有梯度
- **它是 bigram**：位置 `t` 的 logits 只由 `idx[:, t]` 這一個 token 決定，和它在哪個位置、前面是什麼都無關

#### `model(idx, targets=None) -> tuple[logits, loss]`（`BigramLanguageModel.forward`）

- `idx`：`(B, T)` 的 int64 tensor，值在 `0..vocab_size − 1`
- 沒有 `targets`：回傳 `(logits, None)`，`logits` 形狀 `(B, T, vocab_size)`
- 有 `targets`（`(B, T)` 的 int64，和 `idx` 同形狀）：
  - `loss` 是 0 維 tensor（可以 `.backward()`），等於**全部 `B × T` 個位置**的 cross-entropy 平均（自然對數），
    和 `F.cross_entropy(logits.view(B*T, V), targets.view(B*T))` 相同
  - `logits` 可以是 `(B, T, vocab_size)`，也可以照影片 reshape 成 `(B·T, vocab_size)`，兩種都接受；數值要和沒有
    `targets` 時相同
- `T` 沒有上限：訓練測試會把整份 val（11 萬多個 token）當成 `B = 1` 的一條序列丟進來算 loss

#### `model.generate(idx, max_new_tokens, generator) -> torch.Tensor`

- `idx`：`(B, T)` 的 int64，`T ≥ 1`；測試的 `B` 最大是 20000
- 重複 `max_new_tokens` 次：
  1. **透過 `self(...)` 呼叫 forward 取得 logits**（測試會用覆寫 `forward` 的子類別檢查 `generate`，所以不能繞過
     forward 直接讀參數）。可以把整段序列丟進去，也可以只丟最後一個 token
  2. 取**最後一個位置**的 logits，做 softmax 得到機率
  3. 依這個機率為每一列抽一個 token（是抽樣，不是取最大值），所有隨機性只能來自 `generator`
  4. 接在序列後面，下一步把接長的序列再丟回 forward
- 回傳 `(B, T + max_new_tokens)` 的 int64：前 `T` 欄就是輸入，後面是依序生成的 token。不能改動傳入的 `idx`
- 機率是 0 的 token 不能被抽到

#### `estimate_loss(model, data, block_size, batch_size, eval_iters, generator) -> float`（`bigram_lm.py`）

- 估計 `model` 在 `data` 上的 loss（影片的 `estimate_loss`，但一次只算一份資料；train、val 各呼叫一次）
- 做法：
  1. 切到 eval 模式（`model.eval()`），整個過程不追蹤梯度
  2. 重複 `eval_iters` 次：用 **`char_data.get_batch(data, block_size, batch_size, generator)`** 抽一批，呼叫
     `model(x, y)` 取得 `(logits, loss)`。`generator` 只給 `get_batch` 用，依序抽，不能拿去做別的事
  3. 回傳這 `eval_iters` 個 loss 的平均，型別是 Python `float`
  4. 結束前切回 train 模式（`model.train()`）
- `model` 可以是任何符合 `model(idx, targets) -> (logits, loss)` 的 `nn.Module`：測試傳入的 model 回傳的 `logits`
  是 `None`，所以只能用 `loss`，不要碰 `logits`

#### `train_bigram(train_data, vocab_size, seed) -> BigramLanguageModel`（`bigram_lm.py`，測試預設跳過）

- 輸入：`train_data` 是 `split_data(整份資料)` 的 `train`（1 維 int64，CPU）；`vocab_size` 是 65；`seed` 是整數
- 只能用 `train_data` 訓練。模型大小以外的設定（`block_size`、`batch_size`、學習率、步數、optimizer、device）自己決定
- 所有隨機性只能來自 `seed`（例如在函式開頭 `torch.manual_seed(seed)`，`get_batch` 用 seed 建的 `torch.Generator`）
- 回傳訓練好的模型。在哪個 device 都可以，測試會 `.to("cpu")`、`.eval()` 之後再檢查
- **通過標準**：
  - 整份 val 的 loss（把 val 當成一條序列：`model(val[:-1][None], val[1:][None])`）≤
    **計數法 bigram 在 val 上的 NLL + 0.02**。計數法用 `train_data` 的 bigram 次數、每格加 1 smoothing，約 2.482，
    門檻約 2.502
  - 能用 `generate` 從 `[[0]]` 生成 200 個 token，`decode` 回文字
  - **同 seed 要得到相同的模型**：測試會先把全域亂數設成不同的值（`torch.manual_seed(1)`、`torch.manual_seed(2)`），
    各呼叫一次 `train_bigram(train_data, 65, 0)`，兩個模型對每個 token 的 logits 相差要 ≤ 1e-4。參數初始化也算隨機性
- 時間：影片的設定（`block_size` 8、`batch_size` 32、AdamW）在 CPU 上幾秒內就能過門檻。訓練測試會呼叫 `train_bigram` 兩次。訓練結果照 CLAUDE.md §4
  記錄在 NOTES.md 的實驗紀錄

### 建議順序

每一步都有對應的測試，寫完一個就跑一次：

1. `build_vocab` → `test_build_vocab_*`
2. `encode`、`decode` → `test_encode_decode_*`
3. `split_data` → `test_split_data`
4. `get_batch` → `test_get_batch_*`
5. `BigramLanguageModel` 的 forward → `test_forward_*`、`test_backward`
6. `generate` → `test_generate_*`
7. `estimate_loss` → `test_estimate_loss`
8. `train_bigram` → `test_train_bigram`（要加 `--run-train`）

### 跑測試

```bash
uv run pytest 03_gpt/test_char_data.py                                   # char_data 的測試
uv run pytest 03_gpt/test_bigram_lm.py                                   # bigram_lm 的測試（訓練測試跳過）
uv run pytest 03_gpt/test_char_data.py -k get_batch                      # 只跑 get_batch 的測試
uv run pytest 03_gpt/test_bigram_lm.py --run-train -k train_bigram -s    # 只跑訓練測試；-s 會印出 loss 與生成的文字
```

`char_data.py` 不存在時會看到 `ModuleNotFoundError: No module named 'char_data'`。`test_char_data.py` 一開始就
import 5 個函式，`test_bigram_lm.py` 一開始就 import `BigramLanguageModel`、`estimate_loss` 與 `char_data` 的函式：
還沒寫到的 class 先寫成空的 class，函式先放 `raise NotImplementedError`。`train_bigram` 只在訓練測試裡才 import。

只要有一個測試檔收集失敗，`uv run pytest 03_gpt` 整個會在收集階段中斷，這時指定檔案跑。

每項測試檢查什麼、門檻怎麼來，寫在測試檔開頭（裡面有測資，實作時不用看）。

### 失敗時先看哪裡

- **`test_build_vocab_*` 編號不對**：字元表有沒有排序？
- **`test_split_data[15-0.9]` 失敗、其他兩個通過**：`n` 是無條件捨去還是四捨五入？
- **`test_get_batch_offsets` 說少了最後一個起點**：起點的上限差一。最後一個合法起點是 `len(data) − block_size − 1`，
  這時 `y` 剛好用到 `data` 的最後一個元素。`torch.randint(high, ...)` 抽的範圍不含 `high`
- **`test_get_batch_values` 說「y 應該是 x 往後移一格」**：`y` 的切片範圍
- **`test_get_batch_generator` 同 seed 結果不同**：`torch.randint` 沒有傳 `generator=`，用到全域亂數
- **`test_forward_loss` 的 loss 不對**：是不是所有 `B × T` 個位置一起平均？logits 和 targets reshape 的方式要對應
- **`test_generate_feeds_back` 失敗**：用的是不是最後一個位置的 logits？抽到的 token 有沒有接上去、下一步再丟回 forward？
  有沒有透過 `self(...)` 呼叫 forward？
- **`test_generate_samples_softmax` 比例不對**：logits 要先 softmax 變成機率再抽樣，不是取 argmax
- **`test_generate_generator` 不同 seed 結果相同**：可能用了 argmax；同 seed 結果不同：`torch.multinomial` 沒有傳 `generator=`
- **`test_estimate_loss` 說 eval 模式／梯度／train 模式**：照約定的 1、4 步；不追蹤梯度可以用 `torch.no_grad()`
- **`test_estimate_loss` 數值不對**：是不是每一批都用 `char_data.get_batch` 和傳入的 `generator`、依序抽 `eval_iters` 批？
- **MPS 上 `generate` 報錯**：MPS 上的 tensor 不能搭配 CPU 的 `torch.Generator` 做 `multinomial`。
  測試在 CPU 上跑 `generate`，自己在 MPS 上生成時要注意
- **`test_train_bigram_same_seed` 失敗**：`nn.Embedding`、`nn.Linear` 建立時用全域亂數初始化；`train_bigram` 有沒有在建立
  模型之前，讓這些亂數也由 `seed` 決定？
- **訓練測試的 val loss 沒過門檻**：先確認其他測試都通過；再看訓練後的 train loss，也接近計數法的值嗎？

## 第 2 段：矩陣技巧與單一 self-attention head

### 這一段要做的事

影片的中段：

1. **矩陣技巧**：每個位置拿到「自己和前面所有位置」的平均。影片寫了三個版本（for 迴圈、`tril` 矩陣乘法、
   `masked_fill(-inf)` + softmax），測試只檢查結果，三個版本都可以自己練，交一個函式 `causal_mean`
2. **單一 head**：把「平均」換成由資料決定的加權平均：每個位置發出 query、key、value，權重 = softmax(q·k / √head_size)，
   而且只能看自己和前面的位置（causal mask）
3. **單一 head 的語言模型**：token embedding 加上 position embedding → `Head` → `lm_head`。因為 position embedding 只有
   `block_size` 個位置，`generate` 時要把 context 截斷到最後 `block_size` 個 token

dropout 照影片的順序，第 3 段才加。

### 介面約定

檔案：`03_gpt/attention.py`、`03_gpt/single_head_lm.py`。測試會 `from attention import causal_mean, Head`、
`from single_head_lm import SingleHeadLM`，訓練測試另外會 `from single_head_lm import train_single_head`。訓練測試也會用
第 1 段 `char_data.py` 的 `build_vocab`、`encode`、`decode`、`split_data`。

共通約定同第 1 段：隨機性只能來自傳入的 `generator`／`seed`、不能改動輸入、import 時不能跑重的工作。這一段的
`nn.Module` 參數初始化可以用全域亂數（測試建立模型前會自己設 `torch.manual_seed`）。

#### `causal_mean(x) -> torch.Tensor`（`attention.py`）

- 輸入：`x` 是 `(B, T, C)` 的浮點數 tensor，`T ≥ 1`（測試有 `T = 1`）
- 輸出：形狀同 `x`，`out[b, t] = x[b, 0], x[b, 1], …, x[b, t]` 這 `t + 1` 個向量的平均（含 `t` 自己，不含 `t` 之後的）
- 例：`x[0, :, 0] = [1, 3, 5]` → `out[0, :, 0] = [1, 2, 3]`

#### `Head(n_embd, head_size, block_size)`（`attention.py`）

- 繼承 `nn.Module`
- 屬性（測試會直接用它們算正確答案）：
  - `head.key`、`head.query`、`head.value`：都是 `nn.Linear(n_embd, head_size, bias=False)`
- 要訓練的參數**剛好**是這三個 Linear 的 weight。causal mask 不是參數：要存的話用 `register_buffer`（`.to(device)` 時
  才會跟著移動），或每次 forward 現做
- `head(x)`：
  - `x` 是 `(B, T, n_embd)`，`1 ≤ T ≤ block_size`（`T` 可以小於 `block_size`；大於的情況不會被測）
  - `q = query(x)`、`k = key(x)`、`v = value(x)`，形狀都是 `(B, T, head_size)`
  - 權重 `wei[b, t, s] = q[b, t] · k[b, s] / √head_size`。**縮放用 `head_size`**（論文的 1/√d_k），不是 `n_embd`
    （影片 gpt.py 寫成 `C**-0.5`，這裡不同）。測試的 `n_embd` 和 `head_size` 不同
  - causal mask：`s > t` 的位置在 softmax 前設成 `-inf`，softmax 之後就是 0
  - softmax 沿著最後一維（對每個 `t`，所有 `s` 的權重加總為 1）
  - 輸出 `wei @ v`，形狀 `(B, T, head_size)`
  - 和 `F.scaled_dot_product_attention(q, k, v, is_causal=True)` 相同，測試用它當正確答案（你不能用它）
- 梯度：對 `x` 和三個 weight 的梯度要和上面的算法相同（用一般的 torch 運算寫，autograd 自然會對）

#### `SingleHeadLM(vocab_size, n_embd, block_size)`（`single_head_lm.py`）

- 繼承 `nn.Module`；建構子就是這三個參數（測試會寫子類別並用 `子類別(vocab_size, n_embd, block_size)` 建立）
- `model.block_size`：建構時傳入的 `block_size`
- 結構（影片的做法）：token embedding `(vocab_size, n_embd)` 加上 position embedding `(block_size, n_embd)` →
  `Head(n_embd, head_size=n_embd, block_size)` → `nn.Linear(n_embd, vocab_size)`。測試不檢查內部屬性名稱，但檢查下面的行為
- 每個參數都要用得到：`loss.backward()` 之後每個參數都要有梯度
- `model(idx, targets=None) -> (logits, loss)`：
  - `idx` 是 `(B, T)` 的 int64，`1 ≤ T ≤ block_size`
  - 其餘同第 1 段：沒有 `targets` 時 logits 是 `(B, T, vocab_size)`、loss 是 `None`；有 `targets` 時 loss 是全部 `B × T`
    個位置的 cross-entropy 平均，logits 可以是 `(B, T, V)` 或 `(B·T, V)`
- 行為：
  - **causal**：位置 `t` 的 logits 只由 `idx[:, :t+1]` 決定
  - **知道位置**：輸入全是同一個 token 時，不同位置的 logits 不同
  - **看得到 context**：改動前面的 token，後面位置的 logits 會變（不再是 bigram）

#### `model.generate(idx, max_new_tokens, generator) -> torch.Tensor`

- 同第 1 段的 `generate`（每一步透過 `self(...)` 呼叫 forward、取最後一個位置、softmax、用 `generator` 抽樣、接回去；
  回傳 `(B, T + max_new_tokens)`、前 `T` 欄是輸入、不改動輸入），另外：
- **每一步丟進 forward 的，是目前序列的最後 `min(目前長度, block_size)` 個 token**（不多也不少）。測試會用覆寫 `forward`
  的子類別記錄每一步收到的 `idx`
- 輸入本身比 `block_size` 長、或生成超過 `block_size` 個 token 都要能跑

#### `train_single_head(train_data, vocab_size, seed) -> SingleHeadLM`（`single_head_lm.py`，測試預設跳過）

- 同第 1 段的 `train_bigram`：只能用 `train_data` 訓練；`n_embd`、`block_size`、`batch_size`、學習率、步數、optimizer、
  device 自己決定；所有隨機性（包括參數初始化）只能來自 `seed`；回傳的模型在哪個 device 都可以（測試會 `.to("cpu")`、
  `.eval()`）。可以用第 1 段的 `estimate_loss` 觀察訓練
- **通過標準**：
  - val loss ≤ **計數法 bigram 在 val 上的 NLL − 0.05**（約 2.482 − 0.05 ≈ 2.432）。val loss 的算法：把 val 切成不重疊、
    長度 `model.block_size` 的段落（最後不足一段的丟掉），每段當成一列，用你的 forward 算全部位置的 loss
  - 能用 `generate` 從 `[[0]]` 生成 200 個 token（會超過 `block_size`），`decode` 回文字
  - 同 seed 要得到相同的模型（測試方式同第 1 段）
- 時間：影片的設定（`n_embd` 32、`block_size` 8、`batch_size` 32、AdamW lr 1e-3、5000 步）在 CPU 上幾秒就能過門檻。
  訓練測試會呼叫 `train_single_head` 兩次。訓練結果照 CLAUDE.md §4 記錄在 NOTES.md 的實驗紀錄

### 建議順序

1. `causal_mean` → `test_causal_mean`（三個版本都可以寫寫看，交其中一個）
2. `Head` → `test_head_*`
3. `SingleHeadLM` 的 forward → `test_forward_*`、`test_causal`、`test_uses_*`、`test_backward`
4. `generate` → `test_generate_*`
5. `train_single_head` → `test_train_single_head*`（要加 `--run-train`）

### 跑測試

```bash
uv run pytest 03_gpt/test_attention.py                                          # causal_mean 與 Head
uv run pytest 03_gpt/test_single_head_lm.py                                     # 模型與 generate（訓練測試跳過）
uv run pytest 03_gpt/test_single_head_lm.py --run-train -k train_single -s      # 只跑訓練測試；-s 會印出 loss 與生成的文字
```

`attention.py` 不存在時兩個測試檔都會 `ModuleNotFoundError`（`single_head_lm.py` 通常會 import `Head`）。測試一開始就
import 的名稱：`test_attention.py` 是 `causal_mean`、`Head`；`test_single_head_lm.py` 是 `SingleHeadLM`。還沒寫到的先寫成
空的 class 或 `raise NotImplementedError`。`train_single_head` 只在訓練測試裡才 import。

每項測試檢查什麼、門檻怎麼來，寫在測試檔開頭（裡面有測資，實作時不用看）。

### 失敗時先看哪裡

- **`test_causal_mean` 在 T = 1 通過、其他失敗**：每個位置除以的是自己的 `t + 1`，還是整個 `T`？有沒有混進未來的位置？
- **`test_head_attributes` 說參數不只三個 weight**：mask 被註冊成 `nn.Parameter`，或 Linear 有 bias
- **`test_head_matches_sdpa` 在 T=1 通過、T=16／5 失敗**：先查縮放（`head_size` 還是 `n_embd`）、mask 有沒有套、softmax
  沿哪一維。T = 1 時 softmax 只有一項，這三種錯都看不出來
- **`test_head_matches_sdpa[T=5]` 報形狀錯誤、T=16 通過**：mask 要切成 `T × T`，不是整個 `block_size × block_size`
- **`test_head_causal` 失敗**：mask 要在 softmax **之前**把未來位置設成 `-inf`。softmax 之後才設成 0，每列的權重加總就不是 1
- **`test_uses_position` 失敗**：position embedding 有沒有加到 token embedding 上？
- **`test_uses_context` 失敗**：embedding 有沒有經過 `Head` 再接 `lm_head`？
- **`test_backward` 說某個參數沒有梯度**：那個參數建了但 forward 沒用到
- **`generate` 報 index out of range**：context 超過 `block_size`，position embedding 查不到。每一步要截斷
- **`test_generate_crops_context` 失敗**：丟進 forward 的要剛好是最後 `min(長度, block_size)` 個 token，不是整段、也不是只有最後一個
- **訓練測試的 val loss 沒過門檻**：先確認其他測試都通過；比較 train 和 val loss；和第 1 段 bigram 的 2.48 比，有沒有明顯進步？

## 第 3 段：LayerNorm、multi-head、FeedForward、Transformer block 與小 GPT

### 這一段要做的事

影片從 multi-head attention 到 LayerNorm、dropout 的部分。第 2 段的單一 head 只有一個「溝通」的管道，這一段把它
擴充成完整的 Transformer block，再疊成小 GPT。下面依實作的相依順序排列（被用到的先寫），和影片講的順序不同：

1. **LayerNorm 自己寫**：和第 2 階段的 BatchNorm1d 類似，但統計量是對**每個位置自己的那個向量**（最後一維）算，
   不需要 running stats，train 和 eval 的行為一樣
2. **dropout**：訓練時隨機把一部分值設成 0，減少過擬合
3. **multi-head**：幾個比較小的 head 並排，各自做 attention，輸出接起來後經過一個投影層 `proj`
4. **FeedForward**：每個位置各自通過一個 MLP（attention 是位置之間「溝通」，FeedForward 是每個位置自己「思考」）
5. **residual + LayerNorm**：網路變深之後訓練會變難。residual 讓梯度有一條直通的路；LayerNorm 把每個位置的向量正規化。
   用影片的 **pre-norm**：先 LayerNorm 再進 attention／FeedForward，結果加回原本的 `x`
6. **小 GPT**：token + position embedding → `n_layer` 個 Block → 最後一層 LayerNorm → `lm_head`，用影片中段的大小訓練

### 介面約定

檔案：`03_gpt/attention.py`（接著第 2 段寫）、`03_gpt/gpt.py`。測試會
`from attention import Head, MultiHeadAttention, FeedForward, LayerNorm, Block`、`from gpt import GPTLanguageModel`，
訓練測試另外會 `from gpt import train_small_gpt`。第 2 段的測試（`test_attention.py`、`test_single_head_lm.py`）要繼續通過。

共通約定同前兩段。另外：

- 這一段的模組都繼承 `nn.Module`，參數初始化與 dropout 可以用全域亂數（`nn.Dropout` 就是用全域亂數）；訓練函式要在
  開頭用 `seed` 設定全域亂數，讓參數初始化和 dropout 都由 `seed` 決定
- **dropout 的約定**（`Head`、`MultiHeadAttention`、`FeedForward`、`Block`、`GPTLanguageModel` 都適用）：
  - 參數名稱是 `dropout`（機率 p），預設 `0.0`
  - **eval 模式**（`.eval()`）：dropout 完全不起作用，結果等於沒有 dropout 的算法
  - **train 模式**、`p > 0`：輸出是隨機的（同樣的輸入呼叫兩次結果不同，也和 eval 不同）
  - `p = 0`：train 模式和 eval 模式相同
  - 影片放 dropout 的位置：`Head` 的 softmax 之後（attention 權重上）、`MultiHeadAttention` 的 `proj` 之後、
    `FeedForward` 的最後。測試只檢查上面的外部行為，不檢查位置
- 「要訓練的參數剛好是 …」：測試會比對 `module.parameters()`，所以不能多出別的參數（例如把 mask 註冊成參數）

#### `LayerNorm(dim, eps=1e-5)`（`attention.py`）

- 自己寫：不能繼承 `nn.LayerNorm`，forward 裡也不能呼叫 `F.layer_norm`、`torch.layer_norm`（測試會把它們換成會報錯的函式）
- 屬性：
  - `ln.weight`：形狀 `(dim,)` 的 `nn.Parameter`，初始全 1
  - `ln.bias`：形狀 `(dim,)` 的 `nn.Parameter`，初始全 0
  - `ln.eps`：建構時傳入的 `eps`
- 要訓練的參數剛好是 `weight`、`bias`；沒有 running stats
- `ln(x)`：`x` 是 `(…, dim)`（測試用 2 維和 3 維）。對**最後一維**，每個向量各自算 mean 和 **biased** variance（除以 `dim`），
  輸出 `(x − mean) / √(variance + eps) × weight + bias`，形狀同 `x`。`eps` 加在根號**裡面**
- 和 `nn.LayerNorm(dim, eps=eps)` 相同，測試用 `F.layer_norm` 當正確答案；train 和 eval 模式行為相同

#### `Head(n_embd, head_size, block_size, dropout=0.0)`（`attention.py`，擴充）

- 第 2 段的約定不變，只多了 `dropout`：套在 softmax 之後的 attention 權重上。`dropout = 0.0` 時和第 2 段完全相同

#### `MultiHeadAttention(n_embd, n_head, block_size, dropout=0.0)`（`attention.py`）

- `n_embd` 是 `n_head` 的倍數；每個 head 的大小 `head_size = n_embd // n_head`
- 屬性（測試會用它們算正確答案）：
  - `mha.heads`：`nn.ModuleList`，裡面依序是 `n_head` 個 `Head(n_embd, head_size, block_size, dropout)`
  - `mha.proj`：`nn.Linear(n_embd, n_embd)`（**有** bias）
- 要訓練的參數剛好是各 head 和 `proj` 的參數
- `mha(x)`：`x` 是 `(B, T, n_embd)`，`1 ≤ T ≤ block_size`。每個 head 各自算出 `(B, T, head_size)`，**依 `heads` 的順序**沿最後
  一維接成 `(B, T, n_embd)`，經過 `proj`（再 dropout），輸出 `(B, T, n_embd)`。causal（位置 `t` 看不到 `t` 之後）

#### `FeedForward(n_embd, dropout=0.0)`（`attention.py`）

- 屬性：`ffwd.fc1` 是 `nn.Linear(n_embd, 4 * n_embd)`、`ffwd.fc2` 是 `nn.Linear(4 * n_embd, n_embd)`，都**有** bias
- 要訓練的參數剛好是 `fc1`、`fc2` 的參數
- `ffwd(x)`：`x` 是 `(…, n_embd)`（測試用 3 維），輸出 `fc2(ReLU(fc1(x)))`（再 dropout），形狀同 `x`。內層 4 倍、ReLU 是論文
  和影片的做法

#### `Block(n_embd, n_head, block_size, dropout=0.0)`（`attention.py`）

- 屬性：
  - `blk.ln1`、`blk.ln2`：`attention.LayerNorm(n_embd)`（你自己的 LayerNorm）
  - `blk.sa`：`MultiHeadAttention(n_embd, n_head, block_size, dropout)`
  - `blk.ffwd`：`FeedForward(n_embd, dropout)`
- 要訓練的參數剛好是這四個子模組的參數
- `blk(x)`：`x` 是 `(B, T, n_embd)`，**pre-norm + residual**：
  1. `x = x + sa(ln1(x))`
  2. `x = x + ffwd(ln2(x))`
  
  輸出 `(B, T, n_embd)`。不能改動輸入（`x = x + …` 會建立新的 tensor；`x += …` 會改到輸入）

#### `GPTLanguageModel(vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.0)`（`gpt.py`）

- 繼承 `nn.Module`；建構子就是這些參數，依這個順序（測試會寫子類別並用 `子類別(vocab_size, n_embd, n_head, n_layer, block_size)` 建立）
- `model.block_size`：建構時傳入的 `block_size`
- 結構：token embedding `(vocab_size, n_embd)` 加上 position embedding `(block_size, n_embd)` → **剛好 `n_layer` 個**
  `attention.Block(n_embd, n_head, block_size, dropout)` → 最後一層 `attention.LayerNorm(n_embd)` → `nn.Linear(n_embd, vocab_size)`
  - 整個模型裡 `attention.LayerNorm` 剛好 `2 × n_layer + 1` 個（每個 Block 兩個，加上最後一層），不能有 `nn.LayerNorm`
  - `dropout` 要傳進每個 Block
  - 每個參數都要用得到（`loss.backward()` 之後都有梯度）
- `model(idx, targets=None)`、`model.generate(idx, max_new_tokens, generator)`：約定同第 2 段的 `SingleHeadLM`
  （`1 ≤ T ≤ block_size`；causal；知道位置；看得到 context；`generate` 每一步丟進 forward 的是最後 `min(長度, block_size)` 個 token）

#### `train_small_gpt(train_data, vocab_size, seed) -> GPTLanguageModel`（`gpt.py`，測試預設跳過）

- 同第 2 段的 `train_single_head`：只能用 `train_data`；模型大小與訓練超參數自己決定；所有隨機性（參數初始化、dropout、
  `get_batch`）只能來自 `seed`；回傳的模型在哪個 device 都可以（測試會 `.to("cpu")`、`.eval()`）
- **通過標準**：
  - val loss（算法同第 2 段：val 切成長度 `model.block_size` 的段落）≤ **計數法 bigram 的 val NLL − 0.35**（約 2.132）
  - 能用 `generate` 從 `[[0]]` 生成 200 個 token，`decode` 回文字
  - 同 seed 要得到相同的模型
- 時間：影片中段的設定（`n_embd` 32、`n_head` 4、`n_layer` 3、`block_size` 8、`batch_size` 32、AdamW lr 1e-3、5000 步）在 CPU 上
  約 30 秒就能過門檻。訓練測試會呼叫 `train_small_gpt` 兩次，所以單次建議控制在 1 分鐘內；預計超過 10 分鐘的訓練要先確認
  （CLAUDE.md §4）。訓練結果照 CLAUDE.md §4 記錄在 NOTES.md 的實驗紀錄

### 建議順序

1. `LayerNorm` → `test_layernorm_*`
2. `Head` 加 dropout → `test_head_dropout`；改完跑一次第 2 段的測試
3. `MultiHeadAttention` → `test_mha_*`
4. `FeedForward` → `test_ffwd_*`
5. `Block` → `test_block_*`
6. `GPTLanguageModel` → `test_gpt.py` 裡 train 以外的測試
7. `train_small_gpt` → `test_train_small_gpt*`（要加 `--run-train`）

### 跑測試

```bash
uv run pytest 03_gpt/test_transformer_block.py                          # LayerNorm、dropout、MHA、FeedForward、Block
uv run pytest 03_gpt/test_transformer_block.py -k layernorm             # 只跑 LayerNorm
uv run pytest 03_gpt/test_attention.py 03_gpt/test_single_head_lm.py    # 改了 Head 之後，確認第 2 段沒壞
uv run pytest 03_gpt/test_gpt.py                                        # 小 GPT（訓練測試跳過）
uv run pytest 03_gpt/test_gpt.py --run-train -k train_small -s          # 只跑訓練測試；-s 會印出 loss 與生成的文字
```

測試一開始就 import 的名稱：`test_transformer_block.py` 是 `Head`、`MultiHeadAttention`、`FeedForward`、`LayerNorm`、`Block`；
`test_gpt.py` 是 `attention` 的 `Block`、`LayerNorm` 和 `GPTLanguageModel`。還沒寫到的先寫成空的 class。`train_small_gpt` 只在
訓練測試裡才 import。

每項測試檢查什麼、門檻怎麼來，寫在測試檔開頭（裡面有測資，實作時不用看）。

### 失敗時先看哪裡

- **`test_layernorm_matches_torch` 差一點點**：variance 是 biased（除以 `dim`）還是 unbiased？
- **只有 `test_layernorm_custom_eps` 失敗**：`eps` 有沒有加在根號裡面？有沒有用傳入的 `eps`？
- **LayerNorm 的結果整個不對**：統計量要對最後一維（每個位置自己的向量）算，不是對 batch（那是 BatchNorm）
- **`test_*_dropout` 說「eval 模式下 dropout 不應該起作用」**：要用 `nn.Dropout`（它會看 `self.training`），或自己判斷模式；
  也可能是 dropout 以外的算法本身就不對，先看同一個元件的 `matches_reference` 有沒有過
- **`test_*_dropout` 說「train 模式…應該不同」**：dropout 沒有套上，或 `p` 沒有傳進子模組
- **`test_mha_matches_reference` 失敗**：各 head 的輸出是沿最後一維、依 `heads` 的順序接起來嗎？接起來之後有沒有經過 `proj`？
- **`test_*_attributes` 失敗**：看訊息裡是哪個屬性的形狀或 bias 不對
- **`test_block_matches_reference` 失敗**：是 pre-norm（先 LayerNorm 再進 sa／ffwd）嗎？residual 有沒有加回去？`ln1` 給 sa、
  `ln2` 給 ffwd，有沒有用反？
- **`test_block_backward` 報 `modified by an inplace operation`、`test_block_matches_reference` 也失敗**：residual 寫成了
  `x += …`，改到了輸入，autograd 需要的值也被覆蓋。改成 `x = x + …`
- **`test_structure` 說 LayerNorm 數量不對**：最後一層 LayerNorm 有沒有加？有沒有用到 `nn.LayerNorm`？
- **`test_backward` 說某個參數沒有梯度**：那個參數建了但 forward 沒用到（例如最後一層 LayerNorm、`proj`）
- **`test_dropout`（test_gpt.py）失敗**：`dropout` 有沒有傳進每個 Block
- **訓練測試的 val loss 沒過門檻**：先確認其他測試都通過；比較 train 和 val loss；和第 2 段單一 head 的結果比有沒有進步？
  小模型加 dropout 可能反而變差
