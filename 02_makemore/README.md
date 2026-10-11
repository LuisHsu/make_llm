# 階段 2：makemore

教材、筆記與檢核在 `NOTES.md`。這份檔案說明怎麼開始、測試怎麼跑、失敗時看哪裡。
五段都有介面約定與測試，依序做。

## 分工

| 誰 | 寫什麼 |
|---|---|
| 學習者 | `bigram.py`：字元表、計數、機率、NLL、取樣、神經網路版 bigram |
| 學習者 | `mlp.py`（及訓練用的程式）：資料集、切分、評估、取樣、MLP 模型與訓練 |
| 學習者 | `deep_mlp.py`：Linear、BatchNorm1d、Tanh、深層模型的初始化與訓練（第 5 段再讓 BatchNorm1d 支援 3 維輸入） |
| 學習者 | `backprop_ninja.py`：不用 autograd，手推每個變數的梯度，並用手推梯度訓練 |
| 學習者 | `wavenet.py`：Embedding、FlattenConsecutive、Sequential、階層式模型的初始化與訓練 |
| Claude | `names_data.py`（資料下載與讀取）、`test_bigram.py`、`test_mlp.py`、`test_deep_mlp.py`、`test_backprop_ninja.py`、`test_wavenet.py`、`golden/`、`conftest.py` |

依專案 CLAUDE.md §1.1，核心實作由學習者親手寫，Claude 不提供骨架或範例程式碼。

## 目錄

```
02_makemore/
├── README.md
├── NOTES.md
├── bigram.py            # ← 學習者寫（第 1 段）
├── mlp.py               # ← 學習者寫（第 2 段）
├── deep_mlp.py          # ← 學習者寫（第 3 段）
├── backprop_ninja.py    # ← 學習者寫（第 4 段）
├── wavenet.py           # ← 學習者寫（第 5 段）
├── names_data.py        # 下載 names.txt、load_words()
├── conftest.py          # 讓測試能 import 這個資料夾的模組；提供 words fixture；--run-train 選項
├── test_bigram.py       # 第 1 段的測試
├── test_mlp.py          # 第 2 段的測試
├── test_deep_mlp.py     # 第 3 段的測試
├── test_backprop_ninja.py  # 第 4 段的測試
├── test_wavenet.py      # 第 5 段的測試
└── golden/
    └── bigram.json      # 27×27 計數矩陣與 NLL 參考值
```

## 資料

```bash
uv run python 02_makemore/names_data.py
```

會把 `names.txt`（32033 個名字，全部是小寫 a–z）下載到 repo 根目錄的 `data/`，並核對
sha256。檔案已存在且內容正確就不會重新下載。`data/` 不進 git。

在自己的程式裡讀資料：

```python
from names_data import load_words
words = load_words()
```

從 repo 根目錄用 `uv run python 02_makemore/xxx.py` 執行時，`02_makemore/` 會在
`sys.path` 上，可以直接 `import names_data`。

## 第 1 段：bigram

### 介面約定

檔案：`02_makemore/bigram.py`。測試會 `from bigram import stoi, itos, count_bigrams, count_probs, nll, sample, train_nn`，
名稱、參數順序或回傳型別不同，測試就會失敗。

#### 共通約定

- **字元表**：共 27 個字元。`'.'` 的編號是 0，代表名字的開頭與結尾；`'a'`..`'z'` 依字母順序是 1..26
- **words**：`list[str]`，每個字串是一個名字，只含小寫 `a`–`z`，長度至少 1。可能有重複的名字，重複的要各算一次
- **bigram**：每個名字前後各補一個 `'.'` 之後，相鄰的兩個字元。例如 `"ab"` → `.a`、`ab`、`b.` 共 3 個；
  長度 L 的名字有 L + 1 個 bigram
- **27×27 矩陣的方向**：列（第一個 index）是前一個字元，行（第二個 index）是下一個字元，index 都是字元表的編號
- **機率矩陣 P**：27×27 的浮點數 `torch.Tensor`，`P[i, j]` = 前一個字元是 i 時，下一個字元是 j 的機率；每一列加總為 1
- **回傳的 tensor 一律放在 CPU 上**。中間計算可以用 MPS，但回傳前要 `.cpu()`：測試會拿回傳值和 CPU tensor
  比較，還會轉成 float64，而 MPS 不支援 float64，兩種情況都會直接報錯
- **import 時不能跑重的工作**：測試 import `bigram` 時，模組最上層的程式碼會被執行。訓練、印結果這類程式放在
  `if __name__ == "__main__":` 底下

#### `stoi`、`itos`（模組層級的 dict 變數，不是函式）

- `stoi: dict[str, int]`：鍵是單一字元的字串（`'.'`、`'a'`..`'z'`），值是它的編號。例如 `stoi['.'] == 0`、`stoi['a'] == 1`、`stoi['z'] == 26`
- `itos: dict[int, str]`：`stoi` 反過來，鍵是 0..26 的 `int`，值是單一字元的字串。例如 `itos[0] == '.'`
- 兩者都剛好 27 個項目，且對每個字元 `c`，`itos[stoi[c]] == c`

#### `count_bigrams(words) -> torch.Tensor`

- 輸入：`words`（見共通約定）
- 輸出：形狀 (27, 27) 的 `torch.Tensor`，`N[i, j]` = 所有名字的所有 bigram 裡，前一個字元是 i、下一個字元是 j
  的次數。dtype 不限（整數或浮點數都可以），但數值必須剛好是整數次數

#### `count_probs(N, smoothing=1) -> torch.Tensor`

- 輸入：
  - `N`：形狀 (27, 27) 的計數 tensor。可能是你的 `count_bigrams` 的輸出，也可能是 dtype 為 `torch.int64`
    的 tensor，兩種都要能處理
  - `smoothing`：非負整數，預設 1。測試會用位置參數傳入（`count_probs(N, 0)`），也會省略它（`count_probs(N)`）
- 輸出：形狀 (27, 27) 的機率矩陣 P（浮點數）。每一格先加上 `smoothing`，再把每一列除以該列的總和
- 不能改動傳入的 `N`：測試呼叫完之後還會繼續用同一個 `N`
- `smoothing = 0` 時，計數是 0 的格子機率要剛好是 0
- 同樣的輸入要得到完全相同的輸出
- 整列都是 0 且 `smoothing = 0` 的情況不會被測，行為自己決定

#### `nll(P, words) -> float`

- 輸入：機率矩陣 `P`、`words`
- 輸出：Python 內建的 `float`，不能是 tensor（0 維 tensor 也不行）
- 算法：把 `words` 裡**所有名字的所有 bigram** 放在一起，對每個 bigram 取 `-ln P[前一個, 下一個]`
  （自然對數），再除以 bigram 的總數。不是先算每個名字的平均再平均
- 測試用的 `P` 不一定來自 `count_probs`，也可能是手動建的 float32 tensor

#### `sample(P, n, generator) -> list[str]`

- 輸入：
  - `P`：機率矩陣
  - `n`：要產生幾個名字（正整數），測試最多會用到 3000
  - `generator`：已經設好 seed 的 `torch.Generator`（CPU）
- 輸出：長度 `n` 的 `list[str]`，依產生的順序排列
- 產生一個名字的規則：目前字元從 `'.'` 開始，依 `P` 中目前字元那一列的機率，取樣下一個字元；取到 `'.'`
  就結束。名字是途中取到的字元依序接起來，不含開頭和結尾的 `'.'`。第一步就取到 `'.'` 的話，就是空字串
- 所有隨機性都只能來自傳入的 `generator`：同一個 seed 的兩個 generator，要得到完全相同的 list。
  不能用全域的亂數狀態，也不能自己另外建 generator
- 速度：`n = 3000` 要在幾秒內跑完

#### `train_nn(words) -> torch.Tensor`

- 輸入：`words`，測試會傳入整份 `names.txt`（32033 個名字）
- 要做的事：訓練一個神經網路版的 bigram，輸入是前一個字元，輸出是下一個字元的機率分佈，用梯度下降訓練。
  網路結構、初始化、超參數、訓練步數都自己決定
- 輸出：形狀 (27, 27) 的機率矩陣。第 i 列 = 輸入字元 i 時，訓練好的網路輸出的機率分佈
  - 可以帶 `requires_grad`（測試會先 `.detach()`），dtype float32 或 float64 都可以，要在 CPU 上
  - 每一格 ≥ 0，每一列加總為 1（容許誤差 1e-4）
- 通過標準：用你的 `nll` 算這個 P 在同一份 `words` 上的 NLL，要落在下面兩個值之間
  - 下限：`count_probs(N, 0)` 的 NLL − 0.001（N 是整份資料的計數）
  - 上限：`count_probs(N, 1)` 的 NLL + 0.05
- 權重是隨機初始化的話，建議固定 seed，否則每次跑的結果不同
- 速度：測試沒有設時間上限，但整份測試跑一次會呼叫一次 `train_nn`，建議控制在一分鐘內

### 建議順序

每一步都有對應的測試，寫完一個就跑一次：

1. `stoi`、`itos` → `test_vocab`
2. `count_bigrams` → `test_count_bigrams_*`
3. `count_probs` → `test_count_probs_*`
4. `nll` → `test_nll_*`
5. `sample` → `test_sample_*`
6. `train_nn` → `test_train_nn_matches_counting`

### 跑測試

```bash
uv run pytest 02_makemore                  # 全部
uv run pytest 02_makemore -k count_bigrams # 名稱含 count_bigrams 的測試
uv run pytest 02_makemore -x               # 第一個失敗就停
```

`bigram.py` 不存在時，會看到 `ModuleNotFoundError: No module named 'bigram'`。
只寫了一部分時，**整個檔案的測試都會因為 import 失敗而無法收集**，因為測試一開始就 import
全部 7 個名稱。所以還沒寫到的函式可以先放一個 `raise NotImplementedError`，讓已寫好的部分能跑。

每項測試檢查什麼、golden 值從哪裡來，寫在 `test_bigram.py` 開頭（裡面有測資，實作時不用看）。

### 失敗時先看哪裡

- **pytest 卡住不動**：幾乎一定是 `sample` 停不下來，先檢查「取到 `'.'` 就結束」那段。
  取樣測試的 P 是用 golden 計數建的，不是用你的 `count_bigrams`，所以計數寫錯不會讓取樣卡住。
- **取樣測試失敗**：其中兩項（同 seed、平均長度）是把 golden 計數交給**你的 `count_probs`** 算出 P 再取樣，
  所以先確定 `count_probs` 的測試都通過，再查 `sample`。
- **`nll` 和 `train_nn` 的測試會用到前面的函式**：`test_nll_matches_golden` 用你的 `count_bigrams`、
  `count_probs`；`test_train_nn_matches_counting` 用你的 `nll`。前面的測試沒過時，這幾項的失敗不一定是它們本身的問題。
- **整份資料的比對失敗，但小例子的測試通過**：小例子只檢查幾個格子，整份資料的比對會檢查整個矩陣。
  可以自己再造幾個能手算的小例子（重複的名字、長度 1 的名字、同一個字母連續出現），印出結果和手算對照，
  特別注意共通約定裡「bigram」和「矩陣方向」那兩條。
- **`test_train_nn_matches_counting` 的 NLL 太高**：代表還沒訓練到位，這是超參數或訓練步驟的
  問題；如果比計數法的最小值還低，就是 NLL 或機率的算法有錯。

## 第 2 段：MLP 語言模型

### 這一段要做的事

bigram 只看前 1 個字元。這一段改成看前 `block_size` 個字元：每個字元先查 embedding 表變成向量，
接起來經過隱藏層，輸出下一個字元的機率。另外要把資料切成 train／dev／test，在沒看過的資料上評估。

### 介面約定

檔案：`02_makemore/mlp.py`。測試會 `from mlp import build_dataset, split_words, evaluate, sample`，訓練測試另外會 `from mlp import train_mlp`，
名稱、參數順序或回傳型別不同，測試就會失敗。

#### 共通約定

- **字元表**：和第 1 段相同（`'.'` 是 0，`'a'`..`'z'` 是 1..26）。可以 `from bigram import stoi, itos` 重用，
  也可以自己再定義一次，編號必須一樣
- **words**：同第 1 段（`list[str]`，只含小寫 `a`–`z`，長度至少 1，可能重複）
- **context**：長度 `block_size` 的字元編號序列，代表「下一個字元之前的 `block_size` 個字元」。名字開頭
  不足 `block_size` 個字元時，前面用 `'.'`（0）補滿
- **model**：任何可以這樣呼叫的物件 —— `model(X)`，`X` 是形狀 `(n, block_size)`、dtype `torch.int64`
  的 context，回傳形狀 `(n, 27)` 的浮點數 **logits**（還沒經過 softmax）。第 i 列對應 `X[i]` 這個 context
  之後的下一個字元。可以是函式、有 `__call__` 的類別，或 `torch.nn.Module`。測試會用自己做的假 model
  來測 `evaluate` 和 `sample`，所以這兩個函式只能透過 `model(X)` 使用模型，不能讀模型內部的參數
- **回傳的 tensor 一律放在 CPU 上**；import 時不能跑重的工作（同第 1 段）

#### `build_dataset(words, block_size) -> tuple[torch.Tensor, torch.Tensor]`

- 輸入：`words`、`block_size`（正整數，測試會用 1、3、5）
- 輸出：`(X, Y)`
  - `X`：形狀 `(m, block_size)`，dtype `torch.int64`；每一列是一個 context
  - `Y`：形狀 `(m,)`，dtype `torch.int64`；`Y[k]` 是 `X[k]` 之後真正的下一個字元
- 每個名字產生 `len(名字) + 1` 筆：從全部是 `'.'` 的 context 開始，依序預測名字的每個字元，最後一筆預測結尾的
  `'.'`。每預測完一個字元，context 就往左移一格，把剛才的字元接在最右邊
- 例：`build_dataset(["emma"], 3)` 產生 5 筆，依序是

  | context | 下一個字元 |
  |---|---|
  | `...` | `e` |
  | `..e` | `m` |
  | `.em` | `m` |
  | `emm` | `a` |
  | `mma` | `.` |

- 順序：依 `words` 的順序，每個名字內依位置順序
- `block_size = 1` 時，`(X[:, 0], Y)` 剛好就是第 1 段的所有 bigram

#### `split_words(words, seed) -> tuple[list[str], list[str], list[str]]`

- 輸入：`words`、`seed`（整數）
- 輸出：`(train, dev, test)` 三個 `list[str]`
- 以**名字**為單位切，不是以 `build_dataset` 產生的樣本為單位切：同一個名字的所有樣本必須落在同一份裡，
  否則 dev／test 會含有訓練時看過的名字片段
- **不去重**：直接切原本的 `words`（影片的做法）。`names.txt` 有重複的名字（32033 個名字裡只有 29494 個
  不重複），所以同一個名字仍可能同時出現在 train 和 dev，dev loss 會因此稍微偏樂觀
- 先依 `seed` 把名字隨機打亂，再依序切成 80%／10%／10%。設 `n = len(words)`：
  - `len(train) == int(0.8 * n)`
  - `len(dev) == int(0.9 * n) - int(0.8 * n)`
  - `len(test) == n - int(0.9 * n)`
- 三份合起來剛好是原本的 `words`（當成多重集合比較：重複的名字各算一次）
- 同一個 `seed` 要得到完全相同的結果；打亂的隨機性只能來自 `seed`，不能用全域的亂數狀態
- 不能改動傳入的 `words`

#### `evaluate(model, X, Y) -> float`

- 輸入：`model`（見共通約定）、`X`、`Y`（`build_dataset` 的輸出格式）
- 輸出：Python 內建的 `float`：`model(X)` 對 `Y` 的平均 negative log-likelihood（自然對數，除以樣本數）。
  也就是第 1 段 `nll` 的推廣
- 不能改動 `X`、`Y`；評估時不需要計算梯度
- `X` 可能有 20 萬列以上，要能一次處理完

#### `sample(model, block_size, n, generator) -> list[str]`

- 輸入：`model`、`block_size`（`model` 預期的 context 長度）、`n`（正整數）、`generator`（CPU 的 `torch.Generator`）
- 輸出：長度 `n` 的 `list[str]`
- 產生一個名字：context 從全部是 `'.'` 開始；把 context 做成形狀 `(1, block_size)` 的 `X` 丟給 `model`，
  對輸出的 logits 做 softmax 得到機率，依機率取樣下一個字元；取到 `'.'` 就結束。沒有結束的話，就把取到的
  字元接到 context 最右邊（最左邊的丟掉），再取樣下一個。名字不含 `'.'`，可能是空字串
- 隨機性只能來自 `generator`（同第 1 段）
- 測試最多會用到 `n = 3000`，要在幾秒內跑完

#### `train_mlp(train_words, seed) -> tuple[model, int]`（測試預設跳過）

MLP 的結構（embedding 維度、隱藏層大小、初始化）和訓練（`block_size`、batch size、學習率、步數、學習率衰減）
都由你決定，不寫進介面。測試只透過這個函式拿到訓練好的模型：

- 輸入：
  - `train_words`：`list[str]`，訓練用的名字（測試傳入 `split_words(整份資料, 42)` 的 train）
  - `seed`：整數，訓練的所有隨機性（初始化、mini-batch 抽樣）都只能來自它；同一個 seed 要得到同樣的模型
- 輸出：`(model, block_size)`
  - `model`：符合共通約定的 `model(X)`。**要能直接吃 CPU 上的 `X`、回傳 CPU 上的 logits**：用 MPS 訓練的話，
    回傳前要把模型搬回 CPU
  - `block_size`：Python `int`，這個模型用的 context 長度。測試用它對 dev 呼叫你的 `build_dataset`
- 只能用 `train_words` 訓練，不能讀 dev／test 或整份資料
- 通過標準：用你的 `evaluate` 算出的 dev NLL ≤ **bigram 在 dev 上的 NLL − 0.2**
  - bigram 的 NLL 由測試自己算：第 1 段的計數法（平滑 1）只在同一份 train 上計數，再在 dev 上算平均 NLL。
    數值依 `split_words` 的實作而定：README 只規定依 seed 打亂，打亂方法不同，同一個 seed 切出的名字就不同
    （例：split seed 42 時，定門檻用的參考 split 約 2.453，以 `torch.randperm` 打亂的 split 約 2.465）
  - 測試也會檢查 `model(X)` 的形狀、裝置，以及能不能交給 `sample` 產生名字
- 時間：測試沒有設上限，但每次加 `--run-train` 都會完整訓練一次。建議先讓它在幾十秒到幾分鐘內跑完；預計
  超過 10 分鐘的訓練要先確認（CLAUDE.md §4）

訓練結果照 CLAUDE.md §4 記錄在 NOTES.md 的實驗紀錄：git commit、超參數、seed、資料量、最終 train／dev loss、
tokens/sec、峰值記憶體；沒量的寫「未測」。

### 建議順序

1. `build_dataset` → `test_build_dataset_*`
2. `split_words` → `test_split_*`
3. `evaluate` → `test_evaluate_*`
4. `sample` → `test_sample_*`
5. `train_mlp` → `test_train_mlp_beats_bigram`（要加 `--run-train`）：訓練好之後也可以自己用 `evaluate` 算 train／dev loss、用 `sample` 產生名字

### 跑測試

```bash
uv run pytest 02_makemore/test_mlp.py                              # 第 2 段（訓練測試跳過）
uv run pytest 02_makemore/test_mlp.py -k split                     # 名稱含 split 的測試
uv run pytest 02_makemore/test_mlp.py --run-train -k train_mlp -s  # 只跑訓練測試，-s 會印出 dev NLL 與門檻
```

**訓練測試預設跳過**：沒加 `--run-train` 時，`test_train_mlp_beats_bigram` 顯示為 `s`（skipped）。`--run-train`
只在指定 `02_makemore` 或其中檔案的路徑時才認得；在 repo 根目錄不指定路徑直接加，pytest 會回報
`unrecognized arguments: --run-train`。另外，在 repo 根目錄不指定路徑跑 `uv run pytest`，會連 `build-Debug/` 裡
googletest 自帶的 Python 檔也一起收集而出錯（這是原本就有的狀況），所以一律指定路徑。

`mlp.py` 不存在時，會看到 `ModuleNotFoundError: No module named 'mlp'`。和第 1 段一樣，測試一開始就 import
`build_dataset`、`split_words`、`evaluate`、`sample` 這 4 個名稱，還沒寫到的函式先放 `raise NotImplementedError`。
`train_mlp` 只在訓練測試裡才 import，還沒寫也不影響其他測試。

注意：只要 `test_mlp.py` 收集失敗（例如 `mlp.py` 還不存在），`uv run pytest 02_makemore` 整個會在收集階段中斷，
**連第 1 段的測試都不會跑**。這時要跑第 1 段，就指定檔案：`uv run pytest 02_makemore/test_bigram.py`。

每項測試檢查什麼，寫在 `test_mlp.py` 開頭（裡面有測資，實作時不用看）。

### 失敗時先看哪裡

- **錯誤訊息是「model 收到的 X 形狀／dtype 是 …」**：`evaluate` 或 `sample` 丟給 `model` 的 `X` 不符合共通約定。
  `sample` 每一步也要丟 2 維的 `(1, block_size)`，不是 1 維的 context
- **錯誤訊息是「model 已被呼叫超過 … 次」**：`sample` 停不下來。先檢查取到 `'.'` 有沒有結束，以及是不是依機率
  取樣（例如誤用機率最大的字元，就永遠取不到機率較小的 `'.'`）
- **`build_dataset` 的小例子通過、整份資料的比對失敗**：注意每個名字開始時 context 要重新補滿 `'.'`
- **訓練測試的 dev NLL 沒過門檻**：先確認前 4 個函式的測試都通過，因為訓練測試會用到它們。再看 train loss：
  train 也高代表還沒訓練到位（步數、學習率）；train 很低但 dev 高代表 overfitting
- **訓練測試報「model(X) 回傳的 tensor 在 mps」**：用 MPS 訓練後，回傳前沒有把模型搬回 CPU
- **`split_words` 改動了輸入**：測試用的 `words` 是整個測試過程共用的，被改動後，後面其他用到 `words` 的測試也可能
  跟著失敗。先修 `test_split_does_not_mutate_input`

## 第 3 段：Activations、Gradients、BatchNorm

### 這一段要做的事

第 2 段的模型訓練前 loss 是 8.24，遠高於「27 個字元機率相同」時的 ln 27 ≈ 3.30，前面一段訓練都花在修正
這件事。這一段先把初始化修好，再把網路加深，用 BatchNorm 讓每一層的 activation 維持在合理範圍。照影片的做法，
把 Linear、BatchNorm1d、Tanh 寫成用法像 PyTorch 模組的手寫 class（不繼承 `nn.Module`），並用每一層存下的
輸出畫 activation／gradient 統計圖（階段 2 的檢核項目）。

### 介面約定

檔案：`02_makemore/deep_mlp.py`。測試會 `from deep_mlp import Linear, BatchNorm1d, Tanh, init_deep`，訓練測試
另外會 `from deep_mlp import train_deep`。測試也會用第 2 段 `mlp.py` 的 `build_dataset`、`split_words`、`evaluate`、
`sample`，所以第 2 段的測試要先通過。

#### 三種層的共通約定

- 手寫 class，不繼承 `nn.Module`
- **呼叫**：`layer(x)`，`x` 是形狀 `(n, 輸入維度)` 的 float32 tensor，回傳 `(n, 輸出維度)`。回傳值同時存到
  `layer.out`（畫統計圖用；測試會檢查 `layer.out` 就是這次的回傳值）
- **`layer.parameters()`**：回傳 `list`，依各層規定的順序放要訓練的 tensor；沒有就回傳 `[]`
- 不能改動輸入；回傳的 tensor 在 CPU
- `requires_grad` 由誰設定不規定。測試需要梯度時會自己對參數呼叫 `requires_grad_(True)`，所以參數要是直接建立的
  tensor（leaf tensor），不能是由其他需要梯度的 tensor 算出來的

#### `Linear(fan_in, fan_out, bias=True, generator=None)`

- `.weight`：形狀 `(fan_in, fan_out)`；`.bias`：形狀 `(fan_out,)`，`bias=False` 時是 `None`
- 輸出 = `x @ weight + bias`（沒有 bias 就只有 `x @ weight`）
- `parameters()`：`[weight, bias]`，`bias=False` 時是 `[weight]`
- 初始值由你決定（這一段的重點之一）。隨機性只能來自 `generator`：同 seed 的 generator 得到相同權重，不受全域亂數
  狀態影響

#### `Tanh()`

- 輸出 = `tanh(x)`；`parameters()` 是 `[]`

#### `BatchNorm1d(dim, eps=1e-5, momentum=0.1)`

- 屬性：`eps`、`momentum`、`training`（初始 `True`），以及形狀都是 `(dim,)` 的 `gamma`（全 1）、`beta`（全 0）、
  `running_mean`（全 0）、`running_var`（全 1）
- 行為和 `torch.nn.BatchNorm1d(dim, eps=eps, momentum=momentum)` 相同（`gamma`、`beta` 對應它的 `weight`、`bias`），
  測試用它當正確答案。要注意的細節：
  - `training` 為 `True`：用這個 batch 每一欄的平均與 variance 正規化，再乘 `gamma`、加 `beta`。這裡的 variance 是
    **biased**（除以 n）
  - 同時更新 running stats：`running = (1 − momentum) × running + momentum × 這個 batch 的統計量`。`running_var`
    用的是 **unbiased** variance（除以 n − 1）。這個更新不能被 autograd 追蹤：更新之後 running stats 不能帶梯度
  - `training` 為 `False`：改用 `running_mean`、`running_var` 正規化，不更新 running stats；`n = 1` 也要能算
  - `training` 為 `True` 時需要 `n ≥ 2`（測試不會給 `n = 1`）
- `parameters()`：`[gamma, beta]`，不含 running stats

#### `init_deep(seed) -> tuple[model, int]`

- 回傳 `(還沒訓練的模型, block_size)`。隨機性只能來自 `seed`：同 seed 得到相同的模型，不受全域亂數狀態影響
- **model**：
  - 符合第 2 段的 model 約定：`model(X)`，`X` 是 `(n, block_size)` 的 int64，回傳 `(n, 27)` 的 logits，在 CPU 上
  - `model.layers`：`list`，依 forward 的順序放模型用到的每一個 `Linear`／`BatchNorm1d`／`Tanh`（只能放 `deep_mlp`
    的這三種 class）。embedding 查表不算一層，不放進來
  - 至少 3 個 `Tanh`，每個 `Tanh` 前面緊接著一個 `BatchNorm1d`
  - 切換訓練／評估模式的方式：把 `model.layers` 裡每個 `BatchNorm1d` 的 `training` 設成 `True`／`False`（影片的
    做法）。測試會自己這樣切換，所以 `BatchNorm1d` 不能放在 `model.layers` 以外的地方
- **通過標準**（seed 0、1；BatchNorm 在訓練模式；對 `split_words(整份資料, 42)` 的整份 dev 算一次 forward）：
  - dev NLL 和 ln 27 ≈ 3.296 相差 < 0.1
  - 每個 `Tanh` 層的輸出中，絕對值 > 0.97（飽和）的比例 < 10%

#### `train_deep(train_words, seed) -> tuple[model, int]`（測試預設跳過）

- 回傳 `(訓練好的模型, block_size)`；model 的約定同 `init_deep`
- 結構與訓練超參數由你決定；只能用 `train_words` 訓練；所有隨機性只能來自 `seed`
- 回傳時是哪種模式都可以，測試會把所有 `BatchNorm1d` 切到 `training = False` 再檢查
- **通過標準**（eval 模式）：
  - 同一列的輸出不能因為同一批裡有其他列而改變：`model(X[:1])` 要等於 `model(X[:5])[:1]`
  - 用你的 `evaluate` 算出的 dev NLL ≤ **bigram 在 dev 上的 NLL − 0.25**。bigram 的算法同第 2 段；用
    `torch.randperm` 打亂的 split 約 2.465，門檻約 2.215。第 2 段的 MLP（dev 2.252）達不到
  - 能交給 `sample` 產生名字
- 時間：參考實作 2 萬步約 20 秒就過門檻。建議讓它在幾十秒到幾分鐘內跑完；預計超過 10 分鐘的訓練要先確認
  （CLAUDE.md §4）。訓練結果照 CLAUDE.md §4 記錄在 NOTES.md 的實驗紀錄

### 建議順序

1. `Linear`、`Tanh` → `test_linear_*`、`test_tanh`
2. `BatchNorm1d` → `test_batchnorm_*`
3. `init_deep` → `test_init_*`：先想訓練前的 logits 應該長什麼樣子，再用 `layer.out` 看每一層的分佈
4. `train_deep` → `test_train_deep_beats_bigram`（要加 `--run-train`）

### 跑測試

```bash
uv run pytest 02_makemore/test_deep_mlp.py                                # 第 3 段（訓練測試跳過）
uv run pytest 02_makemore/test_deep_mlp.py -k batchnorm                   # 只跑 BatchNorm1d 的測試
uv run pytest 02_makemore/test_deep_mlp.py -k init -s                     # -s 會印出訓練前的 NLL 與各層飽和比例
uv run pytest 02_makemore/test_deep_mlp.py --run-train -k train_deep -s   # 只跑訓練測試
```

`deep_mlp.py` 不存在時會看到 `ModuleNotFoundError: No module named 'deep_mlp'`。測試一開始就 import `Linear`、
`BatchNorm1d`、`Tanh`、`init_deep` 這 4 個名稱：還沒寫到的 class 先寫成空的 class，函式先放
`raise NotImplementedError`。`train_deep` 只在訓練測試裡才 import。

和第 2 段一樣，只要 `test_deep_mlp.py` 收集失敗，`uv run pytest 02_makemore` 整個會在收集階段中斷，這時指定檔案跑
其他段的測試。`--run-train` 與路徑的注意事項同第 2 段。

每項測試檢查什麼，寫在 `test_deep_mlp.py` 開頭（裡面有測資，實作時不用看）。

### 失敗時先看哪裡

- **BatchNorm1d 訓練模式的輸出和 torch 差一點點**：正規化用的 variance 是 biased 還是 unbiased？batch 越小差越多
- **`running_var` 和 torch 不同、`running_mean` 相同**：同上，這次是更新 running stats 用的那一個
- **「running stats 不能帶梯度」**：更新 running stats 的運算被 autograd 追蹤了
- **eval 的測試失敗**：eval 時仍用這個 batch 的統計量，或 eval 時仍在更新 running stats
- **「model.layers 只能放 deep_mlp 的 Linear／BatchNorm1d／Tanh」**：例如放了 `torch.nn.Linear`，或把 embedding
  也放進去
- **訓練前 dev NLL 離 ln 27 太遠**：回想第 2 段的 8.24。模型一開始什麼都不知道時，輸出層的 logits 應該長什麼樣子？
- **Tanh 飽和比例太高**：看飽和那一層的前一層輸出分佈多寬
- **訓練測試報「eval 模式下，同一列的輸出…改變」**：BatchNorm1d 在 `training = False` 時沒有改用 running stats
- **訓練測試的 dev NLL 沒過門檻**：先確認其他測試都通過，再比較 train 和 dev loss（同第 2 段）

## 第 4 段：Becoming a Backprop Ninja

### 這一段要做的事

不用 autograd，自己推出網路裡每個變數的梯度。網路是單一隱藏層的版本：embedding → Linear → BatchNorm → tanh →
Linear → cross-entropy。照影片的 4 個練習：

1. 把 forward 拆成 19 個中間變數（`forward`），每個變數與參數的梯度各寫一個函式（`d_<名稱>`，共 26 個），從 loss
   往回一個一個推
2. cross-entropy：直接從 logits 算出 logits 的梯度，不經過中間變數（`cross_entropy_backward`）
3. BatchNorm：直接從 hpreact 的梯度算出 hprebn 的梯度（`batchnorm_backward`）
4. 用 2、3 組出全部參數的梯度（`loss_and_grads`），不呼叫 `.backward()` 訓練模型（`train_manual`）

所有梯度都以 PyTorch autograd 算出的值當正確答案比對（影片的 `cmp()`）。

### 介面約定

檔案：`02_makemore/backprop_ninja.py`。測試會 `from backprop_ninja import forward, cross_entropy_backward,
batchnorm_backward, loss_and_grads`，並用 `getattr(backprop_ninja, "d_<名稱>")` 取練習 1 的 26 個函式；訓練測試
另外會 `from backprop_ninja import train_manual`，並用第 2 段 `mlp.py` 的 `build_dataset`、`split_words`、`evaluate`、
`sample`。

#### 共通約定

- **不能用 autograd**：測試在 `torch.no_grad()` 裡呼叫你的函式，傳入的 tensor 也都不帶 `requires_grad`。呼叫
  `.backward()` 或 `torch.autograd.grad` 會直接報 `RuntimeError`。forward 照常算就好，在 `no_grad` 裡一樣能跑
- **params**：長度 7 的 `list`，順序固定是 `[C, W1, b1, W2, b2, bngain, bnbias]`（影片的順序），都是 float32：

  | 參數 | 形狀 |
  |---|---|
  | `C` | `(27, n_embd)` |
  | `W1` | `(block_size × n_embd, n_hidden)` |
  | `b1` | `(n_hidden,)` |
  | `W2` | `(n_hidden, 27)` |
  | `b2` | `(27,)` |
  | `bngain` | `(1, n_hidden)` |
  | `bnbias` | `(1, n_hidden)` |

  `n_embd`、`n_hidden`、`block_size` 和 batch 大小 `n` 都要從傳入的形狀讀出來，不能寫死。測試用三組資料：`video` 是
  影片的大小（`n` 32、`block_size` 3、`n_embd` 10、`n_hidden` 64）；`small` 是 `n` 7、`block_size` 4、`n_embd` 5、
  `n_hidden` 13；`tiny_var` 和 `video` 同大小，但 `bnvar` 很小（約 1e-5，和 `eps` 同一個數量級）。`batchnorm_backward`
  的測試也有一組 variance 這麼小的資料
- **Xb**：形狀 `(n, block_size)`、dtype `torch.int64`，值在 0..26；同一個字元可能在 batch 裡出現很多次。
  **Yb**：形狀 `(n,)`、dtype `torch.int64`，值在 0..26
- 不能改動傳入的任何 tensor（`params` 裡的每一個、`Xb`、`Yb`）
- 回傳的 tensor 在 CPU；import 時不能跑重的工作（同第 1 段）

#### forward 的定義

`forward` 的中間變數、以及所有函式的 loss，都照這張表算（影片的 forward）。`n` 是 batch 大小，
`eps` = 1e-5：

| 變數 | 定義 | 形狀 |
|---|---|---|
| `emb` | `C[Xb]` | `(n, block_size, n_embd)` |
| `embcat` | `emb` 每一列的 `block_size` 個 embedding 依序接起來（`emb.view(n, -1)`） | `(n, block_size × n_embd)` |
| `hprebn` | `embcat @ W1 + b1` | `(n, n_hidden)` |
| `bnmeani` | `hprebn` 每一欄的平均（除以 `n`） | `(1, n_hidden)` |
| `bndiff` | `hprebn − bnmeani` | `(n, n_hidden)` |
| `bndiff2` | `bndiff²`（逐元素） | `(n, n_hidden)` |
| `bnvar` | `bndiff2` 每一欄的和 **除以 `n − 1`** | `(1, n_hidden)` |
| `bnvar_inv` | `(bnvar + eps) ^ (−1/2)` | `(1, n_hidden)` |
| `bnraw` | `bndiff × bnvar_inv` | `(n, n_hidden)` |
| `hpreact` | `bngain × bnraw + bnbias` | `(n, n_hidden)` |
| `h` | `tanh(hpreact)` | `(n, n_hidden)` |
| `logits` | `h @ W2 + b2` | `(n, 27)` |
| `logit_maxes` | `logits` 每一列的最大值 | `(n, 1)` |
| `norm_logits` | `logits − logit_maxes` | `(n, 27)` |
| `counts` | `exp(norm_logits)` | `(n, 27)` |
| `counts_sum` | `counts` 每一列的和 | `(n, 1)` |
| `counts_sum_inv` | `counts_sum ^ (−1)` | `(n, 1)` |
| `probs` | `counts × counts_sum_inv` | `(n, 27)` |
| `logprobs` | `ln(probs)` | `(n, 27)` |
| loss | `−(1/n) Σᵢ logprobs[i, Yb[i]]` | 純量 |

表裡的 `×`、`−`、`+` 都是逐元素運算（形狀不同時 broadcast），`@` 是矩陣乘法。

**注意 `bnvar` 除以 `n − 1`（unbiased）**：這是影片的做法，和第 3 段的 `BatchNorm1d`、`torch.nn.BatchNorm1d`
訓練模式正規化用的 biased（除以 `n`）不同。這一段的所有函式都用 `n − 1`。

#### `Forward`（dataclass，已經定義在 `backprop_ninja.py`）

- 28 個欄位，型別都是 `torch.Tensor`：表裡的 19 個中間變數（`emb` … `logprobs`）、7 個參數（`C`、`W1`、`b1`、
  `W2`、`b2`、`bngain`、`bnbias`），以及 `Xb`、`Yb`。欄位名稱和表裡的名稱一字不差，用 `fwd.probs` 這樣取值
- 測試會用關鍵字參數建立它（`Forward(emb=..., ..., Yb=...)`），所以欄位名稱不能改；可以加方法，但不能加沒有預設值
  的欄位

#### `forward(params, Xb, Yb) -> tuple[float, Forward]`（練習 1）

- 回傳 `(loss, fwd)`
  - `loss`：Python `float`，表裡的 loss
  - `fwd`：`Forward`。中間變數的值和形狀照表的定義；參數、`Xb`、`Yb` 就是傳入的那個 tensor
- `fwd` 可以直接傳給自己的 `d_<名稱>`
- 通過標準：loss 相差 < 1e-5；每個中間變數和表的定義 `torch.allclose(atol=1e-6, rtol=1e-4)`

#### `d_<名稱>(fwd, d<上游>...) -> torch.Tensor`（練習 1，共 26 個）

- 函式名稱：`d_` 加上表裡的 19 個中間變數或 7 個參數的名稱
- 第一個參數 `fwd` 是 `Forward`。後面是**上游梯度**：表裡直接用到這個變數的每個中間變數，loss 對它的梯度，依表的
  順序排列，參數名稱是 `d` 加變數名稱（影片的命名）。測試用**關鍵字參數**傳入上游梯度，所以名稱要一字不差：

  | 函式 | 上游梯度 |
  |---|---|
  | `d_logprobs` | （沒有，只有 `fwd`） |
  | `d_probs` | `dlogprobs` |
  | `d_counts_sum_inv` | `dprobs` |
  | `d_counts_sum` | `dcounts_sum_inv` |
  | `d_counts` | `dcounts_sum`、`dprobs` |
  | `d_norm_logits` | `dcounts` |
  | `d_logit_maxes` | `dnorm_logits` |
  | `d_logits` | `dlogit_maxes`、`dnorm_logits` |
  | `d_h`、`d_W2`、`d_b2` | `dlogits` |
  | `d_hpreact` | `dh` |
  | `d_bngain`、`d_bnbias`、`d_bnraw` | `dhpreact` |
  | `d_bnvar_inv` | `dbnraw` |
  | `d_bnvar` | `dbnvar_inv` |
  | `d_bndiff2` | `dbnvar` |
  | `d_bndiff` | `dbndiff2`、`dbnraw` |
  | `d_bnmeani` | `dbndiff` |
  | `d_hprebn` | `dbnmeani`、`dbndiff` |
  | `d_embcat`、`d_W1`、`d_b1` | `dhprebn` |
  | `d_emb` | `dembcat` |
  | `d_C` | `demb` |

  `backprop_ninja.py` 裡已經照這張表寫好 26 個簽名
- 輸出：loss 對這個變數的梯度，形狀和該變數相同，包括 `(1, n_hidden)`、`(n, 1)` 這種保留的維度。對中間變數 `v`，
  等於在 autograd 裡對 `v` 呼叫 `retain_grad()` 之後 `v.grad` 的值
- 不能改動傳入的 `fwd`（包括裡面的 tensor）和上游梯度
- 測試會**各自單獨**呼叫每個函式：`fwd` 是測試照表算出的值，上游梯度是 autograd 算出的正確值。所以一個函式
  寫錯只有它自己那一項失敗，也可以不照順序寫
- 通過標準：和 autograd 的值 `torch.allclose(atol=1e-6, rtol=1e-4)`
- 測試分辨不出對錯的地方：`logit_maxes` 和 `b1` 的正確梯度非常接近 0（為什麼是推導的一部分），容許誤差 1e-6
  蓋得過它們；`d_logits` 裡經過 `logit_maxes` 的那一份也一樣小，漏掉它測試看不出來。這幾項通過之後，要自己
  確認推導

#### `cross_entropy_backward(logits, Yb) -> torch.Tensor`（練習 2）

- 輸入：`logits` 形狀 `(n, 27)`、float32；`Yb` 同共通約定。`n` 可能是 1
- 輸出：形狀 `(n, 27)`，loss 對 `logits` 的梯度。這裡的 loss 是 forward 表裡從 `logits` 到 loss 那一段，也就是
  `F.cross_entropy(logits, Yb)`（平均）
- 數值穩定：測試會給所有值都在 200 附近的 logits，結果不能有 `inf`／`nan`
- 練習的目的是不經過中間變數，直接寫出這個梯度的式子；測試只比對數值，不檢查寫法

#### `batchnorm_backward(dhpreact, hprebn, bngain) -> torch.Tensor`（練習 3）

- 輸入：
  - `dhpreact`：形狀 `(n, d)`，loss 對 `hpreact` 的梯度
  - `hprebn`：形狀 `(n, d)`
  - `bngain`：形狀 `(1, d)`
- 輸出：形狀 `(n, d)`，loss 對 `hprebn` 的梯度。`hprebn` 到 `hpreact` 的關係照 forward 表（`bnmeani` 到 `hpreact`
  那幾列，variance 除以 `n − 1`，`eps` = 1e-5）。`bnbias` 不影響這個梯度，所以不傳入
- `n ≥ 2`（測試用 2、5、32），`d` 任意
- 和練習 2 一樣，目的是直接寫出式子；測試只比對數值

#### `loss_and_grads(params, Xb, Yb) -> tuple[float, list[torch.Tensor]]`（練習 4）

- 回傳 `(loss, grads)`：`loss` 是 Python `float`（forward 定義的 loss）；`grads` 是 7 個 tensor 的 `list`，順序和形狀
  都和 `params` 相同
- 用練習 2、3 的函式組出來（影片練習 4 的做法）。forward 不需要拆成表裡的每個中間變數，例如 loss 可以直接用
  `F.cross_entropy` 算；數值要和表的定義相同
- 通過標準同練習 1

#### `train_manual(train_words, seed) -> tuple[model, int]`（練習 4，測試預設跳過）

- 回傳 `(訓練好的模型, block_size)`；model 符合第 2 段的 model 約定（`model(X)`，`X` 是 `(n, block_size)` 的 int64，
  回傳 `(n, 27)` 的 logits，在 CPU 上）
- **測試在 `torch.no_grad()` 裡呼叫 `train_manual`**：梯度用 `loss_and_grads` 算，參數自己更新
- 模型結構照上面的 forward 定義（才能用 `loss_and_grads`）；`n_embd`、`n_hidden`、`block_size`、初始化、學習率、
  步數都由你決定。只能用 `train_words` 訓練；所有隨機性只能來自 `seed`
- 評估時 BatchNorm 要用固定的 mean／variance，不能用這一批的統計量：`model(X[:1])` 要等於 `model(X[:5])[:1]`。
  固定的統計量怎麼來由你決定（例如影片在訓練完之後用整份 train 算一次，或第 3 段的 running stats）
- 通過標準：用你的 `evaluate` 算出的 dev NLL ≤ **bigram 在 dev 上的 NLL − 0.2**（split 與 bigram 的算法同第 2 段；
  用 `torch.randperm` 打亂的 split 約 2.465，門檻約 2.265）；也會檢查形狀、裝置，以及能不能交給 `sample`
- 時間：參考實作 2 萬步約 6 秒。預計超過 10 分鐘的訓練要先確認（CLAUDE.md §4）。訓練結果照 CLAUDE.md §4 記錄在
  NOTES.md 的實驗紀錄

### 建議順序

1. `forward` → `test_forward_*`
2. 26 個 `d_<名稱>` → `test_grad`：建議從 `d_logprobs` 開始往回寫（測試也依這個順序排列）。每個函式單獨測，還沒寫的
   只會讓自己那一項失敗
3. `cross_entropy_backward` → `test_cross_entropy_*`
4. `batchnorm_backward` → `test_batchnorm_backward`
5. `loss_and_grads` → `test_loss_and_grads`
6. `train_manual` → `test_train_manual_beats_bigram`（要加 `--run-train`）

### 跑測試

```bash
uv run pytest 02_makemore/test_backprop_ninja.py                                  # 第 4 段（訓練測試跳過）
uv run pytest 02_makemore/test_backprop_ninja.py -k "forward or test_grad"        # 練習 1
uv run pytest 02_makemore/test_backprop_ninja.py -k "test_grad and counts_sum"    # 名稱含 counts_sum 的梯度（d_counts_sum、d_counts_sum_inv）
uv run pytest 02_makemore/test_backprop_ninja.py -k "test_grad and small"         # 只跑 small 那一組大小
uv run pytest 02_makemore/test_backprop_ninja.py --run-train -k train_manual -s   # 只跑訓練測試
```

`backprop_ninja.py` 不存在時會看到 `ModuleNotFoundError: No module named 'backprop_ninja'`。測試一開始就 import
`forward`、`cross_entropy_backward`、`batchnorm_backward`、`loss_and_grads` 這 4 個名稱；`d_<名稱>` 在各自的測試裡才
取，少了只會讓那一項失敗（「backprop_ninja 裡沒有 d_xxx」）。`train_manual` 只在訓練測試裡才 import。

`forward` 對每組大小只呼叫一次，結果給該組的 20 項 `test_forward_*` 共用。它丟出例外時，那 20 項會一起顯示
`ERROR`，訊息都一樣，看第一個就好。

和前兩段一樣，只要 `test_backprop_ninja.py` 收集失敗，`uv run pytest 02_makemore` 整個會在收集階段中斷，這時指定
檔案跑其他段的測試。`--run-train` 與路徑的注意事項同第 2 段。

每項測試檢查什麼、門檻怎麼來，寫在 `test_backprop_ninja.py` 開頭（裡面有測資，實作時不用看）。

### 失敗時先看哪裡

- **「fwd 要是 Forward」、`AttributeError: 'dict' object has no attribute …`**：`forward` 回傳的第二個值不是 `Forward`
- **`test_forward_value` 從某一項開始全部失敗**：錯在失敗的第一項（表裡最上面那一項）
- **`TypeError: ... unexpected keyword argument`**：`d_<名稱>` 的參數名稱和上游梯度表不一樣
- **形狀不同**：梯度的形狀要和 forward 裡那個變數一樣。看這個變數在 forward 裡是和什麼形狀的 tensor 一起運算的
- **`d_emb` 通過、`d_C` 失敗**：同一個字元在 batch 裡出現很多次
- **video 那一組通過、small 那一組失敗或 `ERROR`**：寫死了某個維度或 batch 大小
- **只有 `tiny_var`（或 `batchnorm_backward` 最後一組）失敗**：variance 小到和 `eps` 差不多時才看得出差別。檢查反向
  的式子對 `eps` 的處理是不是和 forward 一致
- **`d_bndiff2`（或 `test_forward_value` 的 `bnvar`）失敗**；或 **`batchnorm_backward` 差一點點、batch 越小差越多**：除以 `n` 還是
  `n − 1`？
- **`RuntimeError: element 0 of tensors does not require grad`**：用了 `.backward()`
- **`test_cross_entropy_backward_large_logits` 出現 `inf`／`nan`**：logits 很大時 `exp` 會溢位。回想第 2 段 `log_softmax`
  比先 softmax 再 log 穩定的原因
- **訓練測試報「同一列的輸出不能因為同一批裡有其他列而改變」**：評估時 BatchNorm 用了這一批的統計量
- **訓練時印出的 loss 和 `loss_and_grads` 回傳的 loss 不一樣**：模型的訓練模式 forward 和 `loss_and_grads` 不一致，
  例如正規化用的不是這一批的 mean／variance。梯度只來自 `loss_and_grads`，所以這時印出的 loss 不代表訓練進度
- **訓練測試的 dev NLL 沒過門檻**：先確認 `test_loss_and_grads` 通過；再看第 0 步的 loss 是否接近 ln 27 ≈ 3.30
  （第 3 段的初始化），以及固定學習率時後期 loss 是否還在跳（第 3 段的學習率衰減）；最後比較 train 和 dev loss
  （同第 2 段）

## 第 5 段：WaveNet

### 這一段要做的事

第 2～4 段的模型一次把整個 context 攤平，接進同一層。這一段把 context 拉長，改成**階層式**：先把相鄰的兩個字元
合併，再把相鄰的兩組合併，一層一層往上（影片的 WaveNet 風格）。同時把模型寫成像 PyTorch 的模組：

- 新的層：`Embedding`、`FlattenConsecutive`、`Sequential`
- 中間層的輸出變成 3 維 `(n, L, C)`，所以第 3 段的 `BatchNorm1d` 要能處理 3 維輸入
- `Linear`、`Tanh` 沿用第 3 段 `deep_mlp.py` 的

### 介面約定

檔案：`02_makemore/wavenet.py`（新的層與模型），以及 `02_makemore/deep_mlp.py`（`BatchNorm1d` 擴充）。測試會
`from wavenet import Embedding, FlattenConsecutive, Sequential, init_wavenet`、`from deep_mlp import Linear,
BatchNorm1d, Tanh`，訓練測試另外會 `from wavenet import train_wavenet`。測試也會用第 2 段 `mlp.py` 的
`build_dataset`、`split_words`、`evaluate`、`sample`。

#### 共通約定（同第 3 段）

- 手寫 class，不繼承 `nn.Module`
- **呼叫**：`layer(x)`，回傳值同時存到 `layer.out`（測試會檢查 `layer.out` 就是這次的回傳值）
- **`layer.parameters()`**：回傳 `list`，依各層規定的順序放要訓練的 tensor；沒有就回傳 `[]`
- 不能改動輸入；回傳的 tensor 在 CPU
- `requires_grad` 由誰設定不規定；參數要是直接建立的 tensor（leaf tensor）
- 隨機性只能來自傳入的 `generator`／`seed`，不受全域亂數狀態影響

#### `Embedding(num_embeddings, embedding_dim, generator=None)`

- `.weight`：形狀 `(num_embeddings, embedding_dim)`，初始值由你決定
- `emb(IX)`：`IX` 是任意形狀的 int64 tensor（測試用 1 維和 2 維），輸出 `weight[IX]`，形狀是 `IX` 的形狀後面
  加上 `embedding_dim`。例：`IX` 是 `(n, T)` → 輸出 `(n, T, embedding_dim)`
- `parameters()`：`[weight]`

#### `FlattenConsecutive(n)`

- 輸入 `x`：形狀 `(B, T, C)` 的浮點數 tensor。`T` 是 `n` 的倍數（測試不會給不整除的 `T`）
- 輸出：把時間軸上相鄰的 `n` 個向量依時間順序接成一個，形狀 `(B, T/n, n·C)`。也就是輸出的第 `t` 個位置是
  `x[:, t·n]`、`x[:, t·n+1]`、…、`x[:, t·n+n−1]` 依序接起來
- **`T/n = 1` 時輸出 2 維 `(B, n·C)`**，不是 `(B, 1, n·C)`（影片的做法，最後一層 `Linear` 才會輸出 `(B, 27)`）
- `parameters()`：`[]`

#### `Sequential(layers)`

- `layers`：層的 `list`。`.layers` 就是這些層，順序不變
- `seq(x)`：依序把 `x` 丟進每一層，回傳最後一層的輸出（也存到 `seq.out`）
- `parameters()`：依層的順序把每一層的 `parameters()` 串成一個 `list`

#### `Linear`（`deep_mlp.py`，第 3 段的約定不變）

- 也要能吃 3 維輸入：`(n, L, fan_in)` → `(n, L, fan_out)`，等於 `x @ weight + bias`

#### `BatchNorm1d`（`deep_mlp.py`，擴充）

- variance 沿用第 3 段的規則（和 `torch.nn.BatchNorm1d` 一致），和第 4 段 backprop ninja 全部除以 `n − 1` 不同
- **2 維輸入 `(n, C)`**：行為和第 3 段完全相同（`test_deep_mlp.py` 要繼續通過）
- **3 維輸入 `(n, L, C)`**：每個 channel（最後一維）的 mean／variance 用前兩維的全部 `n × L` 個值一起算。其他規則同
  2 維：正規化用 biased variance（除以 `n·L`）；running stats 用 unbiased variance（除以 `n·L − 1`）更新，更新不能被
  autograd 追蹤；`training = False` 時用 running stats，不更新
- `gamma`、`beta`、`running_mean`、`running_var` 的形狀都維持 `(C,)`，3 維輸入更新之後也一樣
- 等同於把 `x` 轉置成 `(n, C, L)` 丟給 `torch.nn.BatchNorm1d(C)`，再把輸出轉置回來；測試用它當正確答案
- 訓練模式需要 `n·L ≥ 2`（測試有 `n = 1`、`L = 4` 的情況）；eval 模式 `n = 1` 也要能算

#### `init_wavenet(seed) -> tuple[model, int]`

- 回傳 `(還沒訓練的模型, block_size)`；同 seed 得到相同的模型
- **model**：
  - 符合第 2 段的 model 約定：`model(X)`，`X` 是 `(n, block_size)` 的 int64，回傳 `(n, 27)` 的 logits，在 CPU 上
  - `model.layers`：`list`，依 forward 的順序放模型用到的每一層，只能是 `Embedding`、`FlattenConsecutive`、`Linear`、
    `BatchNorm1d`、`Tanh`（不能放 `Sequential`，要攤平成一個 list）。**和第 3 段不同，`Embedding` 現在是一層，要放進來**。
    model 本身可以是 `Sequential`
  - 至少 2 個 `FlattenConsecutive`，而且**第一個 `FlattenConsecutive` 的輸出是 3 維 `(n, L, C)`、`L < block_size`**
    （不是一次把整個 context 攤平）
  - 至少 2 個 `Tanh`，每個 `Tanh` 前面緊接著一個 `BatchNorm1d`
  - 切換訓練／評估模式的方式同第 3 段：測試會把 `model.layers` 裡每個 `BatchNorm1d` 的 `training` 設成 `True`／`False`
- **通過標準**（seed 0、1；BatchNorm 在訓練模式；對 `split_words(整份資料, 42)` 的整份 dev 算一次 forward）：
  - dev NLL 和 ln 27 ≈ 3.296 相差 < 0.1
  - 每個 `Tanh` 層的輸出（可能是 3 維，全部元素一起算）中，絕對值 > 0.97 的比例 < 10%

#### `train_wavenet(train_words, seed) -> tuple[model, int]`（測試預設跳過）

- 回傳 `(訓練好的模型, block_size)`；model 的約定同 `init_wavenet`
- 結構與訓練超參數由你決定；只能用 `train_words` 訓練；所有隨機性只能來自 `seed`
- 回傳時是哪種模式都可以，測試會把所有 `BatchNorm1d` 切到 `training = False` 再檢查
- **通過標準**（eval 模式）：
  - `model(X[:1])` 要等於 `model(X[:5])[:1]`
  - 用你的 `evaluate` 算出的 dev NLL ≤ **bigram 在 dev 上的 NLL − 0.40**（用 `torch.randperm` 打亂的 split 約 2.465，
    門檻約 2.065）。第 3 段的 deep MLP（2.2102）、第 4 段（2.2447）都達不到
  - 能交給 `sample` 產生名字
- 時間：參考實作 6 萬步約 55 秒、dev 約 2.00；只訓練 5000 步是 2.14，不夠。建議控制在 1～2 分鐘；預計超過 10 分鐘的
  訓練要先確認（CLAUDE.md §4）。訓練結果照 CLAUDE.md §4 記錄在 NOTES.md 的實驗紀錄

### 建議順序

1. `Embedding`、`FlattenConsecutive`、`Sequential` → `test_embedding_*`、`test_flatten_consecutive`、`test_sequential`
2. `Linear` 的 3 維輸入 → `test_linear_3d`（第 3 段的寫法可能已經通過）
3. `BatchNorm1d` 的 3 維輸入 → `test_batchnorm_3d_*`、`test_batchnorm_2d_unchanged`；改完也跑一次 `test_deep_mlp.py`
4. `init_wavenet` → `test_init_*`
5. `train_wavenet` → `test_train_wavenet_beats_threshold`（要加 `--run-train`）

### 跑測試

```bash
uv run pytest 02_makemore/test_wavenet.py                                     # 第 5 段（訓練測試跳過）
uv run pytest 02_makemore/test_wavenet.py -k batchnorm                        # 只跑 BatchNorm1d 的測試
uv run pytest 02_makemore/test_deep_mlp.py                                    # 改 BatchNorm1d 之後，確認第 3 段沒壞
uv run pytest 02_makemore/test_wavenet.py -k init -s                          # -s 會印出訓練前的 NLL 與各層飽和比例
uv run pytest 02_makemore/test_wavenet.py --run-train -k train_wavenet -s     # 只跑訓練測試
```

`wavenet.py` 不存在時會看到 `ModuleNotFoundError: No module named 'wavenet'`。測試一開始就 import `Embedding`、
`FlattenConsecutive`、`Sequential`、`init_wavenet` 這 4 個名稱：還沒寫到的 class 先寫成空的 class，函式先放
`raise NotImplementedError`。`train_wavenet` 只在訓練測試裡才 import。

和前幾段一樣，只要 `test_wavenet.py` 收集失敗，`uv run pytest 02_makemore` 整個會在收集階段中斷，這時指定檔案跑
其他段的測試。`--run-train` 與路徑的注意事項同第 2 段。

每項測試檢查什麼、門檻怎麼來，寫在 `test_wavenet.py` 開頭（裡面有測資，實作時不用看）。

### 失敗時先看哪裡

- **`test_batchnorm_3d_*` 的輸出和 torch 不同**：3 維時對哪幾維取 mean／variance？每個 channel 要用 `n × L` 個值
- **3 維輸入之後 running stats 的形狀變成 `(L, C)` 或 `(1, 1, C)`**：取統計量的維度不對，或用了 `keepdim=True`
  （第 3 段卡在哪第 3 點同一種問題）
- **`running_mean` 對、`running_var` 不對**：unbiased variance 除以的是 `n·L − 1`
- **`test_batchnorm_2d_unchanged` 或 `test_deep_mlp.py` 失敗**：擴充 3 維時改到了 2 維的行為
- **改了 `BatchNorm1d`，3 維的測試卻像沒改一樣**：測試是 `from deep_mlp import BatchNorm1d`，改在 `wavenet.py` 的版本
  不會被測到
- **`test_flatten_consecutive` 形狀對、數值不對**：接起來的順序不是依時間順序
- **`T/n = 1` 那幾項的形狀是 `(B, 1, n·C)`**：這時要壓成 2 維
- **`RuntimeError: shape '[…, 0, …]' is invalid`**：`FlattenConsecutive(n)` 的 `n` 比時間軸長度大。`n` 是相鄰幾個合併，
  不是輸出寬度
- **model 輸出 `(n, 2, 27)` 之類的 3 維，或 `evaluate` 報形狀錯誤**：時間軸還沒合併到剩 1。`block_size` 每次除以 `n`，
  要幾層才會剩 1？
- **`TypeError: descriptor '__init__' requires a 'super' object`**：`super.__init__` 少了括號，要寫 `super().__init__`
- **「model.layers 只能放 …」**：放了 `Sequential` 或 `torch.nn` 的層；`Sequential` 要攤平
- **`test_init_hierarchical` 失敗**：第一個 `FlattenConsecutive` 一次把整個 context 攤平了
- **訓練前 dev NLL 離 ln 27 太遠、Tanh 飽和比例太高**：同第 3 段（最後一層的初始化、飽和那一層前面的分佈）
- **訓練測試報「eval 模式下，同一列的輸出…改變」**：BatchNorm1d 在 `training = False` 時沒有改用 running stats
- **訓練測試的 dev NLL 沒過門檻**：先確認其他測試都通過；再看第 0 步的 loss、學習率衰減（太早降到很小，後面的步數
  等於浪費）、步數；最後比較 train 和 dev loss（同第 2 段）。train 也高、兩者差距小是 underfitting，要增加擬合能力，
  例如 embedding 維度、寬度；模型變大時訓練時間也會跟著變長
- **同一份程式跑出不同結果**：seed 固定時結果可以完全重現。先確認跑的是不是同一份程式（例如編輯器裡還沒存檔的修改）
