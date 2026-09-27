# 階段 2：makemore

教材、筆記與檢核在 `NOTES.md`。這份檔案說明怎麼開始、測試怎麼跑、失敗時看哪裡。
目前寫到第 2 段（MLP），之後每開始一段就往下補。

## 分工

| 誰 | 寫什麼 |
|---|---|
| 學習者 | `bigram.py`：字元表、計數、機率、NLL、取樣、神經網路版 bigram |
| 學習者 | `mlp.py`（及訓練用的程式）：資料集、切分、評估、取樣、MLP 模型與訓練 |
| Claude | `names_data.py`（資料下載與讀取）、`test_bigram.py`、`test_mlp.py`、`golden/`、`conftest.py` |

依專案 CLAUDE.md §1.1，核心實作由學習者親手寫，Claude 不提供骨架或範例程式碼。

## 目錄

```
02_makemore/
├── README.md
├── NOTES.md
├── bigram.py            # ← 學習者寫（第 1 段）
├── mlp.py               # ← 學習者寫（第 2 段）
├── names_data.py        # 下載 names.txt、load_words()
├── conftest.py          # 讓測試能 import 這個資料夾的模組；提供 words fixture；--run-train 選項
├── test_bigram.py       # 第 1 段的測試
├── test_mlp.py          # 第 2 段的測試
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
    split seed 42 時約 2.453，所以門檻約 2.253
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
