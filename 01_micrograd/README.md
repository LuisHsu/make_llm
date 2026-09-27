# 階段 1：micrograd（Python 版）

教材、筆記與檢核在 `NOTES.md`。這份檔案說明要寫什麼、測試怎麼跑。C++ 版另見 `cpp/README.md`。

## 分工

| 誰 | 寫什麼 |
|---|---|
| 學習者 | `engine.py`：`Value` 類別、各運算的反向傳播、拓樸排序與 `backward()` |
| 學習者 | `nn.py`：`Neuron`、`Layer`、`MLP` |
| 學習者 | 訓練用的程式（例如 `training.py`，沒有測試） |
| Claude | `conftest.py`、`test_micrograd_engine.py`、`test_micrograd_nn.py` |

依專案 CLAUDE.md §1.1，核心實作由學習者親手寫，Claude 不提供骨架或範例程式碼。

## 目錄

```
01_micrograd/
├── README.md
├── NOTES.md
├── engine.py                  # ← 學習者寫
├── nn.py                      # ← 學習者寫
├── conftest.py                # 讓測試能直接 import engine / nn
├── test_micrograd_engine.py
├── test_micrograd_nn.py
└── cpp/                       # C++ 版（另見 cpp/README.md）
```

## 介面約定：`engine.py`

測試會 `from engine import Value`。

### 建立與讀寫

- `Value(data)`：`data` 是 Python 的 `float`（測試也會傳 `0.0`、`2.0` 這種值）
- `.data`：這個節點的數值，Python `float`。測試會讀，也會直接寫（`p.data -= 0.01 * p.grad`）
- `.grad`：輸出對這個節點的偏微分，Python `float`。建立時是 `0`；測試會讀，也會直接寫（`p.grad = 0.0`）
- 建立新節點的運算**不能改動輸入節點的 `.data`**

### 運算

以下每一種都要回傳新的 `Value`，而且都要能反向傳播：

| 運算 | 寫法 |
|---|---|
| 加、減、乘、除 | `a + b`、`a - b`、`a * b`、`a / b`（兩邊都是 `Value`） |
| 負號 | `-a` |
| 與 Python 數字混用，數字在右邊 | `a + 1`、`a * 2`、`a - 3`、`a / 4`、`w * 0.1` |
| 與 Python 數字混用，數字在左邊 | `1 + a`、`2 * a`、`1 - a`、`3 / a` |
| 次方，指數是 Python 數字 | `a ** 2`、`a ** 3`、`a ** 0.5`、`a ** -1`、`a ** -2.5` |
| 非線性 | `a.tanh()`、`a.relu()`、`a.exp()` |

- 次方只測底數為正（1.7）的情況；指數是 `Value` 的情況不測
- `relu` 只測 x ≠ 0 的點
- 測試的輸入值都不大（|x| 約 3 以內）

### `backward()`

- 對輸出呼叫 `out.backward()` 之後：
  - `out.grad == 1.0`
  - 計算圖裡每個節點的 `.grad` 是 `out` 對它的偏微分
- **同一個節點被使用多次時，梯度要累加**：例如 `a + a`、`a * a`，或一個中間節點同時被好幾個節點用到

### 正確答案怎麼來

每項測試用同一個運算式分別算一次 `Value` 版和 PyTorch（float64）版，比對輸出值與每個輸入的梯度，
容許誤差 `rel_tol=1e-9, abs_tol=1e-12`。

## 介面約定：`nn.py`

測試會 `from nn import MLP, Layer, Neuron`，並 `from engine import Value`。

| 類別 | 建構 | 參數個數 |
|---|---|---|
| `Neuron(nin)` | `nin` 個輸入 | `nin` 個權重 + 1 個 bias |
| `Layer(nin, nout)` | `nout` 個 neuron，每個都有 `nin` 個輸入 | `nout × (nin + 1)` |
| `MLP(nin, nouts)` | `nouts` 是各層輸出數的 list，例如 `MLP(3, [4, 4, 1])` | 各層參數個數的總和 |

- **呼叫**：三者都可以用 list 呼叫，例如 `model([1.0, -2.0, 0.5])`；list 的元素可能是 `float`，也可能是 `Value`
- **回傳**：
  - `Neuron` 回傳一個 `Value`
  - `Layer`、`MLP`：輸出只有一個時回傳 `Value`，多個時回傳 `list[Value]`
- **`.parameters()`**：回傳所有可訓練參數的 `list[Value]`
  - 每個參數都是各自獨立的 `Value`（不能有兩個位置共用同一個物件）
  - 回傳的是模型內部真正在用的 `Value`：測試直接改它們的 `.data`、`.grad`，就要影響模型
- **初始化**：權重與 bias 要隨機初始化。訓練那項測試在建模型前會 `random.seed(0)`
- **訓練**：測試用 4 筆資料、loss = Σ(輸出 − 目標)²（用 `sum(..., Value(0.0))` 加總），做 50 步梯度下降
  （學習率 0.01、每步先把所有參數的 `.grad` 設為 0），loss 要比一開始低

## 觀察計算圖（選用）

實作 `Value` 和 `backward()` 時，把計算圖畫出來看很有幫助：每個節點的 `data`、`grad`，以及它是由哪個運算、
哪些子節點產生的。可以用 pygraphviz 自己寫一個把 `Value` 走訪成圖的小工具（影片裡也有類似的做法）。

pygraphviz 不在預設的套件裡，因為它需要系統先裝好 graphviz 函式庫才能編譯：

```bash
brew install graphviz                        # macOS（Homebrew）
sudo apt install graphviz graphviz-dev       # Debian／Ubuntu
uv sync --group viz                          # 再把 pygraphviz 裝進 .venv
```

- 之後執行時用 `uv run --group viz python ...`。**直接跑不加 `--group viz` 的 `uv sync`，會把 pygraphviz 移除**
- 這個工具是觀察用的，測試不會用到
- 已確認可用：macOS（Homebrew 的 graphviz）。其他系統未實測

## 跑測試

```bash
uv run pytest 01_micrograd                         # 全部
uv run pytest 01_micrograd/test_micrograd_engine.py
uv run pytest 01_micrograd -k tanh                 # 名稱含 tanh 的測試
```

`engine.py` 或 `nn.py` 不存在時，會看到 `ModuleNotFoundError`，那個測試檔整個無法收集。只寫了一部分時，
還沒寫的部分先放 `raise NotImplementedError`，讓已寫好的測試能跑。

每項測試檢查什麼，寫在各測試檔開頭（裡面有測資，實作時不用看）。

## 失敗時先看哪裡

- **基本運算都過，但「計算圖結構」那幾項的梯度不對**：看 `backward()`，以及同一個節點被使用多次時發生什麼事
- **數字在左邊的運算（`2 * a`、`1 - a`、`3 / a`）無限遞迴或結果錯**：查一下 Python 遇到 `數字 運算子 物件` 時，
  會呼叫物件的哪個方法
- **值對、梯度只差一點點**：先確認導數公式，再看它在數值上夠不夠精準
