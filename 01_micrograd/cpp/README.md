# 階段 1 C++ 版：micrograd

Python 版在上一層目錄（`engine.py`、`nn.py`）。這裡是同一個東西的 C++ 實作。

## 分工

| 誰 | 寫什麼 |
|---|---|
| 學習者 | `include/engine.hpp`、`engine.cpp` 等核心實作：`Value`、各運算的反向傳播、拓樸排序與 `backward()` |
| 學習者 | `include/nn.hpp`（及需要的話 `nn.cpp`）：`Neuron`、`Layer`、`MLP` |
| Claude | 建置設定、`tools/` 的 golden 值產生器、`tests/` 底下的測試 |

依專案 CLAUDE.md §1.1，核心實作由學習者親手寫，Claude 不提供骨架或範例程式碼。

## 目錄

```
01_micrograd/cpp/
├── CMakeLists.txt          # 這個階段的建置設定
├── engine.cpp              # ← 學習者寫（可省略，只寫 header 也可以）
├── nn.cpp                  # ← 學習者寫（可省略）
├── include/engine.hpp      # ← 學習者寫
├── include/nn.hpp          # ← 學習者寫
├── tests/
│   ├── test_harness.cpp    # golden 檔本身的健檢（不碰實作，隨時可跑）
│   ├── test_engine.cpp     # engine 與 golden 值逐案比對
│   ├── test_nn.cpp         # nn 與 golden 值逐案比對
│   └── golden/
│       ├── engine_cases.json
│       └── nn_cases.json
└── tools/
    ├── gen_golden.py       # 產生 engine 的 golden 值
    └── gen_nn_golden.py    # 產生 nn 的 golden 值
```

## 介面約定：engine

測試直接寫死這些名稱，改了就要同步改測試。

- 標頭檔：`include/engine.hpp`
- namespace：`mg`
- 型別：`mg::Value`，內部用 `shared_ptr` 建計算圖；**複製 `Value` 共用同一個節點**，不是各自獨立的副本
- 數值型別一律 `double`

測試會用到的寫法：

| 用途 | 寫法 |
|---|---|
| 建葉節點 | `mg::Value a{2.0};` |
| 四則運算 | `a + b`、`a - b`、`a * b`、`a / b` |
| 一元負號 | `-a` |
| 與 `double` 混用（兩側都要支援） | `a + 1.0`、`2.0 * a`、`1.0 - a`、`3.0 / a` |
| 次方（指數是 `double`） | `a.pow(3.0)`、`a.pow(0.5)`、`a.pow(-2.5)` |
| 非線性 | `a.tanh()`、`a.relu()`、`a.exp()` |
| 讀值、讀梯度 | `a.data()`、`a.grad()`，都回傳 `double` |
| 反向傳播 | `out.backward();` 之後各輸入的 `grad()` 就是輸出對它的偏微分 |

約定的行為：

- `grad()` 在 `backward()` 之前是 0
- 一個節點被多個節點使用時，梯度要累加，不是覆蓋
- `relu` 在 x = 0 的梯度取 0，和 PyTorch 一致（`relu_zero` 這個 case 有測）

## 介面約定：nn

`include/nn.hpp`，一樣在 namespace `mg`。三個類別對應 Python 版的 `Neuron`、`Layer`、`MLP`。

| 用途 | 寫法 |
|---|---|
| 建 neuron／layer／MLP | `mg::Neuron n{3};`、`mg::Layer l{3, 4};`、`mg::MLP m{3, std::vector<size_t>{4, 4, 1}};` |
| 前向 | `n(xs)` 回傳 `mg::Value`；`l(xs)`、`m(xs)` 回傳 `std::vector<mg::Value>` |
| 輸入型別 | `const std::vector<mg::Value>&` |
| 參數 | `parameters()` 回傳 `std::vector<mg::Value>` |

約定的行為：

- 大小一律 `size_t`；`MLP` 的第二個參數是 `std::vector<size_t>`，即各層的輸出數
- **`nout == 1` 時 `Layer`／`MLP` 也回傳長度 1 的 vector**，不跟 Python 版回傳單一 `Value`
  的寫法（那個 if 在 C++ 會讓回傳型別不唯一）
- `parameters()` 的順序固定為：層 → neuron → 該 neuron 的權重（依輸入順序）→ bias。
  測試依這個順序把 golden 權重寫進模型，順序不同就會整組對不上
- `parameters()` 回傳的 `Value` 與模型內部**共用節點**（engine 已經是 `shared_ptr` 語意），
  所以測試直接寫 `p.data() = x`、讀 `p.grad()` 就會改到／讀到真正的參數
- 每個參數都是各自獨立的節點，`parameters()` 每次回傳的順序要一樣
- **每個 neuron 都套 `tanh`**（和 `01_micrograd/nn.py` 一樣）。換成別的非線性函數
  golden 值就不是正確答案了，要改 `gen_nn_golden.py` 的 `NONLINEARITY` 重新產生
- 權重要隨機初始化（不能全部相同，否則同一層的 neuron 會永遠一模一樣）

`include/nn.hpp` 不存在或是空的時候，configure 會印訊息並略過 nn 測試。**這是 configure
時檢查的**，所以新增 `include/engine.hpp`、`engine.cpp`、`include/nn.hpp` 或 `nn.cpp` 之後要重跑一次 `cmake -S . -B build`。

## 建置與測試

```bash
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```

以上指令都在 repo 根目錄執行。第一次 configure 會用 `FetchContent` 下載 GoogleTest 與
nlohmann/json，需要網路。`build/` 不進 git。

`include/engine.hpp` 不存在或是空的時候，configure 會印出訊息並略過 engine 與 nn 測試，只跑 golden 檔健檢的
4 項。所以看到「只有 4 項測試」時，先確認那個檔案在不在。`engine.cpp`、`nn.cpp` 存在時才會編進 library，
只寫 header 也可以。

只跑某一個 case：

```bash
./build/01_micrograd/cpp/micrograd_engine_tests --gtest_filter='*deep_composite*'
```

## golden 值

兩個 golden 檔都由 PyTorch float64 算出，是這些測試的正確答案來源。

```bash
uv run python 01_micrograd/cpp/tools/gen_golden.py     # engine_cases.json
uv run python 01_micrograd/cpp/tools/gen_nn_golden.py  # nn_cases.json
```

### `engine_cases.json`

每個 case 有 `name`、`expr`（給人看的說明）、`inputs`、`out`、`grads`。
`test_engine.cpp` 裡有一張同名的運算式表，兩邊必須一一對應，否則
`EngineCases.GoldenAndBuildersMatch` 會失敗——這是為了避免有 case 安靜地沒被測到。

新增 case 的步驟：在 `gen_golden.py` 的 `CASES` 加一項 → 重跑產生器 → 在
`test_engine.cpp` 用同一個名稱加上對應的運算式。

### `nn_cases.json`

神經網路的權重是隨機初始化的，沒辦法直接比對，所以 golden 檔裡也放權重：
`params`（依上面說的順序）、`inputs`、`targets`、`outputs`、`loss`、`param_grads`。
測試先把 `params` 寫進模型，再比對輸出、loss 與 backward 之後的參數梯度。
`loss` 是所有樣本、所有輸出的 `(輸出 − target)²` 總和。

`kind` 欄位（`neuron` / `layer` / `mlp`）決定測試要建哪個類別。

新增 case 只要在 `gen_nn_golden.py` 的 `CASES` 加一項再重跑產生器，測試會自動帶到。

## 測試內容

engine（`test_engine.cpp`）：

- `Golden/EngineCase.MatchesPyTorch/<case 名稱>`：25 個獨立的 case，各自比對輸出值與每個
  輸入的梯度，相對容差 1e-9
- `EngineCases.GoldenAndBuildersMatch`：golden 檔與運算式表一一對應
- `EngineSemantics.CopiesShareTheSameNode`：複製 `Value` 共用節點

nn（`test_nn.cpp`）：

- `Golden/NnCase.MatchesPyTorch/<case 名稱>`：6 個 case，比對輸出、loss 與每個參數的梯度
- `NnShape.*`：參數個數、輸出個數、參數是獨立節點且順序穩定
- `NnInit.WeightsAreNotAllIdentical`：有真的隨機初始化
- `NnTraining.ReducesLoss`：50 步 SGD（lr 0.01）之後 loss 要下降

golden 檔本身（`test_harness.cpp`，不碰實作，實作還沒寫也跑得動）：

- `GoldenFile.*`、`NnGoldenFile.*`：檔案能解析、欄位齊全、形狀一致、數值有限

## 編譯設定

C++20、`-Wall -Wextra`、產生 `compile_commands.json` 供編輯器使用。
