# make_llm — 從零學習設計 LLM

從自動微分開始，一路做到自己設計、訓練、調整專用 LLM 的練習 repo。每個階段都附測試，但**實作是空的**，
要自己寫。教材主要是 Andrej Karpathy 的影片系列。

## 需要的東西

- [uv](https://docs.astral.sh/uv/)：管理 Python 版本與套件。Python 3.12 會由 uv 自動安裝
- git
- （選做）C++ 版 micrograd：CMake、Ninja、支援 C++20 的編譯器。第一次建置時會下載 GoogleTest 與 nlohmann/json
- （選用）觀察計算圖：系統的 graphviz 函式庫，再裝 `viz` group 的 pygraphviz。見 `01_micrograd/README.md`「觀察計算圖」

PyTorch 會自動使用可用的加速器：NVIDIA GPU（CUDA）、Apple Silicon（MPS），沒有的話用 CPU。前幾個階段只用 CPU
也跑得動。

## 開始

```bash
git clone git@github.com:LuisHsu/make_llm.git
cd make_llm
uv sync                                  # 建立 .venv 並安裝套件
uv run python 00_env/check_env.py        # 環境檢查，最後要印出「全部通過」
```

之後照 `CLAUDE.md` Part 2 的學習計畫，從階段 1 開始。

## 每個階段怎麼做

1. 看該階段的教材（列在各階段資料夾的 `NOTES.md`）
2. 讀該階段的 `README.md`：要寫哪些檔案、介面約定、測試怎麼跑、失敗時先看哪裡。
   **只看 README 就能實作，不需要讀測試碼**；測試檔與 `golden/` 裡有測資，實作時不用看
3. 自己寫實作，跑測試直到全過
4. 在 `NOTES.md` 寫下學到什麼、卡在哪、實驗紀錄，並對照「檢核」項目

| 資料夾 | 內容 | 要寫的檔案 |
|---|---|---|
| `00_env/` | 環境準備 | （不用寫，跑 `check_env.py`） |
| `01_micrograd/` | 自動微分與反向傳播 | `engine.py`、`nn.py`；C++ 版（選做）見 `01_micrograd/cpp/README.md` |
| `02_makemore/` | 語言模型基礎：bigram、MLP、深層 MLP 與 BatchNorm、手推 backprop、WaveNet 風格模型 | `bigram.py`、`mlp.py`、`deep_mlp.py`、`backprop_ninja.py`、`wavenet.py` |
| `03_gpt/` | Transformer 與 GPT：資料與 bigram baseline、self-attention head、Transformer block、小 GPT | `char_data.py`、`bigram_lm.py`、`attention.py`、`single_head_lm.py`、`gpt.py` |

階段 3 目前到第 3 段（小 GPT）；第 4 段與後面的階段（Tokenizer、GPT-2、現代架構、post-training、專用 LLM）列在
`CLAUDE.md` 的學習計畫，資料夾還沒建立。

## 跑測試

一律指定階段的路徑：

```bash
uv run pytest 01_micrograd
uv run pytest 02_makemore/test_bigram.py
uv run pytest 02_makemore/test_mlp.py
uv run pytest 02_makemore/test_mlp.py --run-train -k train_mlp -s   # 會實際訓練模型，預設跳過
uv run pytest 02_makemore/test_deep_mlp.py                          # 階段 2 第 3 段；第 4、5 段是 test_backprop_ninja.py、test_wavenet.py
uv run pytest 03_gpt/test_char_data.py                              # 階段 3 第 1 段；其餘見 03_gpt/README.md
```

還沒寫實作時，測試會因為 `ModuleNotFoundError` 無法收集，這是正常的。各階段 README 有說明怎麼只跑已寫好的部分。

階段 2、3 的資料（`names.txt` 約 230KB、tiny shakespeare 約 1.1MB）不在 repo 裡，第一次用之前先下載：

```bash
uv run python 02_makemore/names_data.py
uv run python 03_gpt/shakespeare_data.py
```

## 搭配 Claude Code 使用

`CLAUDE.md` 是給 [Claude Code](https://claude.com/claude-code) 讀的協作規則，重點是：

- **核心實作由你自己寫**：Claude 不會寫出或補完核心邏輯，除非你明確說「幫我寫」或「給我答案」
- **卡住時用提示階梯**：一次給一級提示（先問問題 → 概念 → 指出位置 → 虛擬碼 → 答案）
- **Review 只指出問題**：說明錯在哪、為什麼、怎麼驗證，修正留給你
- Claude 可以直接做的：建環境、寫測試、周邊工具（資料下載、繪圖、logging）、解釋概念

第一次使用時，依自己的背景與硬體調整 `CLAUDE.md` 裡的時間與規模估計。不用 Claude Code 的話，`CLAUDE.md` 的
學習計畫與每階段的檢核項目也可以直接當作進度表。
