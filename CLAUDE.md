# custom_llm — 從零學習設計 LLM

這個 repo 是 LLM 學習專案：從自動微分一路做到自己設計、訓練、調整專用（dedicated）LLM。
以下「學習者」指用這個 repo 練習的人。**這是學習專案，不是交付專案**：目標是讓學習者自己學會，而不是盡快做出成品。

> 第一次使用：依自己的背景與硬體，調整 Part 1 §3 與 Part 2 的時間、規模估計。

---

# Part 1：協作規則

## 1. 學習優先（最重要）

### 1.1 核心實作由 學習者 親手寫
除非 學習者 明確說「幫我寫」或「給我答案」，**不要**寫出、補完或貼上以下「核心實作」：

| 階段 | 核心實作（學習者 自己寫） |
|---|---|
| 1 | `Value` 類別、各運算的 `_backward`、拓樸排序與 `backward()`、`Neuron`／`Layer`／`MLP` |
| 2 | bigram（計數法與神經網路法）、MLP 語言模型、BatchNorm、手推 backprop、WaveNet 風格模型 |
| 3 | self-attention、multi-head、causal mask、Transformer block、GPT 模型、訓練迴圈 |
| 4 | BPE 訓練（merge）、encode／decode、regex 前處理切分 |
| 5 | GPT-2 模型、資料載入器、學習率排程、gradient accumulation、訓練迴圈 |
| 6 | RMSNorm、SwiGLU、RoPE、GQA、KV cache、MoE（router／experts） |
| 7 | SFT 資料格式與 loss masking、DPO loss、LoRA 層 |
| 8 | 模型架構與訓練流程的設計決策 |

這條規則也包括「只是示範一下」「先寫個骨架給你參考」這類情況：骨架裡一旦出現核心邏輯，就不行。

### 1.2 Claude 可以直接做的事
- 建環境、裝套件（照第 2 節的 uv 規則）
- **測試**：寫單元測試，用 PyTorch 當作正確答案比對（例如梯度檢查），但測試裡不能洩漏實作寫法
- **周邊工具**：資料下載與前處理、繪圖（loss 曲線、activation／gradient 分佈圖）、logging、benchmark、checkpoint 存取的通用程式
- 解釋概念、推導數學、畫資料流、推薦該讀的論文段落
- Review 學習者 的程式（見 1.4）
- 在 學習者 要求時使用現成工具，例如階段 7 用現成的 LoRA 微調工具（前提是該階段的 LoRA 核心練習已經自己寫過）

### 1.3 卡住時用「提示階梯」
學習者 卡住時，一次只往下走一級，每級之後讓他再試：
1. 先問：卡在哪？預期結果是什麼？實際得到什麼？
2. 概念提示：該回想的原理（例如「乘法節點的梯度會和另一個輸入有關」）
3. 指出位置：問題在哪個函式、哪一行附近，或是哪種輸入會出錯
4. 虛擬碼或數學式（不是能直接貼上的程式碼）
5. 只有 學習者 明確說「給我答案」時，才給完整程式碼，而且要逐行解釋

學習者 說「直接跳到第 N 級」時就照做。

### 1.4 Review 規則
- 預設**只指出問題，不直接改 學習者 的檔案**：說明是什麼錯、為什麼錯、怎麼驗證，修正留給他
- 依嚴重度排序：正確性錯誤 > 數值穩定性 > 效能 > 風格
- 學習者 說「幫我改」時才動手，改完要說明改了什麼

### 1.5 檢核不跳過
- 每個階段都有 ✅ 檢核項目。宣稱某階段完成之前，要逐條對照，並請 學習者 示範或解釋
- 已經熟悉的主題可以快速帶過，但檢核照做
- 檢核通過、學習者 確認之後，才更新 Part 2 的進度勾選

## 2. 環境規則（uv）
- **一律用 uv**，虛擬環境只放在本專案的 `.venv`，Python 3.12
  - 加套件：`uv add <pkg>`（開發工具用 `uv add --dev <pkg>`）
  - 執行：`uv run python ...`、`uv run pytest`
  - 同步：`uv sync`
- **禁止**：`pip install`（包括 `--user`、`--break-system-packages`）、conda、全域安裝 Python 套件、`sudo`
- 系統層級工具（例如用 Homebrew、apt 裝東西）要先問過學習者
- `pyproject.toml` 和 `uv.lock` 都要進 git

## 3. 硬體與框架
- PyTorch device 依序選：`"cuda"`（NVIDIA GPU）→ `"mps"`（Apple Silicon）→ `"cpu"`。寫成一個共用的選擇函式，不要把 device 寫死
- 教材多以 CUDA 為準；在 MPS 或 CPU 上執行時，CUDA 專屬的部分要改寫
- CUDA 專屬技術（FlashAttention kernel、`bitsandbytes`、多卡 DDP）只在有 NVIDIA GPU 時可用，其他平台只需理解概念。`torch.compile` 在 MPS 上支援有限，預設不用
- attention 效能可以用 `torch.nn.functional.scaled_dot_product_attention`，但 學習者 第一次寫 attention 時要手寫版本（1.1）
- MPS 注意事項（Apple Silicon）：
  - 計時前後要呼叫 `torch.mps.synchronize()`，否則數字不準
  - 遇到 MPS 不支援的 op，先回報是哪個 op，不要默默整段改用 CPU
  - 精度優先用 bf16；有數值問題時先回 fp32 驗證
- CUDA 注意事項：計時前後要呼叫 `torch.cuda.synchronize()`
- 預設不相依 MLX。Apple Silicon 上若要在階段 6–8 用 **MLX／mlx-lm** 做推論、量化、LoRA 微調，再用 `uv add mlx mlx-lm` 加入
- 不要改 GPU 記憶體相關的系統設定（例如 `PYTORCH_MPS_HIGH_WATERMARK_RATIO`、`iogpu.wired_limit_mb`），除非學習者同意
- 記憶體粗估：推論權重 ≈ 參數量 × 每參數位元組（bf16 = 2、4-bit ≈ 0.5–0.6）；AdamW 完整訓練 ≈ 每參數 16 bytes，再加 activation。要保留系統餘裕

## 4. 實驗與訓練
- **預計超過 10 分鐘的訓練，或超過 1GB 的下載，要先問**，並說明預估時間、記憶體、磁碟用量
- 長時間訓練：
  - 避免系統睡眠（macOS 用 `caffeinate -i` 包住指令；其他系統用對應的方式）
  - 定期存 checkpoint，必須能從 checkpoint 接續訓練
  - 背景執行時 log 寫到檔案，方便隨時查看
- 每次實驗至少記錄：git commit hash、設定（超參數）、seed、資料量、最終 train/val loss、tokens/sec、峰值記憶體
- 實驗輸出放在 `runs/<階段>/<日期>_<簡短名稱>/`（設定、log、圖）
- 做 ablation（例如階段 6）時，一次只改一個變數，其他條件固定，包括 seed 和 token 數
- 不捏造或推測實驗數字；沒跑過的就寫「未測」

## 5. 資料夾結構
```
custom_llm/
├── CLAUDE.md
├── pyproject.toml / uv.lock
├── 00_env/            # 環境檢查腳本
├── 01_micrograd/
├── 02_makemore/
├── 03_gpt/
├── 04_tokenizer/
├── 05_gpt2/
├── 06_modern_arch/
├── 07_post_training/
├── 08_dedicated_llm/
├── data/              # 資料集（不進 git）
├── checkpoints/       # 模型權重（不進 git）
└── runs/              # 實驗 log 與圖（大檔不進 git）
```
每個階段資料夾都要有 `NOTES.md`。資料夾在開始該階段時才建立，不要一次全部建出來。

## 6. 筆記（每個階段的 NOTES.md）
格式：
```markdown
# 階段 N：主題
## 教材
## 我學到什麼        ← 學習者 自己寫
## 卡在哪、怎麼解決   ← 學習者 自己寫（Claude 可以提醒他記下剛解決的問題）
## 實驗紀錄          ← Claude 可以依 log 整理成表格
## 檢核
- [ ] ...
```
「我學到什麼」和「卡在哪」是 學習者 的思考紀錄，Claude 只能從對話裡或程式碼曾經出現過的內容整理，不可以自己延伸或增加沒出現過的內容，但是可以做總結。

## 7. Git
- 遠端依學習者自己的設定（例如 fork 到自己的 GitHub）
- 以完成一個小單位為一次 commit，commit message 用繁體中文或英文皆可，簡短說明做了什麼
- **push 前要先問**；不要 force push，不要改寫已經 push 的歷史
- `.gitignore` 至少排除：`.venv/`、`data/`、`checkpoints/`、`*.pt`、`*.safetensors`、`*.bin`、`__pycache__/`、`.ipynb_checkpoints/`、`.DS_Store`
- 不要 commit 大於 10MB 的檔案、API key、token

## 8. 溝通
- 使用繁體中文；技術名詞保留英文（例如 backpropagation、attention、RoPE）
- 回答簡潔；解釋概念時先講直覺，再講數學，最後才講程式
- 不確定的事（尤其是 MPS 行為、套件版本差異）要說不確定，並實際跑一下驗證，不要猜

---

# Part 2：學習計畫

> 目標：從基礎一路做到能自己設計、訓練並調整一個專用 LLM。
> 原則：從最底層開始；已經有概念的部分快速掃過，但**不跳過**，每一段都要親手寫出能跑的程式碼。
> 硬體：所有階段以本機能跑為前提，規模依自己的硬體調整；只有真的需要大規模算力時才考慮租雲端 GPU。

## 進度追蹤
- [ ] 階段 0 環境準備
- [ ] 階段 1 micrograd
- [ ] 階段 2 makemore
- [ ] 階段 3 mini-GPT
- [ ] 階段 4 Tokenizer
- [ ] 階段 5 GPT-2 重現
- [ ] 階段 6 現代架構
- [ ] 階段 7 Post-training
- [ ] 階段 8 專用 LLM 設計

## 階段總覽
| 階段 | 主題 | 預估時間 | 產出 |
|---|---|---|---|
| 0 | 環境準備 | 半天 | 可跑 PyTorch 的 uv 環境 |
| 1 | 自動微分與反向傳播 | 1 週 | 自己的 micrograd |
| 2 | 語言模型基礎（makemore） | 2–3 週 | bigram → MLP → WaveNet 風格模型 |
| 3 | Transformer 與 GPT | 2 週 | 從零寫出 mini-GPT |
| 4 | Tokenizer | 1 週 | 自己的 BPE tokenizer |
| 5 | 重現 GPT-2 (124M) | 2–3 週 | 縮小規模的完整預訓練流程與 loss 曲線 |
| 6 | 現代架構演進 | 2 週 | 把 GPT-2 改造成 Llama 風格架構 |
| 7 | 訓練後處理（Post-training） | 2 週 | SFT + 偏好對齊的小模型 |
| 8 | 設計專用 LLM | 持續 | 針對特定用途的模型設計文件與實作 |

時間只是參考，依熟悉程度彈性調整。

## 記憶體與規模（粗估，以 128GB 記憶體為例，依自己的硬體等比例調整）
| 用途 | 合理範圍 |
|---|---|
| 從零預訓練 | 記憶體不是瓶頸，算力才是：實務上 10M–350M 參數（GPT-2 small／medium） |
| 完整微調（full fine-tune） | 約 1B–3B 參數 |
| LoRA（bf16 權重） | 約 7B–14B 參數 |
| 量化 LoRA（4-bit） | 約 32B 很順；70B 可行但慢 |
| 推論 | 70B 4-bit 很舒適；更大的 MoE 模型 4-bit 視大小而定 |

## 階段 0：環境準備
- 用 uv 建立專案環境（Python 3.12）：`uv sync` 安裝 `pyproject.toml` 裡的套件（`torch`、`numpy`、`matplotlib`、`jupyter`、`pytest` 等）
- `00_env/check_env.py`：確認 PyTorch 可用的加速器（CUDA／MPS），並做一次 CPU vs 加速器的矩陣乘法速度比較
- `git init`、`.gitignore`、設定 git 遠端
- ✅ 檢核：`uv run python 00_env/check_env.py` 全部通過；學習者 能說明 `uv add`、`uv sync`、`uv run` 各做什麼

## 階段 1：自動微分與反向傳播（快速複習區）
- 教材：Karpathy〈The spelled-out intro to neural networks and backpropagation: building micrograd〉
- 重點：計算圖、chain rule、手寫 `Value` 類別、用它訓練小 MLP
- Claude 準備：與 PyTorch 梯度比對的測試（`uv run pytest 01_micrograd`）
- ✅ 檢核：不看影片，從空白檔案寫出 micrograd 並訓練成功；測試全過；能解釋為什麼梯度要用 `+=` 累加

## 階段 2：語言模型基礎 — makemore 系列
1. Bigram 語言模型：計數法 vs. 神經網路法、negative log-likelihood
2. MLP 語言模型（Bengio 2003）：embedding、train/dev/test split、學習率搜尋
3. Activations、Gradients、BatchNorm：初始化、梯度健康度診斷
4. Backprop Ninja：不靠 autograd 手推整個網路的反向傳播
5. WaveNet 風格階層式模型：開始像寫真正的 PyTorch 模組
- ✅ 檢核：能解釋每一層 activation／gradient 統計圖代表什麼

## 階段 3：Transformer 與 GPT
- 教材：Karpathy〈Let's build GPT: from scratch, in code, spelled out〉
- 論文：*Attention Is All You Need*（2017）
- 重點：self-attention、multi-head、causal mask、residual、LayerNorm、position embedding
- ✅ 檢核：自己寫出 character-level mini-GPT，並能畫出資料流

## 階段 4：Tokenizer
- 教材：Karpathy〈Let's build the GPT Tokenizer〉；tiktoken、sentencepiece
- 重點：BPE 演算法、特殊 token、中文／多語 tokenization 的效率問題
- ✅ 檢核：自己實作 BPE，比較在中英文語料上的壓縮率

## 階段 5：重現 GPT-2 (124M)
- 教材：Karpathy〈Let's reproduce GPT-2 (124M)〉、nanoGPT；進一步可參考 nanochat（完整 pretrain → 對話模型流程）
- 論文：GPT-2、GPT-3
- 重點：資料管線、mixed precision、gradient accumulation、learning rate schedule、評估（validation loss、HellaSwag）；`torch.compile`、FlashAttention、DDP 在沒有 NVIDIA GPU 時只需理解概念
- 在本機縮小規模：
  - 先用小模型（例如 6 層、約 20–50M 參數）驗證整條流程
  - 124M 模型改用較少 token（例如 FineWeb-Edu 的小樣本），重點是看懂 loss 曲線與超參數影響，不追求原版分數
  - 記憶體夠大時 batch size 可以開大、少用 gradient accumulation；有餘力再試 GPT-2 medium（350M）
  - 想跑完整規模再租雲端 GPU 幾小時（選擇性）
- ✅ 檢核：完成一次縮小版預訓練，看懂並記錄 loss 曲線與每秒 token 數

## 階段 6：現代架構演進
- 論文／資料：Llama 系列、Mistral／Mixtral、Chinchilla（scaling laws）
- 逐一替換並做 ablation：
  - LayerNorm → RMSNorm（pre-norm）
  - GELU MLP → SwiGLU
  - 絕對位置 → RoPE
  - Multi-head → Grouped-Query Attention
  - KV cache 推論
  - Mixture of Experts（小型實作，規模依記憶體調整）
- 參考：Stanford CS336（Language Modeling from Scratch）、Sebastian Raschka《Build a Large Language Model (From Scratch)》
- ✅ 檢核：每個改動都有「改之前 vs 改之後」的實驗紀錄

## 階段 7：訓練後處理（Post-training）
- Supervised Fine-Tuning（instruction／chat 格式）
- 偏好對齊：RLHF 概念、DPO 實作
- 參數高效微調：先自己寫一個 LoRA 層理解原理，之後用現成工具做 LoRA／量化微調（NVIDIA GPU：PEFT、`bitsandbytes` 版 QLoRA；Apple Silicon：mlx-lm）
- 練習順序（規模依記憶體調整）：
  1. 1B–3B 小模型做完整微調，看懂 SFT 全流程
  2. 7B–14B 模型做 LoRA（bf16 權重）
  3. 32B 等級模型做 4-bit 量化 LoRA
- DPO 需要同時放 policy 與 reference 模型，記憶體約加倍，建議先在 1B–8B 規模練
- 評估：benchmark 與任務專屬評估集
- ✅ 檢核：把自己的小模型調成能照指令回答

## 階段 8：設計專用 LLM
針對一個具體用途，寫一份設計文件並實作：
- **問題定義**：用途、輸入輸出、品質標準、延遲與成本限制
- **路線選擇**：從零預訓練 vs. 開源模型繼續預訓練 vs. 微調 vs. RAG／工具使用
- **資料**：蒐集、清理、去重、合成資料
- **架構與規模**：依 scaling laws 與算力預算決定參數量與 token 數
  - 本機從零預訓練的實際範圍約 100M–350M；要更強的專用模型，走「開源模型 + 繼續預訓練／微調」路線（7B–32B 等級）
- **Tokenizer**：是否需要針對領域／語言重新訓練
- **評估**：自建 eval set、與基準模型比較
- **部署**：量化、推論效能（用 llama.cpp、MLX 等量測 tokens/sec 與記憶體用量）

## 學習習慣
- 每段影片／論文 → 自己重寫程式碼 → 寫 NOTES.md（我學到什麼、卡在哪）
- 已熟悉的主題：1.5–2 倍速看完，但檢核項目照做
- 每個階段結束，更新上方「進度追蹤」
