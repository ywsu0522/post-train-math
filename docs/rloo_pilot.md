# 原始 SFT → correctness-only RLOO，Colab T4 pilot

## 研究主線

`codex/boxed-numeric-rl` 直接從 **4339742** 建立：保留 boxed-numeric-v1 cohort、
verifier、RLOO/GRPO/Dr.GRPO backend 與 reasoning probe。`master` 保持原始 RLVR baseline。
Round 2–4 的格式訓練、FSM、repetition/ngram 控制、continued-SFT 與 curriculum gate
不在新分支的祖先歷史或執行流程中。舊實驗僅留本機 recoverable bundle。

起始權重依使用者選擇採用最初的一輪 SFT，沒有再接 continued-SFT。
須區分程式與權重來源：本機這份原始 SFT 的 `run_config.json` 記錄訓練 commit 是
`806fe2f`；無法把現存權重宣稱為在 `4339742` 上訓練。
本輪使用 `4339742` 既有的簡短 boxed prompt，重新評估 baseline；不恢復舊 grammar prompt。
原始權重 SHA256：`40272e18f4b5e0491ebe9e2daaa06b892c143a0093a2ac66ac311203cd180bf5`。

## 為什麼先 RLOO

**先 RLOO，Dr.GRPO 留作後續對照，這輪不補 SFT。**
RLOO 提供簡單的 leave-one-out baseline，適合先量測現有模型從 terminal reward
實際獲得多少學習訊號。不預設必須先達到某個 boxed rate 才能開始 RL。

| 方法 | Repo 鎖定 TRL 1.12.0 的設定 | 判斷 |
| --- | --- | --- |
| RLOO | leave-one-out；不正規化 advantage；sequence log-prob 是 token log-prob 總和 | 本輪 baseline |
| 普通 GRPO | group mean + reward std scaling；每條回覆依自身長度正規化 loss | 第一輪多了 reward/length weighting 的解讀變因 |
| Dr.GRPO | group mean；無 reward std scaling；固定 max completion length 正規化 | 值得比較，但沒有本模型上優於 RLOO 的實驗證據 |

固定 `G` 時，RLOO advantage = `G/(G-1) * (reward - group_mean)`。
本輪 `G=4`、`num_iterations=1`、`beta=0`；RLOO 不含普通 GRPO 的 reward-std
weighting 或 per-response length average。RLOO 與 Dr.GRPO 的 importance ratio
粒度與 loss 尺度仍不同，不能因這個公式就宣稱兩者完全相同或同 LR 必然公平。
三者都不需 critic；選 RLOO 不是宣稱它比 Dr.GRPO 更省 T4 記憶體。

三者在全錯／全對 group 都沒有相對 advantage 訊號。直接換 Dr.GRPO 不能
憑空解決全零 rewards；這輪先保存真實 rollout 與 mixed-group 比例來確認瓶頸。
參考：[TRL RLOO](https://huggingface.co/docs/trl/rloo_trainer)、
[RLOO 論文](https://arxiv.org/abs/2402.14740)、
[Dr.GRPO 分析](https://arxiv.org/html/2503.20783v1)。論文不是本模型成效的保證。

## Dataset 與 reward

沿用專案 pinned `DigitalLearningGmbH/MATH-lighteval`，revision
`f06834690385b29df31ccc717250746a3ba0322b`，以及既有 train/dev split。

- RL 從 **全部 numeric-eligible MATH train 題**均勻抽樣，保留所有 type 與原始 level 標籤。
  不按參考解答長度、模型成功率或是否出 box 選題。
- 只做問題去重、排除與 dev 重複的問題，以及 1024-token prompt 預算檢查。
  本機核對剩 **5,024 題**（也保留原資料 2 題未知 level），實際 train 數量與固定題目
  保存在 `dataset_manifest.json` / `selection.json`。
- 模型只看到 problem 與原有 boxed prompt；solution 只用來解析 gold。
  Train 題可能已用於 SFT，這是從 SFT 接 RL；不是「SFT 未見題」測試。
- 固定抽 64 題 numeric dev 做 greedy 評估；其中 16 題每題另採樣 4 次。
  每個模型共 128 回覆；SFT / RL 同題、同 seed、同生成設定。
  Dev 曾用於 SFT 監測，不是 untouched test。本輪不讀 test、不用 dev 更新或挑 checkpoint。

Reward 維持 cohort 的唯一驗證器：最後一個 box 是合法 numeric 且等於 gold → **1**；
其餘 → **0**。沒有 format-only reward、推理過程 reward、輸出修補或強制解碼。
合法 box 比例僅是觀察指標，不是準入門檻；最終答對不等於證明中間推理正確。

## 固定 budget

| 項目 | 值 |
| --- | --- |
| Base / adapter | pinned OLMo-2-0425-1B / 原始一輪 SFT LoRA |
| RL | RLOO，50 total optimizer steps，包含 smoke 的 2 steps |
| 每次更新 | 2 個 prompt groups × 4 completions = 8 回覆 |
| Microbatch / accumulation | 1 / 8，single T4 |
| LR / scheduler / grad clip | 1e-6 / linear 50-step schedule / 1.0 |
| Rollout | 最多 512 new tokens；temperature=1、top-p=1、top-k=0 |
| KL / iterations / seed | beta=0 / 1 / 83 |
| T4 | FP16、gradient checkpointing、AdamW；無額外 critic/reference model |
| Checkpoints | smoke step 2、其後每 10 steps；保留最近 2 個 |

預計 400 個 training completions、前後評估合計 256 個 completions。
這只是從全 train pool 抽取 100 個 prompt groups 的可行性測試，不是跑完整個資料集。
中斷後未保存的更新可能重跑，attempt logs 分開記錄，不把重跑算成新增有效更新。

## Colab 怎麼跑

1. 將提供的 `initial-sft-v1-adapter.zip` 放進 Drive 的 `post-train-math-runs` 資料夾。
   ZIP 是從本機原始 `post-train-math-backup/olmo2-1b-lora-sft-v1/final-model` 打包；
   只含 adapter/tokenizer 與來源紀錄，沒有 base weights 或 optimizer。
2. 開啟 [`rloo_pilot_colab.ipynb`](rloo_pilot_colab.ipynb)，runtime 選 **T4 GPU**。
   第一格可修改 `ADAPTER_SOURCE`（支援提供的 ZIP 或原始 final-model 目錄）。
3. 由上往下執行。順序是 setup → prepare → baseline → smoke → train → evaluate → package。
   Baseline / evaluate 都只做推論；smoke / train 才是 RL。
4. 回傳 **analysis_bundle.zip**。成功或失敗都交同一份；可另附有輸出的 notebook。
   完整 checkpoint / `rl/final-model` 保留在 Drive，不必上傳模型權重。

Drive 預設結果目錄：`post-train-math-runs/numeric-rloo-original-sft-v1`。
第一次先核對原始 adapter weights hash，再下載 pinned base/data；prepare 會再核對
base provenance 與 tokenizer，避免花完 baseline GPU 時間才發現模型來源不符。

## 中斷、進度與分析

Session 到期後重新開 T4、重跑 setup 到 prepare，沿用相同 RUN_ROOT。
可重跑所有 stages：已完成的評估逐筆略過，RL 從最新完整 checkpoint 繼續。
Smoke 和 train 共用原來 50-step scheduler，總 budget 是 50，不是 52。
Checkpoint 完成標記要求 weights、optimizer、scheduler、RNG，T4 還要求 FP16 scaler。
如果剛好在最後 export 中斷，會從已完成的最終 checkpoint 恢復，不再多做更新。

命令串流 stdout/stderr；安靜時有 heartbeat，但那只證明程序尚未退出。
實際進度包括 eval 完成筆數、rollouts scored、optimizer step complete、checkpoint saved。
每步紀錄 reward、mixed groups、生成 tokens、GPU peak、gradient norm、entropy 與耗時。
有 NaN/Inf metrics 會停止並留下錯誤。若 OOM，先交 bundle，不在相同 run 中修改參數。
同一 RUN_ROOT 不可由兩個 sessions 同時寫入，改 code/weights/budget 需新目錄。

每個 stage 結束或可捕捉錯誤都會打包；Notebook 最後一格可在全新 session 單獨
mount Drive 並打包，不需下載模型或安裝訓練套件。
Bundle 包含設定、source snapshot、固定題目、原始生成/token IDs、reward、訓練紀錄、
checkpoint events 與前後 paired gains/losses。

判讀以 **同題 dev correctness 變化**為主，配合 sampled success@4、mixed groups、
生成長度與截斷情況。Greedy 64 題與 sampled 16 題的結果分開報告；不能把重複題
混成獨立樣本或完整 MATH 準確率。Training reward 上升不等於泛化改善。
RLOO loss 接近零也可能仍有非零 gradient；須合看 grad norm 與參數是否更新。

若幾乎全是零訊號 groups，先回報這個全 cohort baseline 的 reward sparsity，
不自動加格式 SFT 或啟動 curriculum。50 steps、單 seed、小型 dev 的變動是方向性
證據；後續先擴大評估，再決定更多 RL 或受控的 Dr.GRPO 對照。
