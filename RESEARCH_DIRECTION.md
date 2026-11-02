# Outcome-based RL：研究方向與決策紀錄

更新日期：2026-11-03。本文保存使用者已確認的研究方向，供同專案後續 session 延續工作。
本文區分「已確認方向」、「目前實作」與「待決定設計」；研究問題的確認不代表所有候選實驗已定案或已實作。使用者後續的明確指示優先，方向改變時應同步更新本文。

## 1. 已確認的定位與名稱

本專案研究 **Outcome-based RL 如何改變模型的解題、判斷與修正能力**，並比較 RLOO、GR-REINFORCE、GRPO 的行為與學習軌跡。

- 專案統一使用 **Outcome-based RL** 描述目前依最終答案正確性提供任務 reward 的訓練方向。
- 目前透過規則 verifier 判定最終 boxed 答案，屬於使用可驗證 outcome reward 的 RLVR。Outcome-based 描述監督訊號的位置；RLVR 描述 reward 可驗證的性質，兩者不是所有情境下的同義詞。
- 目前任務 reward 為最終答案精確相等得到 1，否則為 0；沒有中間推理步驟的 process reward。KL 是正則化，不是對推理步驟正確性的評分。
- 研究目的是理解方法造成的變化；沒有固定最佳模型交付、超參數搜尋或 best-model selection 的目標。
- 三個研究問題均已獲使用者確認。研究的確切假說、實驗規模與分析方法仍須逐步具體化。

## 2. 已確認的三個研究問題

### RQ1：解題——RL 讓模型更常答對，還是能解出更多題？

比較三方法的 greedy accuracy、sampled pass@k、逐題得失與回答長度，區分：

1. 單次作答的成功機率是否提高。
2. 在固定多次取樣預算下，能解出的題目覆蓋率是否增加。
3. 格式遵循、長度、截斷與推論成本是否同時改變。

這是共同基礎，供後續判斷與修正實驗解釋結果。Greedy accuracy 與 sampled pass@1 應分開報告；有限 k 下沒有抽到正解，不能證明原模型完全不具備解題能力。

### RQ2：判斷——解題能力改善時，辨識正誤的能力是否一起改善？

保存同一批候選解答，讓 RL 前後模型在**看不到標準答案**的情況下判斷正誤、選擇答案，再由獨立 verifier 評分。

固定候選解答是這個問題的核心控制：避免把「產生了更好的答案」誤認為「更會辨識好答案」。需要區分候選解答的生成模型與評判模型，並保留其來源。

可探索各方法互相評判彼此解答，觀察是否偏好自身來源的錯誤；候選池如何組成、順序如何控制、評判 prompt 與評分方式仍待決定。

目前 verifier 只能建立**最終答案正誤**的標籤，不能驗證整段推理。若要研究模型是否能辨認推理錯誤，需另有經核對的推理過程標註，不能直接將答案正確當作推理正確。

### RQ3：修正——模型看過自己的第一次回答後，能否有效改善？

在接近的計算預算下，比較「兩次獨立作答」與「第一次作答後檢查、修正」。除了最後答對率，也記錄錯→對、對→錯，確認修正是否有淨收益。

- 起始探索條件為沒有外部正誤回饋的自我檢查與修正。
- 提供正誤提示屬於另一個實驗條件，需要與無回饋版本分開。
- 兩次獨立作答如何選出最後提交答案，需事先定義；不能用 gold answer 協助選擇，再當作模型實際表現。
- 修正 prompt、回合數、停止規則與計算預算仍待決定。

先觀察 Outcome-based RL 後是否自然出現較好的修正能力；是否另行訓練 multi-turn correction policy，尚未決定。

## 3. 目前 repo 的實作基線

下表是本文更新時的狀態；後續實作完成時應更新，不能將待做功能寫成既有能力。

| 項目 | 目前狀態 |
| --- | --- |
| 訓練方法 | RLOO、GR-REINFORCE、GRPO 均已接入 CLI，無 TRL 依賴 |
| 共用架構 | `rl_core.py` 執行 rollout → method 準備固定資料 → 重複 gradient estimation／core 更新參數；GRPO 預設每批兩次，另兩方法一次 |
| 方法模組 | 三方法均使用 `prepare_rollouts` 與 `estimate_gradients`；method 持有 advantage／objective，optimizer 由 core 執行 |
| RL 前模型 | 固定的 OLMo-2-0425-1B-SFT 與 Granite-3.3-2B-Instruct；是 SFT／instruct 起點，不是未經 instruction tuning 的 pretrained base |
| 資料 | 原始 MATH train/test 分離；固定 boxed-numeric cohort：5,589 train、3,686 test |
| 固定評估 | 同一份分層抽樣的 512 題，greedy decoding；亦可執行完整 3,686 題 |
| 現有 metrics | 最終答案 exact accuracy、boxed-output rate、numeric-boxed-output rate；保存逐題 completion 與正誤 |
| 現有分析 | checkpoint 曲線、計數與端點差異；尚無 repeated sampling、pass@k、seed 彙整或不確定性分析 |
| RQ2／RQ3 | 尚無專門的判斷與修正評估流程 |

512 題是本專案固定 sample，**不是標準 MATH-500**。目前結果的直接適用範圍是 boxed-numeric cohort 與既定 prompt／解碼預算，不能直接稱為完整 MATH 或一般數學能力。

詳細的資料契約、模型 pins、objective 與操作方式見 [README.md](README.md) 與實作。程式正確性測試不等於模型能力實驗結果。

## 4. 已確定、後續工作應延續的設定

- PDF 與外部論文皆為參考，**不是專案規格**。包括 `hw4.pdf` 與 `llm_rl_default_final_project.pdf`；其中的模型、cohort、prompt、reward、loss reduction、logging 或實驗要求不會自動套用。
- 保留目前固定資料、prompt、verifier 與資料切分。任何未來泛化或變體測試應有獨立定義與紀錄。
- 保留無 reward 差異的 groups；其出現屬於模型與資料的觀測現象，不預設要過濾或重抽。Raw reward 相同也不代表含 KL 的所有方法都沒有 gradient。
- 保留未產生 EOS、達生成上限的 completions，沿用目前評分與訓練行為。
- RLOO 的 KL 係數為 0.01，GR-REINFORCE 與 GRPO 為 0.05；數值與 KL 放置、reduction 不同，不能視為同等正則化強度。
- Rollout iterations 與 optimizer steps 分開計數；learning-rate warmup 以 optimizer steps 計算。
- Gradient overflow／non-finite loss 或 gradient 應中止；不能把 AMP 跳過更新計成成功 optimizer step。
- 目前訓練 sampling 為 temperature=1、top-p=1、top-k=0，配合 raw-policy log-probs。未來若改 sampling，需要重新檢查 objective 與記錄，不能默默沿用原假設。
- Resume 用於恢復同一實驗中斷前的模型、optimizer、RNG 與進度；不是跨方法續訓。

## 5. 評估設計的原則與候選觀測

本節是規劃原則與候選方案，**不是已定案的完整 protocol**。

| 觀測 | 解釋與限制 |
| --- | --- |
| Greedy accuracy | 固定解碼的答對率；不等同隨機取樣成功率 |
| Sampled pass@1 | 單次隨機作答的平均成功率；需記錄 sampling 設定 |
| pass@k | k 次候選中至少有一個正解的成功率；使用 gold 作事後診斷，不能當作模型已能選出正解 |
| Majority-vote accuracy@k | 不靠 gold 選擇答案的投票表現；需定義分數等價、無效輸出與平手規則 |
| 逐題得失 | 分開保存錯→對、對→錯，避免淨分數掩蓋行為變化 |
| 長度與成本 | 分開觀察正確／錯誤回答的 tokens、停止原因、截斷與時間；更長不直接代表更好的推理 |
| 判斷能力 | 對固定候選的正誤判斷與答案選擇；需避免只靠某一類標籤較多得到高分 |
| 修正能力 | 同時報告初始與最終 accuracy、錯→對率、對→錯率，並說明各自分母與淨改善 |

比較時應注意：

1. 三方法首先是在**同一模型內**比較。兩個起點同時有規模、預訓練、SFT、tokenizer 與 prompt 差異，不能將跨模型差異單獨歸因於架構。
2. 目前 RLOO 與 GR-REINFORCE 同時有 advantage、sequence-sum／mean、KL 形式與係數差異。整體比較回答 recipe 差異；單一機制的因果解釋需有針對性的消融。
3. 相同 rollout 數不是相同 optimizer steps，也不是相同算力。比較應保留生成量、更新次數與實測成本；修正與評判也要計入額外 prompt 處理及輸出成本。Tokens 是成本指標，不自動等於跨模型相同 FLOPs。
4. Training seeds 的變異與題目取樣的不確定性分開分析。可考慮逐題配對分析及多 seed，但數量與統計方法尚未定案。
5. 512 題上的少量題目差異不宜過度解讀。若根據這批 test 結果修改方法、挑選條件，應標示為探索；新假說可在未用於決策的題目上確認。
6. 最終答案正確、推理過程正確、對題目變體有泛化能力，是不同層次的證據。推理抽查或受控變體是候選補充，尚未定案。

## 6. GRPO 的研究與工程接點

GRPO 已實作。依使用者 2026-11-03 指示，模糊細節優先參考講義與作業 GitHub，但保留本專案既有設定；並未引入作業任務、模型或交付要求。

值得檢驗的機制問題仍是：**同一批 rollout 多更新一次，是否增加效益，付出多少成本？**

若更新起點 policy=old policy、ratio=1，且 advantage、KL、mask 與 reduction 對齊，第一次 gradient 可與 GR-REINFORCE 相同。微型模型測試已驗證此性質；兩者的 policy loss 數值不同，多次更新後也不等價。

目前採用的 GRPO 配方：

- 同一 BG rollout batch 預設完成兩次完整 gradient accumulation／optimizer update；`--num-iterations 1` 可做單次更新比較。RLOO 與 GR-REINFORCE 限一次。
- Advantage 沿用 GR-REINFORCE 的 population std、ε=1e-6、clip ±5，無第二次標準化。
- 固定 old-policy token log-probs；policy log-ratio 先 clamp ±20，再取 exp，PPO clip ε=0.2。Policy term 為每條 completion token mean，再平均 completions。
- β=0.05，KL 共用 GR-REINFORCE 的 sampled `expm1(delta)-delta`，delta clamp ±20；以全批所有 ranks 的有效 completion tokens 平均，不加 current／old ratio 權重或 entropy bonus。重用資料時不宣稱這是精確 current-policy KL。
- Old-policy log-probs、reference log-probs、rewards、advantages 每批只準備一次並固定，逐次重算 current-policy log-probs。Reference 固定為停用 adapter 的 RL 前模型。
- 保留固定 prompt-major microbatch 順序。與作業逐 minibatch mean 的平均不同，使用全批 KL token mean，維持不等長回答在 microbatch／rank 切分下的同一 objective。
- Core 統一執行梯度檢查、裁剪與更新。每次更新均留紀錄；history 每 rollout 一列，flat metrics 為最後一次更新，`updates` 保留所有更新結果。
- Checkpoint 僅存整批更新完成的狀態；中途失敗從前一完整 checkpoint 重跑，沒有 batch 中途續訓。Warmup 繼續依 optimizer steps 計算。

此配方解決先前尚未定案的 GRPO 實作細節；正式研究比較規模、資源預算與消融組合仍待決定。測試驗證程式與公式，不代表已完成正式 GPU 模型能力實驗。

## 7. 延伸方向與尚未決定的事項

使用者對「解釋 RL 方法如何改變行為」與「探索新機制」較有興趣，但尚未鎖定唯一成果形式。

- **Test-time scaling**：可直接延伸 RQ1，比較不同取樣／計算預算下的方法排名。尚未選定取樣次數與配置。
- **Self-evaluation**：是 RQ2 的探索方向，先用外部 verifier 檢查判斷可靠性。
- **Self-reward**：尚未決定採用。讓模型擔任評判與把自評用作 training reward 是不同實驗；不能因開始 RQ2 就自動改掉現有 reward。評估自評時，gold 僅供事後評分，不能提供給評判模型。
- **Multi-turn**：RQ3 已確認要研究修正問題；專門的 multi-turn RL training、回合數、外部回饋或工具使用尚未定案。
- **泛化與推理過程檢查**：經核對的題目改寫、數值變體、其他 benchmark 與盲評推理抽查皆為候選，不自動取代固定 cohort。
- **資源與規模**：GPU 型號／數量／時數、training seeds、每題取樣次數、重複取樣的 checkpoints、統計檢定與判斷／修正 prompts 尚未決定。先前提過每題 16 次取樣、三個 training seeds 等數量，皆只是建議。

建議推進順序是先建立 RL 前模型的觀測，使用已實作的三方法與必要紀錄進行解題比較；以保存的候選解答開始 RQ2 小型診斷，之後推進 RQ3。這是工作順序建議，使用者可調整，不能據此推定已授權執行所有實驗。

## 8. 討論時使用的研究參考

下列來源提供問題與設計動機，不構成必須重現其模型、資料或超參數的要求。不同論文的結果依賴各自設定，不能當作本專案已得到的結論。

- [DeepSeekMath](https://arxiv.org/abs/2402.03300)：GRPO objective 與同批 rollouts 多次更新的原始參考。
- [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837)：以低／高 k 的 pass@k 差異討論 sampling efficiency 與覆蓋率。
- [Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs](https://arxiv.org/abs/2506.14245)：加入推理過程的評估，對 RLVR 的能力改善提出不同實證結果。
- [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783)：normalization、回答長度與訓練解釋的參考。
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)：取樣與答案整合的參考。
- [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314)：推論預算、題目難度與計算配置的參考。
- [Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798)：無外部回饋的自我修正限制。
- [Training Language Models to Self-Correct via Reinforcement Learning (SCoRe)](https://arxiv.org/abs/2409.12917)：專門訓練 multi-turn correction 的參考。
- [GSM-Symbolic](https://arxiv.org/abs/2410.05229)：受控題目變體與評估穩健性的參考。
