# Task 2: Machine Learning — 銀行客戶流失預測

使用 `train.csv` 訓練模型,預測 `test.csv` 中每位客戶的流失(`Exited`)機率,
並報告模型的預測能力。

## 快速執行

```bash
pip install -r requirements.txt
python main.py
```

- Python 3.10+(開發環境為 3.11)
- **macOS 使用者**:XGBoost 需要 OpenMP 執行庫 → `brew install libomp`
- 執行時間約 1 分鐘;所有結果皆可重現(固定 `random_state=42`)

## 檔案說明

| 檔案 | 內容 |
|---|---|
| `main.py` | 一鍵執行的完整流程:讀檔 → EDA 圖表 → 訓練 → 評估 → 輸出 CSV |
| `churn_analysis.ipynb` | 探索分析 notebook,含完整的 EDA 敘述與推理過程 |
| `requirements.txt` | 相依套件 |
| `train.csv` / `test.csv` | 題目提供的資料(165,034 / 110,023 筆) |

### 輸出檔(由 `python main.py` 產生)

| 輸出 | 內容 |
|---|---|
| `predictions.csv` | `test.csv` 原始欄位 + **`Exited_Prob`**(預測流失機率)+ **`Exited_Pred`**(門檻 0.35 下的二元分類) |
| `metrics.csv` | ROC AUC、F1、Precision、Recall、Accuracy 與決策門檻 |
| `confusion_matrix.csv` | 含行列標籤的 2×2 混淆矩陣 |
| `plots/*.png` | 7 張圖:EDA(01–04)與模型評估(05–07) |

> **評估數字從哪裡來?** `test.csv` 沒有標籤,無法用來評估模型。
> 所有指標皆計算於**保留驗證集**:從 `train.csv` 分層切出的 20%
> (33,007 筆),模型訓練過程完全沒有看過這些資料。

## 方法說明

### 1. EDA 發現(詳見 notebook 與 `plots/01–04`)

- **類別不平衡** — 流失率 21.2%。單看 Accuracy 會誤導,
  因此評估改用 ROC AUC、F1、Precision / Recall。
- **Age** — 流失客戶的年齡明顯偏高。
- **NumOfProducts 非單調** — 持有 1 個產品流失率 35%、2 個僅 **6%**、
  3–4 個卻高達 **88%**。單一線性係數無法表達這種 U 型關係。
- **解釋變數間相關** — 例如 `Geography_Germany` × `Balance` 相關係數 0.54
  (德國客戶幾乎都持有正餘額)。
- **弱特徵** — `CreditScore`、`EstimatedSalary`、`Tenure`、`HasCrCard`
  在流失/留存兩群的分布幾乎重疊。

### 2. 模型選擇 — 由資料特性推導

非線性/非單調的 pattern、混合型態且彼此相關的特徵、無缺失值、
16 萬筆的中型表格資料——這正是梯度提升樹(gradient-boosted trees)
最擅長的場景,因此選擇 **XGBoost**:

- 樹模型自動捕捉 NumOfProducts 的 U 型關係與特徵交互作用,
  不需手工做虛擬變數或分箱。
- 對特徵尺度與共線性不敏感,無須標準化。
- 原生支援類別特徵(`enable_categorical=True`)——`Geography` 與
  `Gender` 直接以 `category` 型態餵入。

超參數保持適度且固定
(`n_estimators=300, learning_rate=0.1, max_depth=6`),不做重度調參——
驗證集 AUC(0.889)已接近此資料集的實務天花板。

### 3. 驗證架構

1. 對 `train.csv` 做 **分層 80/20 切分** —— 分層抽樣讓兩邊都維持
   21.2% 的流失率。
2. 以 80% 訓練;**所有報告的指標與曲線皆來自 20% 驗證集**。
3. 學習曲線(`plots/05`)逐輪監控訓練/驗證 logloss,確認沒有過擬合。

### 4. 決策門檻

模型輸出的是機率。為了同時提供二元分類結果,將門檻在 0.05–0.95 之間
掃描,選擇**驗證集上 F1 最大**的位置 → **t = 0.35**(`plots/07`)。
最佳門檻低於 0.5 的原因是類別不平衡:降低門檻犧牲少量 precision,
換取少數類(流失)明顯更高的 recall。

### 5. 最終預測

直接**沿用於 80% 訓練集上訓練、並已在驗證集上完成評估的同一顆模型**,
對 `test.csv`(110,023 筆)輸出 `Exited_Prob`——因此本文報告的所有
評估指標,描述的正是產出這份預測的模型本身(被評估的模型 = 產出
預測的模型,指標無需外推)。

## 結果(驗證集,33,007 筆)

| 指標 | 數值 |
|---|---|
| ROC AUC | **0.8885** |
| F1 @ t=0.35 | **0.6662** |
| Precision | 0.6616 |
| Recall | 0.6708 |
| Accuracy | 0.8578 |

混淆矩陣 @ t = 0.35:

| | Pred: Stayed | Pred: Exited |
|---|---|---|
| **True: Stayed** | 23,627 | 2,396 |
| **True: Exited** | 2,299 | 4,685 |

**解讀:** 相對 21.2% 的流失底率,precision 0.66 約為 **3 倍提升**——
模型標記的客戶中,每 3 位有 2 位是真流失者,同時抓到了全部流失者的
67%。剩餘誤差主要來自特徵所含資訊的極限(缺少客訴、交易活躍度等
行為訊號),而非模型容量不足。

## 圖表說明

| 檔案 | 內容 |
|---|---|
| `01_target_distribution.png` | 類別不平衡(流失率 21.2%) |
| `02_numeric_distributions.png` | 數值特徵在流失/留存兩群的分布(Age 區分度最高) |
| `03_churn_rate_by_category.png` | 各類別的流失率(NumOfProducts 呈 U 型) |
| `04_correlation_heatmap.png` | 特徵相關係數(含目標變數) |
| `05_learning_curve.png` | 訓練 vs 驗證 logloss 學習曲線 |
| `06_roc_pr_curves.png` | ROC 曲線與 Precision-Recall 曲線(驗證集) |
| `07_threshold_sweep.png` | F1 vs 決策門檻;最佳值 0.35 |

