# Task 3: Computer Vision — MNIST 手寫數字分類（CNN）

以 **ImageNet 預訓練 ResNet-18 遷移學習** 完成 MNIST 數字分類，並針對題目要求的三個關鍵領域——**Modeling、Optimization、Handling Class Imbalance**——設計完整的對照實驗。

完整的程式碼、逐步解說與圖表都在 [`MNIST_image_classification.ipynb`](MNIST_image_classification.ipynb)，本 README 為結果摘要。

---

## 執行方式

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter nbconvert --to notebook --execute --inplace MNIST_image_classification.ipynb
```

- MNIST 資料與 ImageNet 預訓練權重會自動下載（`torchvision`）
- 裝置自動選擇：Apple-Silicon MPS → CUDA → CPU
- 全域 seed = 42；因 GPU 運算的非確定性，重跑數字可能有 ±0.1–0.3% 的浮動

## 方法設計

### 1. 資料策略：三個角色分離的資料集

| 資料集 | 大小 | 分布 | 用途 |
|--------|------|------|------|
| Training | 48,000 → 長尾降採樣後 11,753 | **長尾（100:1）** | 擬合模型權重 |
| Validation | 12,000（8:2 分層切分） | 平衡 | 模型選擇與 early stopping |
| Test（官方） | 10,000 | 平衡 | 只做最終評估 |

MNIST 原始資料幾乎平衡（最大/最小類別比僅 1.24:1），無法展示不平衡處理能力，因此依長尾學習文獻的標準做法（Cui et al., 2019）對訓練集做**指數衰減降採樣**：數字 0 保留 4,738 張 → 數字 9 只剩 47 張，不平衡比 100:1。**Validation 與 test 保持平衡**——這樣才能誠實量測每個類別的退化程度。

![MNIST 原始類別分布](plots/class_distribution.png)

![長尾訓練集分布](plots/longtail_distribution.png)

### 2. Modeling：遷移學習（ImageNet 預訓練 ResNet-18）

- 載入 `torchvision` 的 ResNet-18 預訓練權重，只把分類頭換成 `Linear(512, 10)`
- 輸入落差在資料端解決：resize 64×64、灰階複製三通道、ImageNet 統計值正規化
- **為什麼遷移學習**：類別不平衡真正傷害的是特徵學習；預訓練把通用視覺特徵「外包」給 ImageNet 的 128 萬張圖，尾部類別的 47 張樣本只需學分類邊界即可
- 資料增強：`RandomAffine`（±10° 旋轉、±10% 平移、0.9–1.1 縮放）；刻意不用鏡像翻轉（會破壞 2/5、6/9 的數字語意）

### 3. Optimization

| 設定 | 選擇 |
|------|------|
| 微調策略 | 全網微調 + **差異化學習率**（backbone 1e-4 / 新分類頭 1e-3，防止災難性遺忘） |
| Optimizer | AdamW（weight decay 1e-4） |
| Scheduler | CosineAnnealingLR |
| Early stopping | patience=4，監控 **validation macro-F1**（不平衡情境下 accuracy 會被頭部類別主導） |

### 4. Class Imbalance 對照實驗

同一份預訓練權重、同一批次順序，**唯一變因是 loss function**：

- **Exp 1 — Baseline**：一般 cross-entropy
- **Exp 2 — Weighted CE**：inverse-frequency 類別權重 $w_c = N/(K \cdot n_c)$（數字 9 的單張權重為數字 0 的 100 倍）

![訓練曲線](plots/training_curves.png)

## 結果（平衡 test set，10,000 張）

| 模型 | Accuracy | Balanced Accuracy | Macro-F1 |
|------|----------|-------------------|----------|
| Baseline (CE) | 0.9819 | 0.9817 | 0.9819 |
| Weighted CE | 0.9838 | 0.9835 | 0.9836 |

![Confusion matrices](plots/confusion_matrices.png)

（顏色為列正規化百分比、格內數字為實際張數——每列加總等於該數字的測試張數，兩模型完全相同）

![Per-class F1](plots/per_class_f1.png)

![尾部類別 PR 曲線](plots/pr_curves_tail.png)

![被 weighting 救回的影像](plots/rescued_examples.png)

（尾部類別 7/8/9 中「baseline 分錯、weighted 分對」的測試影像共 62 張，多為筆跡潦草的邊界案例。注意這是單向統計——反方向「baseline 對、weighted 錯」的樣本也存在，兩者的淨效果請看上方 per-class F1 表。）

## 關鍵發現與討論

1. **遷移學習本身就是最有效的不平衡對策。** 從零訓練的對照組（開發過程中先做過一輪自製小型 ResNet）中，數字 9（47 張訓練樣本）的 baseline F1 約 0.965，weighted loss 帶來 +0.02 的明顯增益；換成預訓練 ResNet-18 後，baseline 在尾部已接近天花板，weighted loss 的增益空間大幅縮小。原因：不平衡主要傷害特徵學習，而預訓練已把特徵學習完成，47 張樣本足以學好分類邊界（與 Kang et al., 2020 decoupling 研究的發現一致）。

2. **Weighted loss 的本質是用 precision 換 recall。** 放大尾部類別的 loss 等於把決策邊界推向「更容易判成尾部」，recall 上升、precision 下降；當 baseline recall 已高時，F1 可能持平甚至微降。

3. **極端權重有過擬合風險。** 100 倍權重 × 47 張圖，等於讓模型反覆背誦這 47 張特定影像。文獻的緩解方案：class-balanced loss（有效樣本數）、focal loss、logit adjustment、兩階段訓練。

4. **指標選擇很重要。** 在不平衡情境下用 accuracy 做模型選擇會系統性放棄尾部類別，本專案全程以 macro-F1 做 early stopping 與最終比較。

## 檔案結構

```
Task3/
├── MNIST_image_classification.ipynb   # 主要交付：完整程式碼 + 解說 + 圖表
├── README.md
├── requirements.txt
├── plots/                             # 由 notebook 輸出的圖表
└── data/                              # MNIST（自動下載）
```
