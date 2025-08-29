# Momentum-Sketch and Two-Stage Framework

## Introduction
We present a Two-Stage framework together with a novel Momentum-Sketch to boost throughput and memory efficiency for heavy hitter detection.

In Stage 1, incoming packets are inserted into a two-row CU Sketch, serving as a high-throughput pre-filter that quickly discards most low-volume flows with minimal memory and per-packet overhead. Flows exceeding an admission threshold are forwarded to Stage 2. 

In Stage 2, we introduce Momentum-Sketch, a novel counting data structure that combines arrival strength with estimated frequency via a momentum metric and employs a probabilistic decay strategy to preferentially retain true heavy hitters, effectively reducing estimation errors under skewed distributions. Momentum-Sketch is further enhanced with SIMD instructions to accelerate insert and query operations.

## Experimental Results
Momentum-Sketch excels in accuracy under low memory budgets. At 16 KB, its average F1 score across all datasets surpasses that of other methods by approximately 10% to 520%, and its average recall exceeds these methods by approximately 13% to 600%. Notably, because its counter-decay strategy eliminates overestimation, Momentum-Sketch achieves precision 1.0 under any memory constraint when threshold-based reporting is used. Momentum-Sketch further leverages SIMD acceleration to significantly boost insert and query speeds, improving average insert throughput by 47.3% and average query throughput by 62.7% across all datasets.

The Two-Stage framework enhances performance by filtering non-heavy flows, significantly improving insert and query throughput compared to Momentum-Sketch while maintaining a high F1 score. Specifically, it improves average insert throughput by 59.8% and average query throughput by 25.5% across all datasets.

In conclusion, Momentum-Sketch is highly effective in resource-constrained scenarios, delivering exceptional F1 scores at memory sizes ranging from 16 KB to 128 KB. Conversely, the Two-Stage framework significantly improves throughput while maintaining a high F1 score at memory sizes of 128 KB and beyond, making it an excellent choice for systems with greater memory capacity.

## 簡介
我們提出一個 Two-Stage 框架以及一種新穎的 Momentum-Sketch，以提升巨大流量檢測的效率與可擴展性。

在第一階段 (Stage 1)，輸入資料封包被插入至一個 two-row CU Sketch，作為高吞吐量的預過濾器，快速剔除大多數低流量，以最小的記憶體與每封包開銷實現過濾。超過准入閾值 (admission threshold) 的流量被轉發至第二階段 (Stage 2)。

在第二階段，我們使用 Momentum-Sketch計數資料結構，透過動量指標 (momentum metric) 結合到達強度 (arrival strength) 與估計頻率，並採用機率衰減策略 (probabilistic decay strategy)，優先保留真正的巨大流量，有效降低偏態分佈下的估計誤差。Momentum-Sketch 進一步利用 SIMD 指令加速插入與查詢操作。

## 實驗結果
Momentum-Sketch 在低記憶體預算下表現出卓越的準確性。在 16 KB 記憶體下，其平均 F1 分數比其他方法高出約 10% 至 520%，平均召回率高出約 13% 至 600%。特別的是，由於其計數器衰減策略 (counter decay strategy) 可消除過度估計 (overestimation)，Momentum-Sketch 在採用閾值式報告時，無論記憶體限制為何，皆可達到精確率 (precision) 1.0。Momentum-Sketch 進一步利用 SIMD 加速大幅提升插入和查詢速度，在所有資料集上平均插入吞吐量提升 47.3%，查詢吞吐量提升 62.7%。

Two-Stage 框架透過過濾非重流量提升性能，相較 Momentum-Sketch 大幅提高插入和查詢吞吐量，同時保持高 F1 分數。具體來說，在所有資料集上平均插入吞吐量提升 59.8%，查詢吞吐量提升 25.5%。

綜上所述，Momentum-Sketch 在資源有限的情況下表現出色，在 16 KB 到 128 KB 的記憶體下提供了卓越的 F1 score。而 Two-Stage 框架在 128 KB 及以上的記憶體下顯著提升 throughput 並維持高 F1 score，使其成為記憶體容量較大系統的選擇。
