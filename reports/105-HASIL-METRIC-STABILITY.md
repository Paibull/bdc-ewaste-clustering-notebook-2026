# Hasil metric stability dan audit paper tim lain

## Metric yang ditemukan

Dua draft lokal diaudit tanpa mengubah isinya:

- `Draft BDC12 Semifinal Satria Data 2026.docx` (HyMVD) melaporkan neighbor
  agreement, cross-k Jaccard, seed stability, HDBSCAN clustered fraction,
  parent containment, serta frozen external Top-1/Top-3/MRR dan novelty AUROC.
- `draf KTI 20 September 2026 (1).docx` merencanakan Silhouette,
  Davies-Bouldin, Calinski-Harabasz, DBCV, ARI/NMI antar-run, cluster-wise
  Jaccard, dan co-association matrix. Pada draft tersebut sebagian besar masih
  berupa rancangan evaluasi, bukan hasil numerik.

Metric yang fair untuk partisi K-Means final kita adalah seed ARI/AMI, neighbor
agreement, cross-k Jaccard, Silhouette, CH, dan DB. HDBSCAN clustered fraction
hanya dipakai pada kandidat density-based; parent containment hanya relevan
untuk hierarchy. Keduanya tidak diisikan secara artifisial pada partisi flat.

## Hasil baru

| Lensa | Seed ARI | Seed AMI | ARI null q97,5 | p ARI | AMI null q97,5 | p AMI | Neighbor agreement@15 | Cross-k Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Function | 0,9995 | 0,9994 | 0,6311 | 0,05 | 0,6971 | 0,05 | 0,9605 | 0,9224 |
| Material | 0,9882 | 0,9812 | 0,8991 | 0,05 | 0,8858 | 0,05 | 0,8683 | 0,7437 |
| Condition | 0,9905 | 0,9854 | 0,9909 | 0,15 | 0,9860 | 0,15 | 0,8870 | 0,8057 |
| Data-driven | 0,9746 | 0,9798 | 0,4749 | 0,05 | 0,5792 | 0,05 | 0,9599 | 0,9189 |

Sumber numerik: `semifinal/results/lens_robustness_metrics.csv`.

## Interpretasi

1. Semua partisi repeatable terhadap random seed secara raw.
2. Function dan Data-driven juga kuat secara lokal dan lintas k.
3. Material memiliki neighborhood dan cross-k robustness paling rendah. Ini
   konsisten dengan temuan sebelumnya bahwa material adalah lensa paling lemah.
4. Condition tidak melampaui matched-covariance null pada ARI maupun AMI
   (`p=0,15`). Karena itu Stability ARI 0,991 tidak boleh dipakai sendiri sebagai
   bukti bahwa condition semantics benar.
5. Function, Material, dan Data-driven mencapai `p=0,05`, yaitu batas resolusi
   audit 19 draw. Hasil ini mendukung robustness eksploratif, tetapi belum cukup
   untuk klaim p-value presisi tinggi.

Metric baru memperkuat paper bila dipakai sebagai panel yang saling mengoreksi,
bukan leaderboard scalar. Kontribusi yang paling defensible tetap harmonisasi
acquisition confound dan controlled multi-lens discovery; stability menjelaskan
reproducibility, bukan semantic correctness.

