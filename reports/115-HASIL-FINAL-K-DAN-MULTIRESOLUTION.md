# Keputusan final K dan bukti multi-resolution

## Keputusan K

Gunakan **K=12 sebagai reference resolution**, bukan sebagai jumlah cluster yang dianggap benar. Pada refinement K=12–16, chooseR memilih K=12 secara unik; mean optimizer ARI-to-medoid tertinggi juga terjadi pada K=12 (0,99747; 95% CI 0,99564–0,99931). Pilihan ini berlaku untuk kandidat dan protokol yang diuji, bukan optimum global.

Broad screen fused K-Means telah mencakup K=4–24. K>16 tidak perlu dijalankan ulang: dibanding K=16, silhouette/adjusted function coverage masing-masing turun dari 0,321/0,258 di K=16 menjadi 0,309/0,222 di K=18, 0,293/0,197 di K=20, 0,254/0,156 di K=22, dan 0,257/0,132 di K=24. Function AMI juga turun dari 0,795 di K=16 menjadi 0,788, 0,769, 0,758, dan 0,750. Silhouette biasa bukan aturan pemilihan K; hasil broad screen hanya menunjukkan bahwa resolusi lebih tinggi yang sudah diuji tidak memberi dukungan konsisten untuk mengganti K=12.

## Interpretasi multi-resolution

Pada K=12, laptop cluster C3 (n=324) terpetakan ke dua child di K=16: desktop C3 (n=130; child containment terhadap parent 0,908; q2.5 Jaccard 0,938) dan laptop C10 (n=213; child containment 0,953; q2.5 Jaccard 0,945). Keduanya stabil terhadap resampling. Proporsi source web, setelah join exact key pada seluruh 3.579 discovery rows, adalah 0,1265 pada parent, 0,3000 pada child desktop, dan 0,0329 pada child laptop. Label source berasal dari fraction_table.csv; pada contoh ini proporsinya sama dengan proxy ukuran 150×150. Hasil tetap **PERSISTENT_ONLY** karena komposisi source berbeda dan inspeksi visual belum tersedia dari resolver yang digunakan. Pengukuran proporsi source bukan penyesuaian statistik terhadap source. Istilah child containment menunjukkan overlap antarpartisi, bukan purity terhadap ground truth.

Kasus batasnya adalah display K=12 C10 ke K=14 C13/C8. K14 C13 memiliki q2.5 Jaccard 0,0045, dengan 41% subsample di bawah Jaccard 0,60; K14 C8 memiliki q2.5 0,503, dengan 40% subsample di bawah 0,60. Ini menunjukkan bahwa node yang tampak terpisah pada satu cut dapat sensitif terhadap sampling. Nama node adalah deskriptor post-hoc, dan stabilitas assignment bukan bukti kebenaran semantik.

## Material dan transfer

Perbaikan secondary scoring menghilangkan square-root ganda pada reconstruction fraction. Incremental R², CI, permutation p, dan strict-transfer tidak berubah. Incremental R² material tetap positif: K12 enam material 0,1145 (95% CI 0,0889–0,1409; p=0,001), K12 lima material 0,1528 (0,1210–0,1845; p=0,001), K14 enam material 0,1190 (0,0915–0,1463; p=0,001), dan K14 lima material 0,1571 (0,1259–0,1887; p=0,001). Secondary reconstruction metrics yang terkoreksi: Hellinger reduction 0,0295/0,0463 untuk K12 dan 0,0300/0,0466 untuk K14 (six/five materials); cosine gain 0,0190/0,0310 dan 0,0196/0,0317. Hasil material tetap merupakan dukungan post-hoc pada locked holdout, bukan pixel-level ground truth atau dasar memilih K.

Strict frozen external transfer K12 tetap macro Top-1 0,888 (95% CI 0,865–0,913), paired delta terhadap RADIO K12 95% CI 0,029–0,070. Hasil ini mendukung transfer pada label family yang tersedia, dengan batas coverage dan cakupan kelas pada laporan 114.

**Batas konsistensi partisi:** scorer transfer/material menggunakan K12 lama dari semantic_g0_partitions.npz, sedangkan chooseR dan kasus multi-resolution menggunakan medoid K12 dari k_stability_100/assignments.npz. Urutan key identik, tetapi ARI antarkedua partisi adalah **0,974794**, bukan 1. Jadi hasil 114 adalah validasi partisi K12 lama dan tidak boleh diatribusikan sebagai validasi persis medoid K12 yang dipilih chooseR. Constructor fused juga berbeda. Tidak ada validasi ulang medoid baru pada sesi ini.

Incremental R² pada scorer didefinisikan sebagai 1 − SSE_full/SSE_baseline pada holdout di root space, yaitu penurunan error relatif terhadap model source/statistik akuisisi; bukan R² terhadap total variance. Hellinger dan cosine memakai root space yang sama (H²=1−cosine per citra), sehingga keduanya bukan dua bukti independen. Bootstrap bersyarat pada model/partisi yang telah di-fit dan tidak mencakup uncertainty pemilihan K atau variasi training.

## Kalimat siap adaptasi ke paper

1. “Kami menetapkan K=12 sebagai reference resolution karena prosedur chooseR memilihnya secara unik pada refinement K=12–16, dengan stabilitas optimizer tertinggi di antara kandidat tersebut.”
2. “Broad screen K=4–24 menunjukkan bahwa kenaikan resolusi di atas K=16 tidak memberi perbaikan konsisten pada metrik fungsi dan coverage, sehingga K=12 digunakan sebagai cut utama dan cut lain dipertahankan sebagai analisis sensitivitas.”
3. “Analisis multi-resolution menemukan refinement laptop yang persisten pada K=16, tetapi perbedaan komposisi source dan ketiadaan inspeksi visual membatasi interpretasinya menjadi persistence struktural, bukan bukti subtype semantik yang tervalidasi.”
4. “Pada locked holdout, partisi K12 terdahulu memberikan penurunan error material relatif setelah kontrol source dan covariates akuisisi; hasil ini mendukung koherensi post-hoc, tetapi belum memvalidasi persis medoid K12 baru atau komposisi material pixel-level.”

## Artefak

- Kasus dan metrik: [`cases.csv`](../results/final_multiresolution/cases.csv)
- Ringkasan mesin: [`summary.json`](../results/final_multiresolution/summary.json)
- Figur multi-resolution: [`figure_multiresolution_cases.png`](../results/final_multiresolution/figure_multiresolution_cases.png)
- Metrik material terkoreksi: [`material_holdout_metrics.csv`](../results/final_claim_checks/material_holdout_metrics.csv)
- Laporan transfer dan material: [114-HASIL-STRICT-TRANSFER-MATERIAL-HOLDOUT.md](114-HASIL-STRICT-TRANSFER-MATERIAL-HOLDOUT.md)
