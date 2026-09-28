# Hasil K-Stability 100 Seed

## Keputusan

**C — K=12 menggantikan K=14 sebagai reference cut utama** di antara kandidat yang diuji. K=14 bukan resolusi terbaik menurut chooseR dan tidak boleh disebut sebagai jumlah klaster “benar”. K=12 dipilih unik oleh chooseR; mean optimizer ARI-to-medoid tertinggi juga terjadi pada K=12. Ini mendukung cut yang lebih stabil/kompak, bukan klaim bahwa resolusi di luar rentang 12–16 telah ditolak.

## Protokol dan validasi

Fused space direkonstruksi dari delapan backbone dalam urutan `dinov3, radio, aimv2, siglip2, dinov2, siglip, convnext, eva02`: L2 per view, PCA-32 per view, konkatenasi, L2, PCA-100, lalu L2. Fit hanya pada 3.579 discovery rows; 600 holdout tidak disentuh. K=14, seed 42, `n_init=20` mereproduksi assignment tersimpan dengan **ARI=1,000**.

Optimizer stability memakai seed 0–99 untuk tiap K dan `n_init=20`. CI mean ARI dihitung dari 100 ARI-to-medoid values (t-interval), bukan dari 4.950 pasangan yang saling dependen. Pairwise ARI/AMI di bawah hanya statistik deskriptif. Null memakai 100 permutasi assignment dengan ukuran klaster tetap. chooseR memakai 100 subsample bersama, masing-masing 80%, seed 20260927; transform PCA di-fit ulang pada setiap subsample. Threshold chooseR adalah batas bawah CI bootstrap median tertinggi lintas K, dengan 25.000 bootstrap resample.

## Optimizer stability

| K | Mean ARI ke medoid (95% CI mean) | q2,5 / median / q97,5 / minimum | Run ARI ≥.90 / .95 / .99 | Pairwise ARI / AMI | Silhouette / source NMI (median) | Median min–max size |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 0,99747 [0,99564; 0,99931] | 0,96139 / 0,99938 / 1,00000 / 0,93516 | 100% / 99% / 96% | 0,9953 / 0,9962 | 0,3234 / 0,1600 | 238–461 |
| 13 | 0,99713 [0,99614; 0,99812] | 0,99118 / 0,99853 / 0,99953 / 0,96528 | 100% / 100% / 98% | 0,9954 / 0,9948 | 0,3299 / 0,1624 | 135–461 |
| 14 | 0,97730 [0,97146; 0,98314] | 0,93315 / 0,99648 / 0,99952 / 0,93181 | 100% / 69% / 68% | 0,9680 / 0,9741 | 0,3330 / 0,1576 | 130–461 |
| 15 | 0,98911 [0,98549; 0,99273] | 0,93850 / 0,99774 / 0,99947 / 0,93729 | 100% / 95% / 81% | 0,9817 / 0,9845 | 0,3395 / 0,1676 | 128–454 |
| 16 | 0,97924 [0,97433; 0,98414] | 0,93333 / 0,99563 / 0,99972 / 0,90045 | 100% / 85% / 62% | 0,9671 / 0,9767 | 0,3411 / 0,1666 | 107–455 |

Null ARI-to-medoid berada jauh di bawah observed pada semua K; empirical `p=0,0099` untuk masing-masing K, yaitu batas resolusi minimum dari 100 null draws. K=14 bukan yang paling stabil terhadap optimizer: hanya 69% run mencapai ARI ≥0,95, dibanding 99% pada K=12. Silhouette embedding biasa meningkat sampai K=16, tetapi tidak dipakai sebagai aturan pemilihan.

Margin of error 95% untuk mean ARI-to-medoid adalah ±0,00184 pada K=12 dan ±0,00584 pada K=14. Interval ini menggambarkan variasi random initialization bersyarat pada medoid terpilih; ia bukan interval generalisasi populasi.

## chooseR dan perbandingan resolusi

Threshold chooseR = **0,997354**. K=12 memiliki 7 cluster median silhouette yang melampaui threshold dan bootstrap CI median consensus silhouette **[0,997354; 0,997472]**; K=13: 5 cluster, [0,992962; 0,995620]; K=14: 0, [0,987744; 0,988317]; K=15: 5, [0,994013; 0,996693]; K=16: 2, [0,990200; 0,991515]. Dengan demikian chooseR memilih **K=12 secara unik**.

Consensus silhouette tersebut dihitung pada jarak `1 − co-clustering frequency`, bukan silhouette pada fused embedding. Paired bootstrap membedakan K=14 dari K=15: median selisih (14−15) = −0,00664, CI 95% [−0,00862; −0,00617]. Sembilan dari sepuluh pasangan K mempunyai CI yang tidak mencakup nol; hanya K=13 versus K=15 yang tidak terbedakan. Bukti tidak mendukung plateau statistik menyeluruh K=12–16. Prediction Strength tidak dijalankan: chooseR unik, K=14/15 terbedakan, dan optimizer winner selaras dengan chooseR.

Median cluster-wise Jaccard lintas cluster adalah 0,9955 (K12), 0,9910 (K13), 0,9839 (K14), 0,9948 (K15), dan 0,9897 (K16). Pada K=12 tidak ada cluster dengan q2,5 Jaccard <0,60. Tidak ada cluster yang jatuh di bawah Jaccard 0,60 pada mayoritas subsample. Namun, salah satu node `display` K=14 (cluster 13 setelah alignment ke profil lama) mempunyai lower tail q2,5=0,005; 41% subsample memiliki Jaccard <0,60, dengan indikator split/merge masing-masing 39%/41%. Jadi kestabilan global K=14 menutupi satu subcluster display yang sensitif terhadap sampling.

## Implikasi dan batas bukti

Tahap sebelumnya memilih K=14 pada coarse stage (composite 0,7809), sedangkan refinement memberi K=15 skor 0,8045. Skor semantik/composite tersebut **tidak digunakan** untuk memilih K di sini. Menurut protokol stability yang dikunci, K=12 menggantikan K=14; K=14 dapat dipertahankan hanya sebagai sensitivity cut jika membahas granularitas, bukan sebagai resolusi utama. Hasil external yang sudah ada—Macro Top-1 K12=0,9051 dan K14=0,9031, paired CI delta K14−K12 [−0,0154; +0,0129]—tidak menunjukkan penurunan yang terdeteksi pada K12; tidak ada Bangladesh inference baru.

Batas terpenting: kandidat hanya K=12–16, dan pemenang berada di batas bawah grid, sehingga K=12 adalah yang terbaik **di antara kandidat ini**, bukan optimum global. Source NMI median tetap sekitar 0,16; confound akuisisi berkurang relatif pada run lama tetapi belum hilang. Node semantik adalah nama post-hoc dari profil yang sudah ada, bukan label kebenaran.

Run valid berlangsung **19 menit 12 detik**. Satu percobaan awal sekitar 8 menit 55 detik dibuang karena overflow `int8` pada alignment node setelah seluruh optimizer runs; bug diperbaiki, dan retry lengkap tersimpan. Total compute wall time eksperimen sekitar **28 menit 7 detik**; tidak memakai Modal.
