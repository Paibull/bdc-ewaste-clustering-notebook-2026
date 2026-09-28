# Katalog Metode — Semifinal BDC Satria Data 2026

Kumpulan lengkap opsi per tahap pipeline, hasil riset 3 Sep 2026.
Notasi: **★** = prioritas tinggi untuk kasus kita · **[F]** = jalan di atas embedding frozen (tanpa retrain backbone) · **[G]** = butuh GPU · **[RW]** = untuk related work saja, tidak realistis dikerjakan.

Konteks tetap: 4.179 citra, embedding 8 backbone sudah di-cache, CPU 2-core + Kaggle/Modal sesekali, deadline 28 Sep.

---

## Daftar isi

Dapatkan nomor baris terkini dengan `grep -n '^#' semifinal/01-KATALOG-METODE.md`,
lalu baca **hanya seksi yang dibutuhkan** dengan `sed -n 'X,Yp'`.
Jangan baca seluruh file: 107 KB, sekitar 27k token.

| Seksi | Isi |
|---|---|
| TAHAP 0 | Kurasi & harmonisasi data |
| TAHAP 1 | Backbone: 1.1 yang dipunya · 1.2 tambahan prioritas · 1.3 enumerasi timm · 1.4 di luar timm · **1.5 bukti backbone terbaik untuk clustering** · 1.6 jebakan diam · 1.7 biaya ekstraksi |
| TAHAP 2 | Post-processing fitur: normalisasi, hubness, INLP, CORAL, konkatenasi, SpLiCE |
| TAHAP 3 | Reduksi dimensi + peringatan jangan clustering di 2-D |
| TAHAP 4 | Clustering: 4.1 klasik · **4.2 fitur foundation model frozen** · 4.3 ClustPy · 4.4 butuh retrain (RW) · **4.5 multiple clustering — inti kontribusi** · 4.6 multi-view · 4.7 consensus · 4.8 k tidak diketahui |
| TAHAP 5 | Pemilihan k: gap statistic, prediction strength, stability, ICL, NbClust, SigClust |
| TAHAP 6 | Validasi: 6.1 anggaran compute · 6.2 indeks internal · **6.3 indeks density-based (DBCV)** · 6.5 kritik CVI · **6.6 stabilitas + kritik teoretisnya** · 6.7 null model · **6.8 validasi tanpa label (chooseR, MultiK)** · 6.11 multiverse · 6.12 hubness & dimensi intrinsik · 6.13 checklist |
| TAHAP 7 | Interpretasi: 7.1 sub-ruang konsep · 7.2 penamaan klaster · **7.3 metrik komplementaritas** · 7.4 evaluasi manusia · 7.5 visualisasi |
| TAHAP 8 | Domain e-waste: **8.1 taksonomi resmi (WEEE, UNU-KEY, Basel, B107d Indonesia)** · 8.2 dataset publik + lisensi · 8.3 penelitian terkait · **8.4 nilai recovery** · 8.5 desain taksonomi klaster |
| LIBRARY | Status terpasang, mana yang gagal pip, repo yang perlu di-clone |
| REKOMENDASI | Prioritas eksekusi 1/2/3 + daftar yang HARUS dihindari |
| BELUM TERVERIFIKASI | Satu-satunya bagian yang masih butuh riset baru |


# TAHAP 0 — Kurasi & harmonisasi data

| Metode | Kegunaan | Catatan |
|---|---|---|
| ★ Harmonisasi resolusi (downsample semua ke 150 px lalu upsample ke input backbone) | Menghapus confound dua-korpus yang sudah terukur (AUC probe 0,993) | Satu run embedding ulang. Ini eksperimen pembuka paper |
| Super-resolusi korpus rendah (Real-ESRGAN, SwinIR, HAT, DiffBIR) | Alternatif arah harmonisasi | Memasukkan artefak generatif ke data analisis — harus disebut sebagai limitasi |
| Deduplikasi near-duplicate (cosine > 0,95–0,99, atau pHash/dHash) | Klaster tidak didominasi seri gambar yang sama | Sudah diukur: 122 citra (2,9%) pada 0,99; 659 (15,8%) pada 0,95 |
| Deteksi OOD / non-e-waste (skor jarak ke centroid, Isolation Forest, Mahalanobis, energy score) | Buang screenshot, kolase, gambar rusak | Script `ood_detect.py` dari penyisihan bisa dipakai ulang |
| Cleanlab (confident learning) | Deteksi anomali label — di semifinal dipakai untuk menyaring citra yang bukan e-waste | `pip install cleanlab` |
| Segmentasi objek / crop otomatis (SAM, SAM2, Grounding DINO, `rembg`) | Menghapus latar sebagai variabel pengganggu | Bisa jadi ablasi: klaster sebelum vs sesudah background removal |
| Deteksi latar studio (statistik border putih, saturasi, edge density) | Kuantifikasi gaya akuisisi | Sudah dihitung di `imgstats.csv` |
| Koreksi warna / CLAHE / white balance | Menyeragamkan pencahayaan | Preseden Indonesia: Afifah et al. 2025, *Jurnal RESTI* 9(3) — CLAHE + bilateral filtering terbaik untuk MobileNetV1 |

---

# TAHAP 1 — Representasi (backbone zoo)

## 1.1 Yang sudah dipunya (8 backbone, ter-cache)

`dinov3` ViT-L · `radio` C-RADIOv4-SO400M · `aimv2` L · `siglip2` SO400M-384 · `dinov2` L · `siglip` SO400M-384 · `convnext` ConvNeXtV2-L · `eva02` L

**Temuan penting: C-RADIOv4-SO400M punya text tower yang belum dipakai.** Lewat adaptor `siglip2-g`:

```python
model = torch.hub.load('NVlabs/RADIO','radio_model', version='c-radio_v4-so400m',
                       adaptor_names=['siglip2-g','dino_v3','sam3'])
ad = model.adaptors['siglip2-g']
tfeat = ad.encode_text(ad.tokenizer(prompts).cuda(), normalize=True)
```
Adaptor v4: `siglip2-g`, `dino_v3`, `sam3` → **3 ruang fitur tambahan dari model yang sudah diunduh**, satu forward pass. Ini penambahan diversitas termurah yang ada.

## 1.2 Tambahan berprioritas tinggi

| Model | timm / HF id | dim | Text? | Kenapa |
|---|---|---|---|---|
| ★ **C-RADIOv4-H** | `nvidia/C-RADIOv4-H` | 1280 | ✅ via adaptor | **kNN IN-1k 86,59** — angka kNN frozen tertinggi di literatur. ~3 menit T4 |
| ★ **DINOv3 ViT-H+/16** | `vit_huge_plus_patch16_dinov3.lvd1689m` | 1280 | ❌ | 2,8× kapasitas DINOv3-L yang sekarang dipakai |
| ★ **SigLIP2 g-opt** | `ViT-gopt-16-SigLIP2-384` (open_clip) | 1536 | ✅ | SigLIP2 terbesar (1B); kita cuma punya SO400M |
| ★ **Franca ViT-G/14** | `torch.hub.load('valeoai/Franca','franca_vitg14')` | — | ❌ | **Objektif nested-Matryoshka clustering** — paling clustering-native yang ada, kNN 83,0. Nyaris belum dicoba orang → nilai kebaruan |
| ★ **MSN ViT-L/16** | `facebook/vit-msn-large` | 1024 | ❌ | Backbone terbaik di satu-satunya ablasi head-to-head *clustering* yang ada (TEMI: 61,6% ImageNet) |
| **DINOv2 ViT-g/14-reg4** | `vit_giant_patch14_reg4_dinov2.lvd142m` | 1536 | ❌ | Register menghapus token artefak high-norm; semua SOTA clustering dibangun di DINOv2 |
| **PE-Core G/14-448** | `vit_pe_core_gigantic_patch14_448.fb` | 1536 | ✅ | Linear probe IN1k 89,3 (tertinggi); PE-Spatial G14 ruang berbeda lagi |
| **AIMv2-L LiT** | `apple/aimv2-large-patch14-224-lit` | 1024 | ✅ | Menambah text tower ke ruang AIMv2 yang sudah dipunya |
| **DFN5B ViT-H/14-378** | `ViT-H-14-378-quickgelu` / `dfn5b` | 1024 | ✅ | Distribusi data sangat berbeda dari LAION/WebLI |
| **MetaCLIP2 worldwide** | `ViT-bigG-14-worldwide-378` / `metaclip2_worldwide` | 1280 | ✅ | Dilatih 300+ bahasa — relevan kalau prompt pakai bahasa Indonesia |
| **Marqo-FashionSigLIP** | `hf-hub:Marqo/marqo-fashionSigLIP` | — | ✅ | Fine-tune khusus foto produk e-commerce — cocok dengan karakter data kita yang banyak foto katalog |

## 1.3 Enumerasi lengkap yang tersedia lokal (timm 1.0.29, terverifikasi)

**1.737 tag pretrained.** Per keluarga relevan:

| Keluarga | #tag | Contoh |
|---|---|---|
| CLIP (openai/laion/dfn/metaclip/datacomp) | 127 | `vit_huge_patch14_clip_378.dfn5b` |
| ConvNeXt V1 | 72 | `convnext_xlarge.fb_in22k` |
| SigLIP1 + varian | 56 | `vit_so400m_patch16_siglip_gap_384.v2_webli` |
| SigLIP2 + NaFlex | 35 | `naflexvit_so400m_patch16_siglip.v2_webli` |
| EVA / EVA02 | 31 | `eva02_large_patch14_448.mim_m38m_ft_in22k` |
| MaxViT | 30 | `maxvit_xlarge_tf_512.in21k_ft_in1k` |
| ConvNeXt V2 | 26 | `convnextv2_huge.fcmae` (2816-d) |
| DINOv3 (ViT + ConvNeXt + varian `_qkvb`/`eupe`) | 21 | `vit_7b_patch16_dinov3.lvd1689m` (4096-d) |
| BEiT / BEiTv2 / BEiT3 | 21 | `beit3_large_patch16_224.pt` |
| PaliGemma SigLIP tower | 17 | `vit_so400m_patch14_siglip_gap_448.pali2_10b_pt` |
| Hiera | 16 | `hiera_huge_224.mae` (2048-d) |
| SwinV2 | 15 | `swinv2_large_window12to24_192to384` |
| ViTamin | 15 | `vitamin_large2_224.datacomp1b_clip` |
| ResNeXt WSL/SWSL/SSL | 15 | `resnext101_32x16d.fb_wsl_ig1b_ft_in1k` |
| AIMv2 | 14 | `aimv2_3b_patch14_448.apple_pt` (3072-d) |
| Perception Encoder | 14 | `vit_pe_core_gigantic_patch14_448.fb` |
| RegNetY SEER/SWAG | 13 | `regnety_2560.seer_ft_in1k` |
| FlexiViT | 13 | `flexivit_large.1200ep_in1k` |
| MobileCLIP / MobileCLIP2 | 11 | `fastvit_mci1.apple_mclip2_dfndr2b` |
| SAM / SAM2 | 11 | `samvit_huge_patch16.sa1b` (256-d, 1024px — mahal & lemah semantik) |
| MAE | 9 | `vit_huge_patch14_224.mae` |
| DINOv2 | 8 | `vit_giant_patch14_reg4_dinov2.lvd142m` |
| DINOv1 | 4 | `vit_base_patch8_224.dino` |
| I-JEPA | 4 | `vit_giant_patch16_gap_224.in22k_ijepa` (1408-d) |

**open_clip:** 196 pasangan (model, pretrained), 115 arsitektur unik, 15 di antaranya SigLIP2.

## 1.4 Keluarga lain yang tidak ada di timm

- **Web-SSL** (Meta 2025) — `facebook/webssl-dino{300m,1b,2b,3b,5b,7b}-full2b-224`. DINO tanpa bahasa di 2B citra MetaCLIP. Lisensi **CC-BY-NC-4.0**. Baca `model.config.hidden_size`, jangan percaya tabel README.
- **CAPI** (Darcet et al. 2025) — *Cluster and Predict Latent Patches*, `facebookresearch/capi`. MIM berbasis clustering.
- **iBOT, MoCo v3, EsViT, MUGS, MOCA, data2vec 2.0, SimMIM** — checkpoint di repo masing-masing.
- **InternViT** — `OpenGVLab/InternViT-6B-448px-V2_5` (6B). Varian `InternViT-6B-224px-V1_0` punya sisi teks (InternVL-C/G).
- **EVA-CLIP-18B** — `BAAI/EVA-CLIP-18B`, CLIP terbuka terbesar.
- **OpenVision** — 25+ checkpoint fully-open (arXiv 2505.04601).
- **Fitur model generatif**: DIFT (fitur U-Net Stable Diffusion pada timestep t≈261), Diffusion Hyperfeatures, **SD-DINO** — bukti terpublikasi bahwa *konkatenasi* fitur SD + DINO mengalahkan keduanya sendiri-sendiri. VAE latent (`stabilityai/sd-vae-ft-mse`) sebagai sumbu tekstur/warna murni.
- **Vision tower MLLM**: Qwen2.5/3-VL `.visual`, PaliGemma2, Gemma-3, Molmo, Idefics2/3, InternVL3.5, LLaVA.

## 1.5 Bukti: backbone mana yang paling bagus untuk CLUSTERING

**Peringkat clustering ≠ peringkat linear probe.** Ini poin metodologis yang layak ditulis di paper.

| Sumber | Temuan |
|---|---|
| TEMI (BMVC 2023) — 17 backbone, satu metode | **MSN ViT-L/16 61,6%** dan **DINO ViT-B/16 58,4%** di ImageNet; CLIP dan supervised ResNet/ConvNeXt jauh di bawah. SSL ViT > supervised > CLIP untuk clustering |
| ICCE (arXiv 2511.16213) | Semua di DINOv2 ViT-L/14 frozen: **ImageNet 70,36 / CIFAR-10 99,20 / CIFAR-100 88,14**. Ganti backbone DINO→DINOv2 memberi +8…+20 poin, jauh lebih besar dari perbaikan algoritma clustering manapun |
| ★ **TURTLE (ICML 2024)** | **Dua ruang representasi heterogen (CLIP + DINOv2) mengalahkan salah satunya, sampai +35% absolut.** Ini argumen terkuat yang ada untuk cache 8-backbone kita — bukan ablasi, tapi kontribusi |
| DINOv2 paper | kNN IN-1k: ViT-L 83,5 · ViT-g 83,5 (seri!) sementara linear ViT-g menang. **Kualitas geometri metrik jenuh lebih awal dari separabilitas linear** |
| C-RADIOv4 tech report | **v4-H kNN 86,59** — tertinggi yang ditemukan di literatur, di atas DINOv2-g 83,5 |
| DINOv3 paper Tabel 3 | PE-core G/14 linear 89,3 dan SigLIP2 g/16 89,1 menang linear; **DINOv3 7B menang telak di tugas dense** (ADE20k 55,9 vs DINOv2-g 49,5). Clustering berperilaku seperti kolom kedua |
| Web-SSL (arXiv 2504.01017) | Data cocok, Web-DINO 7B vs MetaCLIP ViT-G: klasifikasi identik 86,4, **tapi SSL menang dense +4,6**. Supervisi bahasa bukan bahan yang menghasilkan geometri metrik bagus — skala data yang menghasilkannya |
| Battle of the Backbones (NeurIPS 2023 D&B) | Supervised ConvNet masih menang banyak tugas — **tapi sudah usang, pra-DINOv2**. Kutip untuk metodologi, bukan sebagai bukti soal era 2025 |

**Kesimpulan yang bisa dipertahankan di paper: multi-ruang (gaya TURTLE) > backbone tunggal terbaik.** Kutip kNN, jangan linear probe.

## 1.6 Jebakan diam yang merusak hasil

1. **SigLIP text harus `padding="max_length", max_length=64`.** `padding=True` diam-diam merusak embedding teks.
2. **timm `_gap_` vs non-`gap` SigLIP adalah dua ruang berbeda.** Non-gap = ruang gabungan sejajar-teks (bisa dicosine dengan teks); gap = GAP token, **tidak** sebanding dengan teks. Ambil dua-duanya — satu unduhan, dua ruang fitur.
3. **Cek varian EVA02 yang di-cache**: `.mim_*` (tanpa teks) vs `.merged2b` CLIP (dengan teks) — ruang yang benar-benar berbeda.
4. SigLIP bersifat sigmoid per-pasangan, bukan softmax antar-kelas → nilai cosine mentahnya kecil (0,00–0,30). **Rank atau z-score per sumbu konsep; jangan threshold nilai cosine absolut.**
5. Input default DINOv3 di timm adalah **256×256**, bukan 224.

## 1.7 Biaya ekstraksi (4.179 citra)

| Model @ resolusi | GFLOP/citra | T4 | A10G |
|---|---:|---:|---:|
| DINOv3 ViT-L/16 @256 | 164 | 0,6 mnt | 0,2 |
| DINOv3 ViT-H+/16 @256 | 340 | 1,3 | 0,5 |
| DINOv3 ViT-7B/16 @256 | 3548 | 13,8 | 5,0 |
| SigLIP2 SO400M/16 @384 | 512 | 2,0 | 0,7 |
| SigLIP2 g-opt/16 @384 | 1386 | 5,4 | 1,9 |
| C-RADIOv4-H @384 | 779 | 3,0 | 1,1 |
| PE-Core-G/14 @448 | 3225 | 12,5 | 4,5 |

Kali 2–3× untuk wall-clock nyata. **Compute bukan masalah** — hampir semuanya job T4 1–15 menit. Yang mengikat adalah ukuran unduhan bobot (DINOv3-7B 27 GB) dan disk Kaggle 30 GB. Cache sebagai float16 `.npy`: 4.179 × 1536 × 2 B = **13 MB per backbone**.

**Kaggle Models:** `keras/dinov3` tersedia (8 preset), `metaresearch/dinov2`, `keras/siglip`, `google/paligemma`. **timm TIDAK dimirror di Kaggle** — pakai notebook internet-on, atau pre-stage bobot sebagai private Kaggle Dataset (`HF_HUB_OFFLINE=1`).

---

# TAHAP 2 — Post-processing fitur

| Metode | Fungsi | Python |
|---|---|---|
| ★ L2-normalisasi | Wajib. Setelah L2-norm, ‖x−y‖² = 2(1−cos) → k-means Euclidean = spherical k-means | numpy |
| PCA / PCA-whitening | Turunkan ke 2–3× dimensi intrinsik. **Indeks berbasis determinan mati di d=1000** | `sklearn.decomposition.PCA` |
| ★ Hubness reduction — Mutual Proximity, Local Scaling, NICDM, DisSimLocal | Di d≈1000 distribusi N_k(x) sangat skewed; hub merusak kNN graph → merusak spectral, HDBSCAN, Leiden, kNN-purity | `scikit-hubness` (`skhubness.reduction`), atau MP manual ~15 baris |
| ★ INLP (iterative nullspace projection) | Buang arah linier yang memprediksi korpus/nuisance | sudah diimplementasi di `analisis_ewaste.py` |
| ★ Penyeragaman per-korpus (gaya CORAL) | Domain adaptation murah | sudah diimplementasi |
| Konkatenasi multi-backbone (+ per-blok L2-norm) | Bukti SD-DINO & TURTLE: gabungan > tunggal | numpy |
| CCA / GCCA multi-view | Cari sub-ruang bersama antar backbone | `sklearn.cross_decomposition`, `mvlearn` |
| ZCA whitening / all-but-the-top | Menghilangkan arah dominan yang isotropik | numpy |
| Kernel PCA (RBF, cosine, poly) | Non-linear | `sklearn.decomposition.KernelPCA` |
| Sparse coding / dictionary learning | Representasi sparse yang lebih interpretable | `sklearn.decomposition.DictionaryLearning` |
| ★ SpLiCE (sparse concept decomposition) | Dekomposisi embedding CLIP jadi kombinasi sparse-nonneg atas kamus 15.000 konsep teks; **post-hoc, tanpa training** | github.com/AI4LIFE-GROUP/SpLiCE |
| Sparse autoencoder (DN-CBM, PatchSAE) | Temukan kamus konsep dari fitur frozen, lalu namai per unit | github.com/neuroexplicit-saar/Discover-then-Name |

**Cosine vs Euclidean — justifikasi sebenarnya:** norm fitur CNN/ViT mengkode skala objek, kontras, confidence — faktor pengganggu untuk clustering *kategori*; arah mengkode semantik. Backbone kontrastif (CLIP, DINO) **dilatih dengan objektif cosine (InfoNCE bertemperatur atas embedding ternormalisasi)** — jadi memakai cosine berarti mencocokkan metrik dengan objektif pelatihan. Itu argumen terkuat yang bisa ditulis. Catatan: Ward linkage **wajib** Euclidean → normalisasi dulu lalu jelaskan kenapa itu sah.

---

# TAHAP 3 — Reduksi dimensi

| Metode | Sifat | Python | Status lokal |
|---|---|---|---|
| PCA / IncrementalPCA / TruncatedSVD | Linear, deterministik | sklearn | ✅ |
| Kernel PCA | Non-linear | sklearn | ✅ |
| ICA (FastICA) | Sumber independen | sklearn | ✅ |
| NMF / MiniBatchNMF | Bagian aditif, non-negatif → interpretable | sklearn | ✅ |
| Factor Analysis | Model laten | sklearn | ✅ |
| Random Projection (Gaussian, Sparse) | Johnson-Lindenstrauss, sangat cepat | sklearn | ✅ |
| ★ UMAP | Manifold, jaga struktur lokal + sebagian global | `umap-learn` | ✅ |
| t-SNE | Visualisasi | sklearn / `openTSNE` | ✅ |
| ★ PaCMAP | Keseimbangan lokal-global lebih baik dari UMAP | `pacmap` | ✅ |
| TriMap | Berbasis triplet, jaga struktur global | `trimap` | ✅ |
| PHATE | Difusi, bagus untuk struktur kontinu/trajektori | `phate` | ✅ |
| Isomap / LLE / LTSA / Hessian LLE / MDS / Spectral Embedding | Manifold klasik | sklearn | ✅ |
| Diffusion maps | Difusi | `datafold`, `pydiffmap` | — |
| SOM (Self-Organizing Map) | Grid topologis — **visualisasi klaster yang jarang dipakai orang, nilai plus** | `minisom` | — |
| Autoencoder / VAE / β-VAE | Latent belajar | torch |  ✅ (torch ada) |
| Matryoshka / MRL truncation | Potong dimensi embedding bertingkat | — | — |

## ⚠️ Jebakan yang wajib dihindari
**Jangan clustering di koordinat UMAP/t-SNE 2-D dan jangan hitung indeks validitas di atasnya.** Embedding 2-D mendistorsi densitas dan jarak antar-klaster sampai menciptakan struktur palsu. Kutip: Chari & Pachter, *The specious art of single-cell genomics*, PLOS CB 2023; rebuttal-nya (PLOS CB 2024) untuk keseimbangan; *Stop Misusing t-SNE and UMAP for Visual Analytics* (arXiv 2506.08725). Posisi dokumentasi UMAP sendiri: clustering di UMAP **berdimensi sedang (10–50)** sah; di visualisasi 2-D tidak. **Pakai 2-D hanya untuk gambar, dan beri label bahwa itu hanya untuk gambar.**

---

# TAHAP 4 — Algoritma clustering

## 4.1 Klasik (semua [F], semua jalan di CPU)

**sklearn.cluster (terverifikasi):** `KMeans, MiniBatchKMeans, BisectingKMeans, AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, HDBSCAN, MeanShift, OPTICS, SpectralClustering, SpectralBiclustering, SpectralCoclustering, FeatureAgglomeration`
**sklearn.mixture:** `GaussianMixture, BayesianGaussianMixture` (DP-GMM)

| Tambahan | Paket | Kenapa menarik |
|---|---|---|
| k-medoids (PAM, alternating, FasterPAM) | `kmedoids`, `scikit-learn-extra` | Medoid = citra nyata → langsung jadi eksemplar klaster |
| Spherical k-means / von Mises-Fisher mixture | `spherecluster`, R `movMF` | Analog cosine yang benar untuk GMM — sentuhan canggih |
| Fuzzy C-Means | `skfuzzy` | Keanggotaan lunak; objek ambigu dapat skor, bukan dipaksa masuk satu klaster |
| Genie / Genie+ | `genieclust` | Hierarchical yang tahan outlier, sering mengalahkan Ward |
| CURE, ROCK, BSAS/MBSAS/TTSAS, SyncNet, X-Means, G-Means | `pyclustering` | Keluarga klasik lengkap |
| ★ Leiden / Louvain | `leidenalg`+`igraph` | **k tidak perlu ditentukan** (pakai resolution). Default komunitas single-cell |
| InfoMap | `infomap` | Map-equation, sering menemukan komunitas lebih halus |
| PhenoGraph | `PhenoGraph` | kNN graph berbobot Jaccard + Louvain, teruji untuk embedding dimensi tinggi |
| PARC | `parc` | kNN + hubness pruning + Leiden |
| Label Propagation / Girvan-Newman / Spectral graph | `scikit-network`, `networkx` | Alternatif komunitas |
| ★ SpectralNet | `spectralnet` | Aproksimasi neural spectral clustering, generalisasi out-of-sample. Salah satu pip install terbersih di daftar ini |
| Mapper / persistent homology | `giotto-tda` | Topological data analysis — sudut pandang yang hampir pasti tidak dipakai tim lain |

**Semua sudah terpasang dan terverifikasi di container sesi ini.**

## 4.2 Clustering di atas fitur foundation model frozen — ini kelas yang relevan

| Metode | Tahun/Venue | Mekanisme | Repo | [F]? | Compute | Benchmark |
|---|---|---|---|---|---|---|
| ★ **TEMI** | 2023 BMVC | Self-distillation, banyak cluster head, loss pointwise-MI berbobot ensemble guru | `HHU-MMBS/TEMI-official-BMVC2023` | ✅ native (`--precomputed`) | **4 GB VRAM**, 5 mnt 1 head / 45 mnt 50 head | CIFAR-10 94,5 · STL-10 98,5 · ImageNet 61,6 |
| ★ **TURTLE** | 2024 ICML | Optimasi bi-level: cari *pelabelan* yang menginduksi classifier linear margin-maksimal di **satu atau lebih** ruang frozen | `mlbio-epfl/turtle` | ✅ native | **<5 menit di ImageNet** | ImageNet 72,9 · CIFAR-10 99,3 · CIFAR-100 87,1 |
| **ICCE** | 2025 arXiv 2511.16213 | 50 cluster head (loss TEMI + Sinkhorn) → **ensembling hipergraf Strehl-Ghosh** → self-training | kode diumumkan | ✅ | A100 24 j (ImageNet); dataset kecil jauh lebih murah | **ImageNet 70,36 · CIFAR-10 99,20 · STL-10 99,58** |
| **CPP** | 2023 | Maximal Coding Rate Reduction di fitur CLIP frozen + self-labeling doubly-stochastic. **Punya algoritma estimasi k** | `LeslieTrue/CPP` | ✅ native | 15–20 epoch | CIFAR-10 97,4 · ImageNet 66 |
| **SCANv2** | 2024 arXiv 2406.01203 | Loss SCAN + self-distillation murni di fitur precomputed | paper saja | ✅ | murah | **ImageNet 69,79** vs TEMI 66,40 vs **k-means 62,72** |
| ★ **TAC** | 2024 ICML | Retrieve noun WordNet diskriminatif sebagai *text counterpart*; mutual distillation lintas modal antara dua MLP head | `XLearning-SCU/2024-ICML-TAC` | ✅ native | **1 menit di CIFAR-10** | CIFAR-10 91,9 · ImageNet 58,2. **Bonus: nama klaster otomatis** |
| **SIC** | 2023 AAAI | Petakan embedding CLIP ke ruang semantik/teks, pseudo-label, self-training | `Bruce-XJChen/SIC` | ✅ | ringan | CIFAR-10 92,6 · ImageNet 47,0 |
| **SCP** "Keep it Light!" | 2025 | CLIP/DINO frozen + MLP 5 layer; CE pasangan augmented + confidence + entropy | belum rilis | ✅ adapt | **~1 menit CIFAR-10 di L4** | CIFAR-10 97,5 (CLIP L/14) |
| **PRO-DSC** | 2025 ICLR | Deep subspace clustering: representasi terstruktur + koefisien self-expressive, dengan regularizer anti-collapse | `mengxianghan123/PRO-DSC` | ✅ native (kirim fitur CLIP siap pakai) | sedang | CIFAR-10 87,1 · ImageNet-10 99,0 |
| **CAE** | 2025 | **Training-free.** Fusi semantik caption + noun via optimal transport | belum rilis | ✅ | **CPU-feasible** | CIFAR-10 81,7 · ImageNet 53,1 |
| **GradNorm** | 2025 | Filter noun positif lewat magnitudo gradien | diumumkan | ✅ | A100 30 ep | CIFAR-10 91,1 · ImageNet 52,6 |
| **GSEC** | 2026 | Caption adaptif dari MLLM + bi-layer ensemble | `2017LI/GSEC` | ✅ | sedang | ImageNet 62,5 |
| **SEIC** | 2025 | Tahap 1 konsistensi semantik lintas modal (frozen); tahap 2 LoRA | belum rilis | tahap 1 ✅ | tahap 1 **5 mnt RTX 3090** | CIFAR-10 97,8 · ImageNet-Dogs 91,0 |
| **MIM-Refiner** | 2025 ICLR | Pasang ensemble head kontrastif di layer *tengah* model MIM | `ml-jku/MIM-Refiner` | konsumsi ✅ | berat untuk refine | ImageNet clustering 67,4 |
| **DCBoost** | 2025 | **Plug-in bebas parameter** yang mem-boost model deep clustering apapun | `l-h-y168/DCBoost` | ✅ | ringan | — |

**Angka paling berguna di seluruh tabel ini: k-means polos di fitur frozen yang bagus mencapai 62,7% di ImageNet 1000-kelas — ~90% akurasi SOTA dengan ~0,1% compute.** Baseline sederhana bukan barang buangan.

## 4.3 Deep clustering berbasis autoencoder — semuanya ada di ClustPy

`pip install clustpy` — **sudah terpasang dan terverifikasi**. API gaya scikit-learn.

- `clustpy.deep`: **ACeDeC, AEC, DCN, DDC, DEC, DKM, DeepECT, DipDECK, DipEncoder, ENRC, IDEC, N2D, VaDE**
- `clustpy.partition`: **DipExt, DipInit, DipMeans, DipNSub, GMeans, GapStatistic, LDAKmeans, PGMeans, ProjectedDipMeans, SkinnyDip, SpecialK, SubKmeans, UniDip, XMeans**
- `clustpy.hierarchical`: **Diana**
- ★ `clustpy.alternative`: **AutoNR, ClusteringInOrthogonalSpaces, NrKmeans, OrthogonalClustering** ← **ini persis mesin "beberapa lensa" versi klasik, tersedia satu pip install**
- `clustpy.density`, `clustpy.metrics`, `clustpy.data`

Sorotan:
- **DeepECT** — menumbuhkan **pohon klaster** bersama AE → dapat dendrogram, sempurna untuk taksonomi e-waste bertingkat
- **DipDECK** — over-cluster lalu merge pasangan yang gagal uji dip → **estimasi k otomatis**
- **SHADE** — deep density-based, tanpa k
- **N2D** — AE → UMAP → GMM. Sangat murah, jalan di fitur apa pun. `pip install n2d`
- **ENRC** — Deep Embedded Non-Redundant Clustering: beberapa clustering *alternatif* di sub-ruang ortogonal

## 4.4 Deep clustering yang butuh retrain backbone — [RW]

Untuk bagian related work, bukan untuk pipeline kita.

**Self-labelling:** DeepCluster, DeepCluster-v2, SeLa (optimal transport / Sinkhorn), SwAV, ODC, PIC, ClusterFit, DeeperCluster
**Kontrastif:** IIC, PICA, SCAN, CC, TCL, GCC, NNM, IDFD, MiCE, SPICE, ProPos, TCC, SeCu, TWIST, DivClust, DTI-Clustering, DPAC, RPSC, LFSS
**Ketidakseimbangan kelas:** ★ **P²OT / PPOT** (ICLR 2024) — Progressive Partial Optimal Transport untuk deep clustering **tidak seimbang**; ini solver assignment, jadi sebenarnya [F]-adapt. Relevan karena klaster e-waste kita pasti tidak seimbang.
**Subspace:** SSC, LRR, EnSC, DSC-Net, SENet, EDESC, NMCE

## 4.5 ★ Multiple clustering / alternative clustering — inti kontribusi kita

Ini literatur yang menopang ide "empat lensa".

### Klasik (shallow)
| Metode | Tahun | Mekanisme |
|---|---|---|
| CIB (Gondek & Hofmann) | 2003 NIPS | Conditional information bottleneck: partisi informatif tentang X tapi *conditionally independent* dari partisi yang sudah diketahui |
| COALA (Bae & Bailey) | 2006 ICDM | Ubah objek di clustering lama jadi **cannot-link constraints**, lalu cluster ulang |
| Orthogonal projection (Cui, Fern, Dy) | 2007 ICDM | Proyeksikan data ke **komplemen ortogonal** sub-ruang clustering sebelumnya |
| Dec-kmeans (Jain, Meka, Dhillon) | 2008 SDM | Dua objektif k-means dioptimasi bersama dengan **penalti dekorelasi** antar sub-ruang mean |
| Dang & Bailey | 2010 | EM meminimalkan mutual information antara partisi baru dan referensi |
| MSC (Hu, Qian, Nie) | 2016/17 | Cari sub-ruang fitur yang memaksimalkan **eigengap Laplacian** → beberapa clustering *stabil* |
| MNMF (Yang & Zhang) | 2017 ML | NMF + regularizer inner-product antar matriks similarity untuk menghukum redundansi |
| ★ MCV (Guérin & Boots) | 2018 BMVC | **Perlakukan berbagai CNN pretrained sebagai view berbeda** — tiap view menghasilkan clustering berbeda. Preseden langsung untuk cache 8-backbone kita |
| MVMC | 2019 IJCAI | Self-representation multi-view + constraint diversitas |

### Deep
| Metode | Tahun/Venue | Mekanisme | Repo |
|---|---|---|---|
| ENRC | 2020 AAAI | AE dilatih bersama beberapa objektif clustering di sub-ruang latent yang disjoint | ada di ClustPy |
| iMClusts | 2022 | AE + multi-head attention → beberapa embedding beragam ⚠️ *judul/venue belum terverifikasi* | — |
| DMClusts | ~2021 | Deep matrix factorization ke sub-ruang beragam | — |
| AugDMC | 2023 IJCNN | **Augmentasi mendefinisikan kriteria**: tiap keluarga augmentasi mempertahankan aspek berbeda | `Alexander-Yao/AugDMC` |
| DDMC | 2024 | Variational EM, representasi disentangled kasar + halus | — |
| ★ **DivClust** | 2023 CVPR | Backbone bersama + K projection head dengan **loss diversitas yang membatasi rata-rata NMI antar-clustering** di bawah target D_T yang diset user | `ManiadisG/DivClust` |

**DivClust adalah yang harus kita "curi" untuk cerita komplementaritas** — ia menjadikan "rata-rata NMI berpasangan antar partisi" sebagai besaran eksplisit yang bisa dikontrol; itu persis angka yang kita butuhkan untuk membuktikan empat lensa tidak redundan.

### ★★ Personalized multiple clustering lewat CLIP proxy — ini setup kita persis
| Metode | Venue | Mekanisme | Repo | Kriteria |
|---|---|---|---|---|
| **Multi-MaP** | CVPR 2024 | GPT-4 memperluas keyword user jadi reference words; **proxy word embedding** dipelajari di prompt "fruit with the *color* of \*" | `Alexander-Yao/Multi-MaP` | keyword user |
| **Multi-Sub** | NeurIPS 2024 | Proxy per-citra = kombinasi berbobot embedding reference-word, disejajarkan di **sub-ruang yang didefinisikan user** | `Alexander-Yao/Multi-Sub` | keyword user |
| **Multi-DProxy** | AAAI 2026 | **Gated cross-modal fusion** + proxy tekstual yang di-refine iteratif | supplementary | keyword → GPT-4 |
| **ESMC** | AAAI 2026 | Pakai **hidden state token teks dari MLLM** (LLaVA-1.5-7B) sebagai embedding spesifik-kriteria | `JCSTARS/Embedding-Selective-Multiple-Clustering` | kalimat prompt |
| **Agent-Centric PMC** | 2025 | Agen MLLM menyusuri graf relasional berbobot embedding terkondisi-kriteria | belum rilis | keyword abstrak |

Semua lini ini **CLIP frozen + objek kecil yang dipelajari di atasnya**. Multi-MaP jalan di satu RTX 2080 Ti untuk 8k citra → Kaggle-feasible untuk 4.179 citra kita. **Menukar CLIP → SigLIP2 adalah sumbu kebaruan kecil yang sah dan bisa dipertahankan.**

### Protokol evaluasi standar yang harus kita tiru
Seluruh rantai Multi-MaP → Multi-Sub → DDMC → Multi-DProxy → ESMC memakai suite yang sama:

| Dataset | #citra | Kriteria |
|---|---|---|
| ALOI | 288 | Color; Shape |
| Fruit | 105 | Color; Species |
| **Fruit360** | **4.856** | Color; Species ← ukuran hampir sama dengan data kita |
| Card | 8.029 | Order (rank); Suits |
| CMUface | 640 | Emotion; Glasses; Identity; Pose |
| Stanford Cars | 1.200 | Color; Type |
| Flowers | 1.600 | Color; Species |

Bundel unduhan: `faculty.washington.edu/juhuah/images/AugDMC_datasets.zip`

**Aturan protokol:** (1) k per kriteria diset ke jumlah klaster ground-truth — **tidak ada yang mengestimasi k di literatur ini**; (2) laporkan **NMI dan RI per kriteria terpisah** (tambahkan AMI/ARI, gratis dan lebih baik); (3) untuk metode stokastik, jalankan k-means 10× lalu rata-rata; (4) baseline standar yang diharapkan reviewer 2026: **MSC, MCV, ENRC, iMClusts, AugDMC, DDMC, Multi-MaP, Multi-Sub** + ablasi `CLIP_gpt` dan `CLIP_label`.

**Dataset sanity-check tambahan: ★ Clevr-4** (Vaze et al., NeurIPS 2023, *No Representation Rules Them All in Category Discovery*) — 4 kriteria **benar-benar ortogonal** (Shape, Color, Texture, Count) atas citra yang sama. `robots.ox.ac.uk/~vgg/data/clevr4` · `github.com/sgvaze/clevr4`. **Kalau trik proyeksi kita bekerja, ia harus bisa memulihkan keempat kriteria Clevr-4.** Ini uji validitas mekanisme yang paling bersih dan murah.

## 4.6 Multi-view clustering (beda dengan multiple clustering!)

**Framing yang harus ditulis eksplisit:** multi-view clustering menggabungkan banyak view jadi **satu** partisi konsensus. Setting kita kebalikannya — **satu himpunan fitur, banyak kriteria, banyak partisi.** Menyebutkan ini adalah cara terbersih memposisikan kerja kita terhadap literatur MVC yang jauh lebih besar.

| Metode | Venue | Repo |
|---|---|---|
| Co-regularized Multi-view Spectral Clustering | NeurIPS 2011 | `lkrmbhlz/CRMVSC` |
| DEMVC | Inf. Sciences 2021 | `SubmissionsIn/DEMVC` |
| SiMVC / CoMVC | CVPR 2021 | `DanielTrosten/mvc` |
| MFLVC | CVPR 2022 | `SubmissionsIn/MFLVC` |
| DealMVC | ACM MM 2023 | ACM DL |
| SCMVC / DCMVC | 2024 | `SongwuJob/SCMVC`, `tweety1028/DCMVC` |

Daftar kurasi: `wangsiwei2010/awesome-multi-view-clustering`, `jinjiaqi1998/Awesome-Deep-Multi-View-Clustering`. Survei: *A Comprehensive Survey on Multi-View Clustering*, IEEE TKDE 2023, DOI 10.1109/TKDE.2023.3270311.

## 4.7 Consensus / ensemble clustering

| Metode | Tahun | Mekanisme |
|---|---|---|
| ★ CSPA / HGPA / MCLA (Strehl & Ghosh) | 2002 JMLR 3 | Konsensus sebagai partisi hipergraf |
| Evidence Accumulation (Fred & Jain) | 2005 TPAMI | **Matriks ko-asosiasi** menghitung seberapa sering dua titik se-klaster, lalu hierarchical di atasnya |
| LWEA / LWGP (Huang, Wang, Lai) | 2018 IEEE T-Cyb | Bobot per-*klaster* (bukan per-partisi) berdasarkan ketidakpastian ensemble |
| HBGF | 2004 ICML | Graf bipartit instance–cluster |

**Python:** `ClusterEnsembles` / `ensembleclustering` (CSPA, HGPA, MCLA, HBGF, NMF) — ⚠️ gagal `pip install` di container ini, ambil dari GitHub `827916600/ClusterEnsembles`; `OpenEnsembles` (JMLR 19, 2018) `NaegleLab/OpenEnsembles`. Atau: matriks ko-asosiasi manual — 4200² float32 = 71 MB, ~20 baris kode.

**Pola TGAICC yang layak ditiru:** perluas tiap kriteria jadi ensemble prompt → cluster tiap prompt → **konsensus *di dalam* kriteria** dengan MCLA/HBGF → AMI antar-kriteria untuk bukti komplementaritas. Ini penguat stabilitas yang murah dan sudah tersitasi, sekaligus memberi ablasi gratis (prompt tunggal vs ensemble prompt).

## 4.8 Metode untuk "k tidak diketahui"

| Metode | Mekanisme | Paket |
|---|---|---|
| ★ HDBSCAN | Hierarki densitas + ekstraksi berbasis stabilitas, **label noise eksplisit** | `hdbscan` / sklearn |
| ★ Leiden / InfoMap | Resolution / description-length, bukan k | `leidenalg`, `infomap` |
| DipDECK / DipMeans / ProjectedDipMeans / SkinnyDip / Dip'n'sub / UniDip | Uji dip Hartigan untuk unimodalitas | ClustPy, `unidip` |
| X-Means / G-Means / PG-Means | BIC / uji normalitas Anderson-Darling / uji KS pada proyeksi acak | ClustPy, `pyclustering` |
| GapStatistic / SpecialK | Distribusi referensi / signifikansi statistik | ClustPy |
| DP-GMM (CRP) | `BayesianGaussianMixture(weight_concentration_prior_type='dirichlet_process')` — set K_max longgar, baca berapa komponen berbobot signifikan. ⚠️ sensitif ke `weight_concentration_prior`, jalankan sweep | sklearn |
| DeepDPM | Split/merge Metropolis-Hastings atas DP mixture | `BGU-CS-VIL/DeepDPM` — realistis GPU-only |
| DeepECT | Pohon klaster, potong di level manapun | ClustPy |
| CPP | Punya pengukuran "jumlah klaster optimal" berbasis rate-reduction | `LeslieTrue/CPP` |
| Eigengap heuristic | λ₁…λ_k kecil, λ_{k+1} relatif besar | `scipy.sparse.linalg.eigsh` |

---

# TAHAP 5 — Pemilihan k

| Metode | Referensi | Catatan | Paket |
|---|---|---|---|
| Elbow + **Kneedle** | Satopää et al., ICDCS 2011 | Normalisasi kurva ke unit square, cari maksimum lokal kurva selisih. **Laporkan sensitivitas S yang dipakai** — jawabannya bergeser | `kneed` ✅ |
| ★ **Gap statistic** | Tibshirani, Walther, Hastie 2001, JRSS-B 63(2):411 | **Pilihan distribusi referensi sangat menentukan**: `spaceH0="scaledPCA"` (uniform di bounding box setelah rotasi SVD — versi rekomendasi Tibshirani) vs `"original"` (naif, terlalu mudah menolak). **Aturan keputusan** (`maxSE`): `firstSEmax` (default), `Tibs2001SEmax`, `globalSEmax`, `firstmax`, `globalmax` — **pilihan aturan mengubah jawaban, sebutkan yang dipakai** | `gap-stat`, `gapstatistics`; referensi = R `cluster::clusGap` |
| gap* (tanpa logaritma) | Mohajer, Englmeier & Schmid | | |
| Weighted gap / DD-weighted gap | Yan & Ye 2007, *Biometrics* | | R `splits::ddwtGap` |
| ★ **Prediction strength** | Tibshirani & Walther 2005, JCGS 14(3):511 | Split 50/50, cluster keduanya, assign test ke centroid train, PS(k) = min proporsi pasangan yang tetap se-klaster. Ambil k terbesar dengan PS ≥ 0,8. **Kriteria out-of-sample, bukan rasio geometris in-sample** — salah satu yang paling bisa dipertahankan | R `fpc::prediction.strength`; ~30 baris di Python |
| ★ Stability selection (Ben-Hur, Elisseeff, Guyon) | PSB 2002 | Dua subsample tumpang tindih (f≈0,8), ukur kesepakatan di irisan. Plot **distribusi** similarity per k | manual |
| ★ Lange, Roth, Braun, Buhmann | Neural Computation 2004 16(6):1299 | Clustering jadi klasifikasi: cluster train → fit classifier → prediksi test → banding dengan clustering test langsung. **Dinormalisasi terhadap baseline label acak** — langkah kritis | `reval` |
| Wang (2010) | *Biometrika* 97(4):893 | Perbaiki inkonsistensi instability naif dengan split tiga-arah | |
| Clest | Dudoit & Fridlyand 2002, *Genome Biology* | Split 2-fold berulang + bandingkan ke distribusi null referensi | |
| Jump statistic | Sugar & James 2003, JASA 98:750 | Distorsi d_k ditransformasi Y_k = d_k^(−p/2), k* = argmax selisih. Justifikasi rate-distortion. ~20 baris | |
| BIC / AIC (GMM) | | ⚠️ **Di d=1000 dengan n=4200, GMM full-covariance punya ~500k parameter per komponen — degenerate.** Wajib PCA ≤50 D atau `covariance_type='diag'` | sklearn |
| ★ **ICL** | Biernacki, Celeux, Govaert 2000, TPAMI 22(7):719 | BIC − entropi posterior assignment. **Menghukum komponen yang tumpang tindih** → memilih klaster yang *terpisah baik* ketimbang model densitas terbaik. Karena kategori e-waste adalah pengelompokan konseptual bukan komponen Gaussian literal, **ICL lebih tepat dari BIC** — poin yang kuat | R `mclust::mclustICL`; 5 baris dari `predict_proba` |
| NbClust 30 indeks (voting) | Charrad et al. 2014, JSS 61(6) | Retoris kuat ("21 dari 30 indeks memilih k=6"). ⚠️ **Dua caveat wajib**: (a) di n=4200 `alllong` akan OOM/hang → subsample ≤1.500 dan jalankan di banyak subsample, laporkan *distribusi* vote; (b) ke-30 indeks sangat berkorelasi dan berbagi bias sferis yang sama — **bukan 30 pendapat independen**, katakan itu | R `NbClust` via rpy2 |
| SigClust / sigclust2 | Liu et al. 2008 JASA; Kimes et al. 2017 *Biometrics* | Uji H₀ "data ini satu Gaussian" dengan koreksi HDLSS. **sigclust2 menguji setiap node dendrogram dengan kontrol FWER** → cluster hierarkis lalu potong di mana split berhenti signifikan. Cocok sekali untuk kita | R `sigclust`, `sigclust2` |
| sc-SHC | Grabski & Irizarry, *Nature Methods* 2023 | Uji hipotesis terintegrasi ke hierarchical clustering | `igrabski/sc-SHC` |

⚠️ **"SpecialK"** ada sebagai kelas di ClustPy, tapi rujukan kanoniknya sulit diverifikasi lewat pencarian web. Verifikasi sitasi sebelum masuk naskah.

---

# TAHAP 6 — Validasi & null model

## 6.1 Anggaran compute (baca duluan)

Di n = 4.179, matriks jarak penuh = 17,5 juta pasangan = **141 MB float64 / 71 MB float32**, dihitung dalam 2–5 detik via BLAS. **Hitung `D` sekali, cache, lalu pakai `metric='precomputed'` di mana-mana.** Fakta ini membuat hampir semua indeks O(n²) terjangkau.

| Kelas metode | Biaya di n=4200, d=1000 | Vonis 2-core |
|---|---|---|
| Matriks jarak (sekali) | 2–5 s, 141 MB | Lakukan sekali |
| Silhouette / Dunn / C-index / McClain-Rao / Point-biserial / Wemmert-Gancarski di `D` precomputed | <1–3 s per partisi | Gratis |
| CH / DB / Ball-Hall / Ray-Turi / Xie-Beni | O(nkd) ≈ 0,1 s | Gratis |
| Gamma / G+ / Tau | naif O(n⁴) = mustahil; implementasi tersortir O(n² log n) = ok | **Hanya lewat kode C clusterCrit, jangan Python naif** |
| Keluarga determinan (Scott-Symons, Rubin, Det-Ratio, Trace WiB) | butuh W invertible → butuh **d < ukuran klaster terkecil** | **Mati di d=1000. PCA ke ≤30–50 D dulu** |
| DBCV / DCSI / CDbw | O(n²) + MST ≈ 5–20 s per partisi | Aman |
| Gap statistic, B=50 × k=2..20 | 950 run k-means | ~40 mnt di d=1000; **~1 mnt di d=50** → PCA dulu |
| Consensus, 100 subsample × 19 k | 1.900 run + matriks konsensus | ~1–2 jam di d=50 |
| NbClust `alllong` di n=4200 | memori kuadratik + Gamma/Tau | **Akan hang/OOM** — subsample ≤1500 |

**Urutan pipeline yang tidak bisa ditawar: L2-normalisasi → PCA 50–100 D (opsional whiten) → baru jalankan indeks.**

## 6.2 Indeks validitas internal

**Keluarga centroid/scatter — HANYA VALID untuk klaster kira-kira sferis, konveks, berukuran mirip**

| Indeks | Intuisi | Bias yang diketahui | Python |
|---|---|---|---|
| Silhouette (Rousseeuw 1987) | (b−a)/max(a,b) | Condong ke **k=2**; menghukum klaster memanjang/manifold | sklearn |
| Calinski-Harabasz (1974) | [tr(B)/(k−1)]/[tr(W)/(n−k)] | Bias sferis/ukuran-sama kuat | sklearn |
| Davies-Bouldin (1979) | rata-rata max_j (S_i+S_j)/M_ij | Pasangan-terburuk → satu klaster jelek mendominasi; rapuh outlier | sklearn |
| Ball-Hall (1965) | W_k / k | **Monoton turun terhadap k** → wajib aturan selisih-maksimum, jangan argmax | `permetrics`, `genieclust` |
| Banfield-Raftery (1993) | Σ n_k·log(tr(W_k)/n_k) | Meledak pada klaster singleton | `permetrics` |
| Ratkowsky-Lance (1978) | mean_j √(BSS_j/TSS_j) ÷ √k | Sensitif skala | clusterCrit |
| Scott-Symons (1971) | Σ n_k·log det(W_k/n_k) | **Butuh n_k > d. Mati di d=1000** | clusterCrit |
| Trace W / Trace WiB / Rubin / Ksq-DetW | | Monoton atau butuh invertibilitas | clusterCrit |
| PBM (2004) | ((1/k)(E₁/E_k)D_k)² | Suku separasi-maksimum rapuh outlier | clusterCrit |
| Ray-Turi (1999) | (WCSS/n)/min‖c_i−c_j‖² | Analog crisp Xie-Beni | clusterCrit |
| Xie-Beni (1991) | | Asalnya **fuzzy (FCM)**; cenderung turun dengan k | `permetrics`, `skfuzzy` |
| ★ **Wemmert-Gancarski** | 1 − (1/n)Σ ‖x−c_own‖/min_{j≠own}‖x−c_j‖ | Margin per-titik ke centroid terbaik-kedua. Murah O(nkd), lebih robust dari DB. **Jarang dipakai → pembeda bagus di paper** | clusterCrit |
| Log-SS-Ratio | log(BGSS/WGSS) | Monoton | `permetrics` |
| Hartigan (1975) | H(k)=(W_k/W_{k+1}−1)(n−k−1), stop ≤10 | Cutoff "10" rule of thumb | `permetrics`, NbClust |
| Krzanowski-Lai (1988) | \|DIFF(k)/DIFF(k+1)\| | Tidak terdefinisi di k=1; tidak stabil saat DIFF≈0 | NbClust |
| CCC (Sarle 1983) | vs null uniform | Asumsi khas SAS | NbClust |
| Duda-Hart / pseudo-t² / Beale | uji split-vs-no-split | Asumsi Gaussian, kovarians sama | NbClust, `fpc` |

**Keluarga jarak berpasangan / berbasis rank — METRIC-AGNOSTIC (bisa cosine/geodesik), tapi tetap bias kompaktnes**

| Indeks | Intuisi | Bias | Paket |
|---|---|---|---|
| Dunn (1973) | min jarak antar-klaster ÷ max diameter intra | **Sangat rapuh outlier** — satu titik nyasar menentukan penyebut. Tier *terburuk* di Arbelaitz 2013 | `validclust` ✅, `permetrics` |
| ★ **Generalized Dunn GDI11–GDI53** (Bezdek & Pal 1998) | 5 pembilang × 3 penyebut, mengganti min/max dengan mean/jarak-centroid | **Memperbaiki kerapuhan Dunn** — varian GDI masuk tier *teratas* | clusterCrit (15 varian); `genieclust` |
| ★ Graph Dunn: DMST / DRNG / DGG (Pal & Biswas 1997) | Diameter intra dari MST / relative-neighbourhood / Gabriel graph | **Benar-benar shape-agnostic** — salah satu dari sedikit indeks klasik yang aman untuk klaster non-konveks | `scipy.sparse.csgraph.minimum_spanning_tree` |
| C-index (Hubert & Levin 1976) | (S_w−S_min)/(S_max−S_min) | Kuat di Milligan-Cooper 1985, lemah di Arbelaitz 2013 | clusterCrit, `fpc` |
| Point-biserial (Milligan 1980) | korelasi vektor jarak vs indikator biner se-klaster | Robust, mudah dijelaskan | clusterCrit |
| McClain-Rao (1975) | (mean jarak intra)/(mean jarak antar) | **Bias monoton ke k kecil** | clusterCrit |
| Baker-Hubert Gamma / G+ / Tau | concordant/discordant pair-of-pairs | O(n² log n) terbaik; NbClust hang di n≈4200 | clusterCrit, `permetrics` |
| Frey & Van Groenewoud (1972) | rasio antar/intra lintas level hierarki | | NbClust |
| Hubert Γ / D-index | grafis (elbow visual) | tidak otomatis | NbClust |

**Hibrida densitas-scatter**

| Indeks | Intuisi | Catatan |
|---|---|---|
| ★ S_Dbw (Halkidi & Vazirgiannis 2001) | Scat(k) + Dens_bw(k); Dens_bw menghitung titik dekat *midpoint* antar centroid | Terbaik di Liu et al. 2010 untuk noise/densitas/skew/subcluster. `s-dbw` di PyPI |
| SD index (Halkidi 2000) | Dis(k_max)·Scat(k) + Dis(k) | **Bergantung k_max** → nilai tidak sebanding lintas rentang k berbeda |

## 6.3 ⚠️ Indeks yang benar-benar valid untuk klaster non-konveks / density-based

**Ini pembeda metodologis terpenting di seluruh dokumen.** Semua indeks di atas memakai centroid atau mean jarak intra, jadi semuanya akan *menghukum* clustering crescent/manifold/nested yang benar. **Kalau menjalankan HDBSCAN atau spectral lalu mengevaluasinya dengan silhouette, itu menilai ikan dari kemampuan memanjat pohon** — dan juri memberi nilai kepada tim yang menyatakan ini secara eksplisit.

| Indeks | Referensi | Mekanisme | Python |
|---|---|---|---|
| ★★ **DBCV** | Moulavi, Jaskowiak, Campello, Zimek, Sander, **SDM 2014** | Graf mutual-reachability → MST per klaster. Density Sparseness = edge MST internal maksimum; Density Separation = mutual-reachability minimum antar node internal dua klaster. Range [−1,1], maks | ★ `hdbscan` → **`clusterer.relative_validity_`** (aproksimasi MST, praktis gratis); `FelSiq/DBCV`; `k-DBCV`; `dbcv-revisited` |
| DCSI | Gauss, Scheipl, Herrmann 2023 (arXiv 2310.12806) | Sep = jarak min antar **core point** klaster berbeda; Conn = edge MST maks di antara core point satu klaster; DCSI = q/(1+q) | R `JanaGauss/dcsi` (~40 baris untuk di-port) |
| CDbw | Halkidi & Vazirgiannis 2008, PRL | **Beberapa representatif per klaster** (bukan satu centroid) | `cdbw` PyPI, R `fpc::cdbw` |
| VIASCKDE | Şenol 2022 | Silhouette per-titik **dibobot densitas KDE lokal** | repo penulis; mudah diimplementasi ulang |
| CVDD | Hu & Zhong 2019, IEEE Access | Jarak komposit path-based + densitas relatif | `hulianyu/CVDD` (MATLAB) |
| CVNN | Liu et al. 2013 | Separasi = fraksi kNN tiap titik di luar klasternya | ada di repo DCSI |
| DSI / CVI-DSI | Guan & Loew 2020 | Jarak Kolmogorov-Smirnov antara *distribusi* jarak intra-kelas dan antar-kelas | `ShuyueG/CVI_using_DSI` |

**Bukti head-to-head (PeerJ CS 2025):** *"DBCV lebih informatif dari DCSI, CDbw, dan VIASCKDE untuk penilaian internal klaster cekung dan berbasis densitas"* — peerj.com/articles/cs-3095. DBCV konsisten dengan ARI di 9/15 uji vs DCSI 7/15, VIASCKDE 4/15, CDbw 0/15. **Kutip paper ini untuk membenarkan DBCV sebagai indeks utama untuk kandidat density-based.**

Caveat DBCV yang harus dinyatakan: (a) implementasi berbeda dalam menangani titik noise HDBSCAN (label −1) — nyatakan apakah dibuang atau diperlakukan sebagai klaster; (b) `relative_validity_` adalah aproksimasi — laporkan keduanya; (c) DBCV tidak sebanding lintas metrik jarak berbeda.

## 6.4 Indeks modern (2020+)

- **CDI** (Fang, Xie et al., *Genome Biology* 2022) — negative log-likelihood terpenalti, ℓ̃ = −2ℓ̂ + c_pen·d. **Ide yang bisa dipindahkan:** ganti negative-binomial dengan likelihood Gaussian/vMF per klaster di embedding → kriteria internal berbasis model, bukan rasio geometris ad-hoc.
- **DuNN / DuNNowa** (Gagolewski, Bartoszuk, Cena 2021, *Information Sciences*) — Dunn dibangun ulang dengan operator **OWA** dan graf near-neighbour, dirancang eksplisit untuk "memisahkan sub-ruang padat dengan bentuk beragam". Di `genieclust.cluster_validity` (Python + R).
- **GPVI** (2026), **"An internal validity index for arbitrarily shaped clusters"** (ESWA 2023).

## 6.5 ★ Kritik yang wajib ditanggapi (ini yang memberi nilai metodologi)

**Gagolewski, Bartoszuk & Cena, "Are Cluster Validity Measures (In)valid?", *Information Sciences* 2021** (arXiv 2208.01261) — mereka **mengoptimasi langsung** CH, Dunn, DB, dan Silhouette sebagai fungsi objektif, dan menemukan partisi hasilnya "cocok dengan pengetahuan pakar dengan sangat buruk."
→ **Implikasi: CVI adalah *diagnostik*, bukan *objektif*. Jangan pernah argmax satu indeks lalu menyebutnya jawaban.** Framing jujur: pakai *panel* indeks yang membentang bias berbeda, laporkan di mana mereka sepakat dan di mana tidak, dan perlakukan ketidaksepakatan sebagai informasi.

**Arbelaitz et al. (2013), *Pattern Recognition*** — 30 indeks × 720 sintetis + 20 dataset nyata:
- Tier atas: Silhouette, DB, CH, varian generalized Dunn, COP, S_Dbw. **Silhouette satu-satunya yang melewati success rate 50%**
- Tier bawah: Dunn, Gamma, C-index, Negentropy Increment, OS-index
- **Noise:** semua indeks *kecuali Silhouette dan S_Dbw* turun ~3× di bawah 10% noise ← langsung relevan, scrape citra e-waste itu berisik
- **Overlap:** success rate rata-rata anjlok 52,9% → 17,6% saat klaster tumpang tindih
- Kebanyakan CVI berkinerja *paling buruk* justru untuk partisi k-means

Juga: Todeschini et al., "Extended multivariate comparison of **68** cluster validity indices", *Chemometrics and Intelligent Laboratory Systems* 2024; Botta-Dukát 2023 (arXiv 2308.03894).

## 6.6 Stabilitas — dan kritik teoretisnya

### Protokol praktis
**`fpc::clusterboot()`** — Hennig (2007), *CSDA* 52:258. Skema resampling: `boot`, `subset`, `noise`, `jitter`, `bojit`. Ukuran: **rata-rata Jaccard** tiap klaster asli terhadap klaster paling mirip di tiap solusi resampled.

**Ambang interpretasi Hennig — kutip verbatim di paper:**
- **≤ 0,5** → klaster **"dissolved"**
- **< 0,6** → jangan dipercaya
- **0,6–0,75** → pola ada, keanggotaan dipertanyakan
- **≥ 0,75** → **klaster valid dan stabil**
- **≥ 0,85** → **sangat stabil**

**Ini per-klaster, bukan per-partisi** — itu kekuatan terbesarnya. Kita bisa menulis: *"klaster 1,2,4,5 stabil (J = 0,81–0,93); klaster 3 dissolved (J = 0,47) dan kami laporkan sebagai residual heterogen."* Itu persis nuansa yang dihargai juri. Implementasi ulang di Python ~50 baris (`sklearn` + `scipy.optimize.linear_sum_assignment`).

### Consensus clustering dan kegagalannya yang terdokumentasi
- Monti, Tamayo, Mesirov, Golub (2003), *ML* 52:91 — matriks ko-asosiasi, CDF konsensus, Δ(K).
- ⚠️ **Şenbabaoğlu, Michailidis, Li (2014), "Critical limitations of consensus clustering in class discovery", *Scientific Reports* 4:6207 — WAJIB DIBACA DAN DIKUTIP.**
  - Consensus clustering *"sangat sensitif"* dengan **false-positive rate ekstrem**: ia memartisi data acak tanpa struktur menjadi klaster yang tampak rapi dan meyakinkan
  - **Δ(K) tidak reliabel** — selalu memberi elbow di K≈3–4 *terlepas dari K sebenarnya*
  - **Heatmap konsensus menyesatkan**: data null yang mempertahankan struktur korelasi fitur menghasilkan heatmap yang tak bisa dibedakan dari struktur nyata
  - **Solusi: PAC (Proportion of Ambiguous Clustering)** = fraksi pasangan dengan konsensus di (0,1 · 0,9). Minimalkan
- ★ **M3C** (John et al. 2020, *Sci Rep* 10:1816) — null via **simulasi berbasis PCA** (tarik skor PC acak yang cocok dengan SD PC teramati, transform balik) sehingga null mempertahankan struktur kovarians tapi tanpa klaster. **RCSI_K = log₁₀(mean PAC referensi) − log₁₀(PAC nyata)**; p-value Monte Carlo. **Port ide ini ke Python — ~60 baris, dan ini potongan metodologi paling bisa dipertahankan di seluruh dokumen.**
- **rPAC** (MultiK, Genome Biology 2021) — menghapus bias PAC sendiri terhadap K.

### ⚠️⚠️ ARGUMEN MELAWAN STABILITAS SEBAGAI KRITERIA PEMILIHAN MODEL

Ini bagian yang paling membedakan paper serius. Literatur stabilitas punya bantahan teoretis yang terkenal dan hampir tidak ada peserta kompetisi yang tahu.

**(1) Ben-David, von Luxburg & Pál (COLT 2006), "A Sober Look at Clustering Stability"**
> **Teorema inti:** secara asimtotik, stabilitas **sepenuhnya ditentukan oleh perilaku fungsi objektif yang diminimalkan algoritma**. Kalau objektif punya **minimizer global unik**, algoritma **stabil — terlepas dari apakah k benar**. Kalau minimizer-nya tidak unik, ia **tidak stabil — lagi-lagi terlepas dari apakah k benar**.

> **Konsekuensi:** stabilitas mengukur properti **lanskap optimasi**, bukan jumlah klaster dalam data. Ketidakstabilan terutama muncul dari **simetri distribusi data** yang bisa sama sekali tidak berhubungan dengan k. Berlaku untuk center-based *dan* spectral.

**(2) Ben-David, Pál & Simon (COLT 2007)** — saat n→∞ dengan optimum unik, **instability → 0 untuk semua k**, jadi stabilitas kehilangan seluruh daya diskriminatif persis di limit sampel besar.

**(3) Ben-David & von Luxburg (COLT 2008)** — stabilitas **dibatasi atas** oleh massa probabilitas di tabung sekitar batas keputusan — tapi implikasinya **satu arah saja**. "Stabil ⇒ batas di daerah densitas rendah" **bukan teorema**.

**(4) von Luxburg (2010), *FnT ML* 2(3):235** — **jaminan konvergensi distribution-free itu mustahil**: ukuran sampel sebesar apa pun bisa menghasilkan nilai stabilitas yang menyesatkan. Rezim di mana stabilitas empiris "bekerja" adalah rezim *sampel kecil, pra-asimtotik* — persis rezim yang tidak dicakup teori manapun.

**(5) Confound praktis:** ketidakstabilan dari optimizer jelek (local minima k-means, varians seed) tak bisa dibedakan dari ketidakstabilan karena ambiguitas data. **Kontrol:** fix `n_init`, fix seed, pakai k-means++, dan **laporkan varians-seed terpisah dari varians-subsample**. Stabilitas juga tidak sebanding lintas k tanpa normalisasi — pelabelan acak ke k grup punya baseline instability makin tinggi seiring k naik. **Selalu normalisasi terhadap baseline label acak, atau pakai indeks *adjusted* (ARI/AMI, jangan Rand mentah atau NMI mentah).**

**Cara memakai stabilitas secara jujur — paragraf ini saja menjawab pertanyaan tersulit juri:**
> "Kami memakai stabilitas sebagai kondisi *perlu tetapi tidak cukup* dan sebagai diagnostik *per-klaster* (Hennig 2007), bukan sebagai kriteria pemilihan k. Mengikuti Ben-David, von Luxburg & Pál (2006), kami tidak memperlakukan stabilitas maksimal sebagai bukti k yang benar; kami memperlakukan stabilitas *rendah* sebagai bukti terhadap suatu klaster tertentu. Semua besaran stabilitas dinormalisasi terhadap null label acak dan null yang cocok-kovarians (gaya M3C), sehingga kami melaporkan stabilitas *relatif terhadap apa yang bisa dicapai secara kebetulan pada data tanpa struktur dengan bentuk dan struktur korelasi yang sama*."

## 6.7 Membangun null yang benar — diurut dari terlemah ke terkuat

| Null | Konstruksi | Mempertahankan | Menghancurkan | Vonis |
|---|---|---|---|---|
| Label acak | permutasi label klaster | ukuran klaster | segalanya | Baseline normalisasi ARI. **Perlu, tak pernah cukup** |
| Uniform hypercube | Uniform atas min/max per fitur | rentang marginal saja | semua korelasi | Versi asli Tibshirani; **terlalu mudah ditolak** di dimensi tinggi |
| PCA-aligned uniform ("scaledPCA") | center → SVD → uniform di bounding box terotasi → rotasi balik | geometri sumbu utama | struktur orde tinggi | Versi rekomendasi Tibshirani sendiri. **Standar minimum yang bisa diterima** |
| Rotasi acak | terapkan Q ortogonal acak | **semua jarak berpasangan persis** | tidak ada | ⚠️ **Tidak berguna sebagai null untuk indeks berbasis jarak** — kesalahan yang umum |
| ★ Marginal shuffle | permutasi tiap kolom secara independen | distribusi marginal tiap fitur *persis* | seluruh dependensi antar-fitur | **Murah, kuat, sepele dijelaskan ke juri.** `np.apply_along_axis(np.random.permutation, 0, X)` |
| ★★ Gaussian cocok-kovarians | fit PCA, tarik skor PC acak dengan SD per-PC yang cocok, transform balik | struktur orde-dua penuh | klaster | **Null M3C. Terkuat yang praktis.** ~10 baris |
| Parametrik satu-Gaussian (SigClust) | satu Gaussian dengan estimasi eigenvalue HDLSS | struktur orde-dua terkoreksi HDLSS | klaster | Uji hipotesis formal dengan p-value |
| ★ Feature-bootstrap / embedding null | lewatkan citra *ter-shuffle* (patch dipermutasi) ke encoder yang sama | geometri encoder itu sendiri | konten semantik | **Spesifik domain dan sangat meyakinkan untuk paper citra** — men-null-kan representasinya, bukan cuma angkanya |

**Pelaporan yang direkomendasikan:** untuk tiap kandidat k, laporkan indeks teramati bersama **distribusi null atas B=100–200 draw** dari minimal marginal-shuffle dan null cocok-kovarians, plus p-value empiris. Ini mengubah "silhouette kami 0,31" (tak bermakna sendirian) menjadi "silhouette kami 0,31 vs persentil-95 null 0,09, p < 0,01" — sebuah klaim ilmiah beneran.

## 6.8 Validasi tanpa label — praktik komunitas single-cell

Komunitas scRNA-seq punya masalah persis sama. Bioconductor OSCA "Clustering, redux":

| Tool scRNA-seq | Fungsi | Analog untuk citra kita |
|---|---|---|
| `bluster::approxSilhouette()` | Silhouette terhadap centroid, O(nk) | Di n=4179 kita mampu exact — pakai `silhouette_samples` dan **laporkan distribusi per-klaster, bukan cuma mean** |
| ★ `bluster::neighborPurity()` | **Proporsi tetangga suatu titik yang se-klaster** | **kNN purity di embedding.** ~5 baris. Intuitif, ramah-juri, dan bekerja untuk klaster non-konveks |
| `bluster::clusterRMSD()` | RMSD per klaster dari centroid; RMSD besar → klaster punya struktur internal | Cara membenarkan *tidak* memecah lebih jauh |
| `bluster::pairwiseModularity(as.ratio=TRUE)` | Rasio bobot edge teramati/ekspektasi antar pasangan klaster | **Matriks konfusi antar-klaster tanpa label** |
| `scran::bootstrapStability()` | Bootstrap → recluster → rasio ARI | = 6.6 |

### ★★ chooseR — protokol paling langsung bisa dipindahkan
Patterson-Cross, Levine & Menon (2021), *BMC Bioinformatics* 22:39 · `rbpatt2019/chooseR`. Resep persisnya:
1. **100 subset acak 80%** tanpa pengembalian (50 repeat di 50% memberi hasil sebanding — fallback kalau compute mepet)
2. Cluster tiap subset di tiap nilai parameter kandidat
3. Bangun **matriks frekuensi ko-clustering**: untuk tiap pasangan (i,j), (# run se-klaster) / (# run di mana keduanya tersampel) ← **penyebutnya penting**
4. Ubah jadi jarak: **D = 1 − frekuensi ko-clustering**
5. Hitung **silhouette memakai D sebagai metrik** (`silhouette_samples(D, labels, metric='precomputed')`) → agregasi jadi median per klaster
6. **Aturan pemilihan:** bootstrap (25.000 resample) CI 95% pada median silhouette tiap nilai parameter; **threshold = batas-bawah-CI tertinggi di semua nilai**; **pilih parameter yang memberi jumlah klaster terbanyak yang median silhouette per-klasternya masih melewati threshold itu**

Ini "aturan 1-SE" dalam semangat — parsimoni-dengan-toleransi — dan sepenuhnya reproducible di Python untuk n=4179 dalam jauh di bawah satu jam. **Kalau cuma satu protokol dari dokumen ini yang diimplementasi, jadikan ini.**

### Tool lain
- ★ **clustree** (Zappia & Oshlack 2018, *GigaScience* 7(7)) — memvisualkan **pohon partisi lintas resolusi**: node = klaster, edge = aliran sampel antar resolusi berdekatan, transparansi edge = "in-proportion". Over-clustering muncul sebagai **edge in-proportion rendah yang datang dari banyak parent**. R saja; port Python ~100 baris `networkx`/`graphviz` dan **figur yang sangat kuat untuk paper**
- **scclusteval** — bootstrap Jaccard per klaster lintas resolusi
- ★ **MultiK** (Liu et al. 2021, *Genome Biology* 22:232) — 100 subsample × 40 resolusi; kelompokkan run berdasarkan K teramati; matriks konsensus per K; **rPAC**; pilih K kandidat dari **convex hull** scatter (1−rPAC, frekuensi); lalu SigClust pada dendrogram klaster untuk melabeli pasangan sebagai "class" yang berbeda-signifikan vs "subclass" yang tidak. **Distinksi class/subclass ini cara indah melaporkan hierarki kepercayaan, dan memetakan sempurna ke taksonomi e-waste** (mis. "layar" adalah class yang percaya diri; "monitor LCD vs TV LCD" adalah subclass split yang tidak kami klaim)
- **scICE** (2025, *Nature Communications* 16) — Inconsistency Coefficient multi-label, cepat
- **sc-SHC** (*Nature Methods* 2023) — uji hipotesis di dalam hierarchical clustering

### ★★ Analog marker-gene untuk citra — bukti interpretabilitas terkuat
Marker gene bekerja karena tiap klaster dicirikan fitur yang *tinggi secara diferensial* di dalamnya. Tiga analog citra, makin meyakinkan:
1. **Analisis fitur diskriminatif:** untuk tiap dimensi embedding, hitung effect size per klaster (Cohen's d atau AUROC klaster-vs-sisanya). Jujur tapi tak interpretable sendirian.
2. **Panel prototipe/medoid + boundary:** tampilkan 9 medoid (paling sentral) dan 9 anggota **silhouette-minimum** tiap klaster. Medoid menunjukkan klaster itu *apa*; kasus batas menunjukkan di mana ia *gagal*. Juri suka ini.
3. ★★ **Probing teks zero-shot CLIP sebagai pseudo-marker.** Encode kosakata istilah e-waste ("printed circuit board", "lithium battery", "CRT monitor", "keyboard", "power cable", "smartphone", "hard drive", "fluorescent lamp", …) dengan text encoder; skor tiap centroid klaster terhadap tiap istilah; laporkan 3 istilah teratas dan **margin terhadap yang ke-4**. Ini memberi tiap klaster *label semantik tanpa anotasi ground-truth apa pun*, plus confidence numerik. Analog struktural persis dari marker-gene enrichment, dan bisa dipertahankan justru karena kosakata teksnya tidak pernah di-fit ke data. Biaya sepele.

## 6.9 Purity / entropy terhadap subset yang dianotasi

Label manual sampel acak berstrata (mis. 300–500 citra, ~10% data), lalu hitung **hanya di subset berlabel**:
- **Purity** = (1/N) Σ_c max_l |c ∩ l| — intuitif, tapi **naik monoton dengan k** → jangan bandingkan lintas k tanpa koreksi
- **Entropi klaster** H(L|C), **V-measure** — juga **tidak** dikoreksi kebetulan
- ★ **Pakai ARI dan AMI sebagai angka utama** — keduanya chance-corrected; Rand mentah dan NMI mentah tidak, dan NMI bias naik dengan k
- ⚠️ **Caveat kritis:** subset berlabel harus ditarik **sebelum** melihat klaster. Kalau menganotasi sambil browsing klaster, estimasi purity jadi sirkular dan juri tajam akan mengatakannya
- Laporkan purity **per-klaster** dengan CI, bukan cuma rata-rata global

## 6.10 Validasi out-of-sample

1. **Prediction strength** — split 50/50, cluster train, assign test ke centroid train, PS(k) = min proporsi ko-assignment. Threshold 0,8.
2. **Transfer classifier gaya Lange** (`reval`) — cluster train → fit classifier → prediksi test → cluster test terpisah → banding ARI → **normalisasi terhadap baseline label acak**.
3. **Protokol penuh:** ulang k-fold (10× 5-fold), laporkan per k: mean ARI out-of-sample, SD-nya, dan versi ternormalisasi-null. **Clustering yang tidak transfer out-of-sample bukan clustering, tapi partisi sampel spesifik kita** — dan kalimat itu bisa jadi tesis paper.

## 6.11 Analisis sensitivitas preprocessing (multiverse)

Framing: **multiverse analysis / specification-curve analysis** (Steegen et al.; Simonsohn, Simmons & Nelson).

| Faktor | Level |
|---|---|
| Backbone | 3–8 (kita punya 8) |
| Normalisasi | none · L2 · standardize · L2-lalu-standardize |
| Reduksi dimensi | none · PCA-50 · PCA-100 · PCA-whitened-50 · UMAP-10 |
| Metrik | Euclidean · cosine · correlation |
| Algoritma | k-means · GMM(diag) · Ward · Spectral(kNN) · HDBSCAN · Leiden |
| k / min_cluster_size | 2–30 |

**Laporkan:** (a) **specification curve** — urutkan semua spesifikasi berdasarkan k terpilih, plot faktor mana yang mendorong outlier; (b) **modal k** dan proporsi spesifikasi yang memilihnya; (c) **heatmap ARI antar partisi final** lintas spesifikasi — kalau partisi dari backbone berbeda sepakat di ARI > 0,6, itu bukti jauh lebih kuat dari nilai silhouette manapun. **Ini analisis dengan value-per-CPU-hour tertinggi yang tersedia, dan sangat sedikit kompetitor yang melakukannya.**

## 6.12 Isu dimensi tinggi

**Concentration of distance** — Beyer, Goldstein, Ramakrishnan & Shaft (ICDT 1999): saat d→∞, (D_max−D_min)/D_min → 0, semua titik jadi ekuidistan. Aggarwal, Hinneburg & Keim (ICDT 2001): **norm fraksional L_p (p<1) berkonsentrasi lebih sedikit** dari L₁ < L₂.
→ **Uji praktis yang bisa dijalankan dan dilaporkan:** relative contrast RC = (D_max−D_min)/D_mean untuk embedding 1000-D mentah, untuk cosine, dan untuk PCA-50. Menunjukkan RC membaik setelah preprocessing = bukti kuantitatif langsung bahwa pilihan pipeline kita perlu. ~10 baris numpy.

**Hubness** — Radovanović, Nanopoulos & Ivanović (2010), *JMLR* 11:2487. Saat d membesar, distribusi N_k(x) (seberapa sering x muncul di daftar kNN orang lain) jadi **sangat skewed kanan**: beberapa "hub" muncul di sangat banyak daftar tetangga, banyak "anti-hub" tidak muncul sama sekali. Hub adalah titik yang lebih dekat ke **mean global** — artefak geometris, bukan semantik. **Hubness merusak kNN graph, jadi merusak spectral clustering, mutual reachability HDBSCAN, Leiden/Louvain, dan metrik kNN-purity apa pun.** Untuk embedding citra d≈1000 ini efek nyata yang bisa diukur.
- Ukuran: *k-skewness* S_{N_k}, dan **Robin-Hood index** (lebih robust)
- Reduksi: **Mutual Proximity** (Schnitzer et al. 2012, *JMLR* 13:2871), **Local Scaling**, **NICDM**, **DisSimLocal**
- Python: `scikit-hubness` — `skhubness.analysis.Hubness`, `skhubness.reduction.{MutualProximity, LocalScaling, DisSimLocal}`. Di n=4179 semuanya instan. ⚠️ paket agak tidak terpelihara terhadap sklearn terbaru — pin versi, atau implementasi MP langsung (~15 baris dari matriks jarak)
- **Yang dilaporkan:** k-skewness sebelum dan sesudah L2-norm/PCA/MP, plus efek MP terhadap ARI/stabilitas final. **Kalau hubness tinggi dan MP memperbaiki stabilitas, itu kontribusi yang terlihat benar-benar baru untuk kompetisi mahasiswa.**

**Dimensi intrinsik** — `scikit-dimension` (skdim): `CorrInt, DANCo, ESS, FisherS, KNN, lPCA, MADA, MiND_ML, MLE, MOM, TLE, TwoNN`. Semua punya `.fit(X)` global dan `.fit_pw(X, n_neighbors=...)` pointwise/lokal.

| Estimator | Metode | Catatan |
|---|---|---|
| ★ **TwoNN** (Facco et al. 2017, *Sci Rep*) | Rasio μ = r₂/r₁ tetangga ke-1 dan ke-2; Pareto dengan eksponen d | **Tercepat dan paling robust terhadap kelengkungan/densitas tak homogen** — default yang direkomendasikan. Sudah kita pakai: ID ≈ 8,8–11,0 |
| MLE (Levina & Bickel 2004) | MLE proses Poisson pada jarak kNN | Klasik. **Bias turun untuk d besar**; koreksi averaging MacKay-Ghahramani standar |
| CorrInt (Grassberger & Procaccia 1983) | Dimensi korelasi C(r) ∝ r^d | Sensitif pilihan scaling region |
| DANCo (Ceruti et al. 2014) | Distribusi jarak NN + sudut vs model hipersfer | Paling akurat di d tinggi; **paling lambat** |
| FisherS | Fisher-separability | Cepat, bagus di d sangat tinggi |

**Cara memakainya di paper:** jalankan TwoNN + MLE + FisherS di embedding mentah. Kalau ID ≈ 10 sementara d = 1000, maka (a) PCA ke ~2–3× ID **berprinsip, bukan sembarang**; (b) itu menjelaskan *kenapa* konsentrasi jarak tidak sekatastrofik yang disugestikan d=1000 (konsentrasi mengikuti dimensi **intrinsik**, bukan ambient); (c) variasi ID lokal antar klaster itu sendiri diagnostik struktur klaster.

**Cluster tendency sebelum memilih k:** Hopkins statistic + dip test pada jarak berpasangan (`pyclustertend`, R `clusterability`) — buktikan struktur yang clusterable memang ada *sebelum* memilih k. Catat kelemahan Hopkins yang diketahui dan jangan bersandar padanya sendirian.

## 6.13 Checklist pelaporan

Van Mechelen, Boulesteix, Dangl, Dean, Hennig, Leisch, Steinley & Warrens, **"Benchmarking in cluster analysis: A white paper"** (arXiv 1809.10496):

- [ ] Nyatakan **cakupan** dan definisikan kriteria kualitas **a priori**; akui kriteria berbeda bisa berarti optimum berbeda
- [ ] Nyatakan apa arti "klaster" **untuk aplikasi ini** sebelum memilih metode (Hennig: definisi klaster adalah *keputusan user*, bukan fakta matematis)
- [ ] Laporkan **semua** parameter tuning, inisialisasi, kriteria konvergensi, dan **random seed**; rilis kode yang bisa dijalankan
- [ ] Beri metode pembanding **anggaran komputasi, informasi a priori, dan usaha implementasi yang setara**
- [ ] Kombinasikan evaluasi **sintetis + nyata**
- [ ] Laporkan **distribusi (boxplot), bukan hanya mean**; laporkan **effect size**, bukan hanya signifikansi
- [ ] Periksa apakah peringkat metode **berubah lintas karakteristik data dan lintas kriteria**; identifikasi metode **Pareto-optimal**, bukan "pemenang"
- [ ] **Sertakan hasil negatif / foil** — kasus di mana metode kita kalah
- [ ] Nyatakan limitasi eksplisit; hindari klaim pemenang universal

Tambahan khusus unsupervised: laporkan **ukuran klaster** dan **fraksi noise/unassigned**; nyatakan apakah hitungan klaster termasuk noise HDBSCAN; laporkan **stabilitas tiap klaster secara individual**; tampilkan **kasus gagal** (anggota silhouette terendah) bersama prototipe.

---

# TAHAP 7 — Interpretasi & penamaan klaster

## 7.1 Sub-ruang konsep dari teks (mekanisme inti empat lensa)

Untuk kriteria *c* dengan himpunan prompt P_c = {p₁…p_m} yang di-encode text tower SigLIP2 jadi T_c ∈ R^(m×d), proyeksikan tiap embedding citra z:
- **similarity mentah** s = softmax(z·T_cᵀ/τ) ← ini persis "soft text counterpart" TAC (ICML 2024)
- **kode sparse nonneg** (SpLiCE)
- **residual** setelah membuang sub-ruang kriteria lain ← trik ortogonalitas/redundansi dari literatur multiple clustering

Lalu cluster **di dalam ruang terproyeksi itu**. Sifatnya: (a) sepenuhnya frozen, (b) O(detik) per kriteria untuk 4.179 citra, (c) sepele diulang untuk 4 kriteria, (d) **belum dipublikasikan dalam bentuk ini untuk multiple-lens clustering** → celah kita.

## 7.2 Metode penamaan & interpretasi

| Metode | Tahun/Venue | Mekanisme | Repo | Frozen? |
|---|---|---|---|---|
| ★ **IC\|TC** | 2024 ICLR | VLM tulis deskripsi terkondisi-kriteria per citra → LLM usulkan label → LLM agregasi jadi K nama → LLM assign. **Nol training.** Mendukung *negative prompt* ("jangan pertimbangkan gender") | `sehyunkwon/ICTC` | ✅ (API) |
| **TGIC** | 2024 EACL | Caption + **VQA berprompt pengetahuan domain** → cluster embedding *teks*. Temuan kunci: **representasi teks sering mengalahkan fitur citra** di 8 dataset | `AndSt/text_guided_cl` | ✅ |
| ★★ **TGAICC** | 2024 arXiv 2406.18589 | (1) GPT-4 parafrase kriteria jadi beberapa prompt; (2) LLaVA-NeXT jawab tiap prompt per citra; (3) embed jawaban + k-means tiap prompt; (4) **kelompokkan partisi hierarkis berdasarkan AMI, lalu konsensus dalam tiap grup** dengan MCLA/HBGF/CSPA/NMF | `AndSt/alternative_image_clustering` | ✅ |
| ★ **ITGC** | 2025 arXiv 2506.12514 | Loop tertutup: LLM usulkan **konsep** dari query user → VLM skor citra terhadap konsep → k-means → **silhouette dipakai sebagai sinyal *unsupervised* untuk mengarahkan generasi konsep ronde berikutnya** | — | ✅ |
| ★ **X-Cluster / OpenSMC** | 2024–25 | **Kriterianya sendiri ditemukan, bukan diberikan.** Criteria Proposer + Semantic Grouper (nama hierarkis kasar/menengah/halus). Benchmark baru COCO-4c dan Food-4c. Metrik **TPR** (cakupan penemuan kriteria), **SAcc** (akurasi semantik nama klaster via Sentence-BERT), **CAcc** | OpenReview PhRYDGqiee | ✅ |
| **OAK** | 2025 CVPR | **Context token** yang dipelajari disuntikkan ke ViT CLIP **frozen** → encoder yang sama menghasilkan fitur spesifik-konteks. Metrik **"Omni accuracy"** = konsistensi lintas konteks | `Wayne2Wang/OAK` | ✅ (hanya context token dilatih) |
| **CaSED** | 2023 NeurIPS | Vocabulary-free: retrieve caption dari DB besar, ekstrak noun sebagai kosakata spesifik-citra, skor dengan CLIP | `altndrr/vic` | ✅ training-free |
| ★ **DN-CBM** | 2024 ECCV | Sparse autoencoder di fitur CLIP frozen → kamus konsep, **lalu namai tiap unit SAE lewat embedding teks terdekat atas 20k kata Inggris tersering** | `neuroexplicit-saar/Discover-then-Name` | ✅ |
| ★ **SpLiCE** | 2024 NeurIPS | Dekomposisi embedding CLIP jadi kombinasi linear **sparse nonnegatif** atas kamus **15.000 konsep teks**; mean-centering menangani modality gap; 5–20 konsep aktif per citra | `AI4LIFE-GROUP/SpLiCE` | ✅ post-hoc |
| **TextSpan** | 2024 ICLR Oral | Dekomposisi representasi CLIP per attention-head per token; **temukan basis teks yang merentang output tiap head**, mengungkap peran head ("head warna", "head lokasi") | `yossigandelsman/clip_text_span` | ✅ |
| **Text-to-Concept** | 2023 ICML | Pelajari **peta linear** yang menyejajarkan ruang fitur model vision apa pun ke ruang teks CLIP → *model apa pun* dapat sumbu konsep berbasis teks | `k1rezaei/Text-to-concept` | ✅ |
| **LF-CBM** | 2023 ICLR | GPT-3 usulkan konsep per kelas; CLIP-Dissect memberi aktivasi konsep; layer akhir sparse — **tanpa label konsep** | `Trustworthy-ML-Lab/Label-free-CBM` | ✅ |
| **LaBo** | 2023 CVPR | GPT-3 generate kalimat konsep per kelas, seleksi submodular, CLIP skor | `YueYANG1996/LaBo` | ✅ |
| **TCAV** | 2018 ICML | Konsep = arah dari classifier linear yang memisahkan aktivasi contoh-konsep dari acak | `tensorflow/tcav` | ✅ post-hoc |
| **k-LLMmeans** | 2025 | k-means di mana tiap centroid diganti **ringkasan LLM** anggotanya | — | ✅ |
| **ClusterLLM** | 2023 EMNLP | Feedback LLM: triplet ("A lebih mirip B atau C?") menetapkan *perspektif*; pairwise hierarkis menetapkan *granularitas* | `zhang-yu-wei/ClusterLLM` | ❌ (embedder di-fine-tune) |
| **GoalEx** | 2023 EMNLP | Usulkan kandidat *penjelasan* natural-language, assign teks ke sana, ILP pilih subset — **klaster didefinisikan oleh deskripsinya** | `ZihanWangKi/GoalEx` | ✅ |
| **TnT-LLM** | 2024 KDD | LLM generate **taksonomi label** end-to-end, lalu pseudo-label skala besar | — (MSFT) | ✅ |

## 7.3 Metrik untuk membuktikan komplementaritas antar-lensa

| Metrik | Chance-corrected? | Metrik sejati? | Kegunaan |
|---|---|---|---|
| NMI | ❌ | ❌ | Metrik pelaporan de-facto di *semua* paper multiple clustering |
| ★ **AMI** | ✅ | ❌ | **Pakai untuk perbandingan antar-partisi ketika K berbeda antar lensa** — NMI bias naik saat K besar. Ini yang dipakai TGAICC. Vinh, Epps & Bailey, JMLR 17 (2016) |
| RI / ARI | RI ❌, ARI ✅ | ❌ | RI metrik kedua di keluarga benchmark Multi-MaP; ARI standar modern |
| ★ **Variation of Information (VI)** | ❌ | ✅ **metrik sejati** | Yang benar kalau ingin *jarak* antar lensa (memenuhi ketaksamaan segitiga). Meilă 2007. VI = H(X\|Y)+H(Y\|X), sepele dari tabel kontingensi |
| ★ **Rata-rata NMI berpasangan antar clustering (D_R)** | ❌ | ❌ | **Ukuran diversitas/redundansi eksplisit DivClust** — laporkan matriks K×K NMI/AMI berpasangan lintas lensa. Off-diagonal rendah = lensa komplementer |
| ACC (Hungarian matching) | — | — | Standar kalau ada label kriteria ground-truth |
| SAcc (Sentence-BERT) | — | — | Untuk menilai *nama* klaster tanpa alignment indeks |
| TPR penemuan kriteria | — | — | Untuk lensa data-driven: apakah kita memulihkan kriteria yang akan disebut manusia? |
| Omni accuracy | — | — | Akurasi gabungan lintas semua lensa sekaligus (OAK) |
| Silhouette | — | — | Seleksi label-free untuk himpunan konsep / K per lensa (ITGC) |

★ **Figur motivasi yang wajib ada:** laporkan matriks AMI K×K yang sama untuk **baseline naif** (k-means di SigLIP2 mentah, empat kali dengan seed berbeda) — hasilnya akan menunjukkan partisi yang nyaris identik. Itu kontras yang membuktikan lensa kita benar-benar berbeda.

## 7.4 Evaluasi manusia

**Desain tugas** (pinjam dari evaluasi topic model — Chang, Boyd-Graber, Gerrish, Wang & Blei, **"Reading Tea Leaves", NIPS 2009**. Temuan kunci mereka — bahwa metrik otomatis berbasis likelihood sering *berkorelasi negatif* dengan interpretabilitas manusia — adalah sitasi sempurna untuk menjelaskan kenapa kita menjalankan studi manusia sama sekali):

1. ★ **Intruder detection (tugas tunggal terbaik).** Tampilkan 5 citra: 4 dari klaster C, 1 dari klaster lain. Tanya: mana yang tidak termasuk? **Akurasi di atas kebetulan (20%) mengukur koherensi klaster** tanpa ground truth. Port langsung dari word-intrusion Chang et al. Laporkan akurasi per-klaster.
2. **Pairwise same/different.** Sampel pasangan intra- dan antar-klaster; tanya "jenis e-waste yang sama?"
3. **Skala rating.** "Seberapa koheren himpunan citra ini?" Likert 1–5. Paling lemah — bias anchoring dan central-tendency.
4. **Free labeling lalu agreement.** Rater independen menamai tiap klaster; ukur apakah nama-nama konvergen.

**Statistik kesepakatan:**
- **Cohen's κ** — tepat 2 rater, nominal (`sklearn.metrics.cohen_kappa_score`)
- **Fleiss' κ** — jumlah rater tetap per item (`statsmodels.stats.inter_rater.fleiss_kappa`)
- ★ **Krippendorff's α** — **jumlah rater bebas, boleh ada data hilang, level pengukuran apa pun.** Default yang benar untuk tim kecil. `pip install krippendorff`
- **Konvensi pelaporan (Krippendorff): α ≥ 0,800 = reliabel; 0,667 ≤ α < 0,800 = kesimpulan tentatif saja; α < 0,667 = jangan menarik kesimpulan.** Landis & Koch untuk κ: 0,41–0,60 moderate, 0,61–0,80 substantial, >0,80 almost perfect

**Berapa sampel per klaster yang bisa dipertahankan:**
- **≥ 3 rater independen** (2 memberi κ tapi tidak bisa memecah seri)
- **≥ 20–30 item per klaster** untuk tugas intruder. Di k=8 itu 160–240 penilaian per rater — 30–60 menit kerja, sangat feasible untuk satu tim
- ★ **Sampel item berstrata berdasarkan silhouette** (atau jarak ke medoid): mis. 10 tinggi / 10 median / 10 rendah. **Sampling acak saja menyembunyikan persis mode kegagalan yang akan ditanya juri.** Laporkan akurasi terpisah per strata
- **Buta dan acak:** rater tidak boleh tahu identitas klaster, urutan item harus diacak. Nyatakan ini
- Laporkan **CI 95% pada akurasi per-klaster** (interval Wilson, `statsmodels.stats.proportion.proportion_confint`) — dengan n=30 CI-nya lebar, dan mengatakannya adalah kekuatan, bukan kelemahan

## 7.5 Visualisasi

| Bentuk | Kegunaan |
|---|---|
| UMAP/PaCMAP 2-D diwarnai klaster | **Hanya untuk figur**, beri label bahwa itu hanya untuk figur |
| Montase eksemplar per klaster (medoid + boundary) | Bukti kualitatif utama |
| clustree-style partition tree lintas resolusi | Menunjukkan hierarki dan over-clustering |
| Sankey/alluvial antar lensa | Menunjukkan bagaimana partisi lensa A pecah di lensa B |
| Heatmap AMI/VI antar lensa | Bukti komplementaritas |
| Specification curve | Sensitivitas preprocessing |
| Heatmap konsensus + kurva PAC | Stabilitas |
| Radar/parallel-coordinates skor sumbu konsep per klaster | Interpretasi lensa |
| Treemap ukuran klaster × nilai recovery | "So what" ekonominya |

---

# TAHAP 8 — Grounding domain e-waste

## 8.1 Taksonomi resmi (supaya nama klaster punya sitasi, bukan karangan)

### WEEE Directive 2012/19/EU Annex III — 6 kategori koleksi (berlaku sejak 15 Ags 2018)
| # | Nama resmi | Kriteria | Contoh |
|---|---|---|---|
| 1 | Temperature exchange equipment | Sirkuit internal dengan zat selain air untuk pendingin/pemanas | Kulkas, freezer, AC, dehumidifier, heat pump |
| 2 | Screens, monitors, equipment with screens > 100 cm² | Luas layar > 100 cm² | TV (CRT + flat), monitor, **laptop**, tablet, e-reader |
| 3 | Lamps | Sumber cahaya listrik dengan base standar | TL, CFL, HID, natrium tekanan rendah, lampu LED |
| 4 | Large equipment (dimensi luar > 50 cm) | Bukan kat. 1–3, ≥1 dimensi > 50 cm | Mesin cuci, pengering, dishwasher, kompor, mesin cetak besar, **panel surya**, vending machine |
| 5 | Small equipment (tidak ada dimensi > 50 cm) | Bukan kat. 1–4/6 | Vacuum cleaner, microwave, setrika, toaster, jam, timbangan, **kamera digital**, mainan kecil, power tools |
| 6 | Small IT & telecom (tidak ada dimensi > 50 cm) | Fungsi IT/telekom | **Ponsel**, GPS, kalkulator, router, PC, **printer**, telepon, keyboard, mouse |

**Tiga kejanggalan yang layak diangkat di paper:**
- **Laptop masuk Kategori 2** (layar), bukan 6 — ambang layar mendominasi
- **PC desktop, printer, router masuk Kategori 6**; **server dan mesin fotokopi profesional masuk Kategori 4**
- ★ **Aturan 50 cm adalah kriteria murni geometris** — clustering berbasis citra **tidak bisa memulihkannya tanpa informasi skala**. Ini **limitasi sah dan bisa dikutip** untuk dinyatakan di paper

### WEEE Annex I — 10 kategori lama (masih banyak dipakai di literatur & statistik nasional)
1. Large household appliances · 2. Small household appliances · 3. IT & telecom · 4. Consumer equipment & PV panels · 5. Lighting · 6. Electrical/electronic tools · 7. Toys, leisure, sports · 8. Medical devices · 9. Monitoring & control · 10. Automatic dispensers

### ★★ UNU-KEYS — klasifikasi 54-key UNITAR/UNU
**Taksonomi paling kuat groundingnya untuk paper kita**, karena (a) standar statistik internasional, (b) memetakan eksplisit ke sistem 6-kategori dan 10-kategori, (c) **sudah ada dataset citra publik yang dibangun mengelilinginya** (Roboflow E-Waste Dataset, 77 kelas).

Definisi: tiap UNU-KEY mengelompokkan produk dengan *fungsi mirip, komposisi material sebanding, bobot rata-rata homogen, dan distribusi umur pakai sebanding*. Sumber: Forti, Baldé & Kuehr, *E-waste Statistics: Guidelines on Classifications, Reporting and Indicators*, ed. ke-2, UNU ViE-SCYCLE, Bonn. **Edisi ke-3 (2026) diumumkan di scycle.info — cek sebelum submit.**

Key yang paling relevan dengan isi klaster kita:
`0102` Dishwashers · `0104` Washing Machines · `0108` Fridges · `0111` Air Conditioners · `0114` Microwaves · `0201` Other Small Household · `0204` Vacuum Cleaners · `0301` Small IT (router, mouse, keyboard, external drive) · `0302` Desktop PCs · `0303` Laptops (incl. tablets) · `0304` Printers · `0306` Mobile Phones · `0308` CRT Monitors · `0309` Flat Display Panel Monitors · `0403` Music Instruments, Radio, HiFi · `0405` Speakers · `0406` Cameras · `0407` CRT TVs · `0408` Flat Panel Display TVs · `0502` CFL · `0503` Straight tube fluorescent · `0505` LED · `0702` Game Consoles

⚠️ Key `0107` dan `0110` tidak ada; penomorannya tidak kontigu. Total tetap 54.

### Basel Convention
**Rezim lama:** A1180 (Annex VIII, berbahaya) · B1110 (Annex IX, dianggap tidak berbahaya) · Y46 (limbah rumah tangga) · Y31 Pb · Y29 Hg · Y26 Cd · Y45 organohalogen.

★ **Amandemen E-waste 2025** (BC-15/18, berlaku **1 Januari 2025**):
- **A1181 BARU** (Annex VIII) — e-waste *berbahaya*, komponennya, dan limbah dari pemrosesan e-waste. Eksplisit mencakup kaca CRT, baterai, saklar/lampu/TL merkuri, kapasitor PCB, komponen asbes, PCB tertentu, perangkat display, dan komponen plastik ber-BFR
- **Y49 BARU** (Annex II) — **semua e-waste lainnya**
- **B1110 dan B4030 DIHAPUS** dari Annex IX
- **Efek bersih: praktis semua e-waste, berbahaya atau tidak, kini tunduk prosedur Prior Informed Consent.** Kalimat framing yang sangat kuat dan sangat kini untuk pendahuluan paper 2026
- ⚠️ Status A1180 pasca-2025 dijelaskan tidak konsisten di sumber sekunder — verifikasi langsung ke keputusan BC-15/18

### ★ INDONESIA — posisi regulatif limbah elektronik
| Regulasi | Relevansi |
|---|---|
| UU No. 18/2008 | Undang-undang dasar pengelolaan sampah; definisi *sampah spesifik* |
| **PP No. 22/2021, Lampiran IX (Daftar Limbah B3)** | Daftar yang memuat kode e-waste |
| PP No. 27/2020 tentang Pengelolaan Sampah Spesifik | E-waste rumah tangga sebagai *sampah spesifik* |
| PermenLHK No. 6/2021 | Tata cara & persyaratan pengelolaan limbah B3 |
| **PermenLHK No. 9/2024** | Instrumen terbaru; sampah mengandung B3 domestik/komersial termasuk baterai bekas, lampu neon |
| Keppres No. 61/1993 | Ratifikasi Konvensi Basel |

**Kode limbah persis dari PP 22/2021 Lampiran IX:**
- ★ **B107d** — *"Limbah elektronik termasuk cathode ray tube (CRT), lampu TL, printed circuit board (PCB), dan kawat logam"* — **Kategori bahaya 2**. **Ini definisi hukum e-waste versi Indonesia, dan perhatikan: ia sendiri sudah berupa taksonomi fraksi material (CRT / lampu / PCB / kawat logam).** Sangat kuat untuk membenarkan lensa material kita
- **A111d** — refrigerant bekas dari peralatan elektronik — Kategori bahaya 1
- Sumber spesifik (Kegiatan 28 "Perakitan komponen/peralatan elektronik", Kegiatan 29 "Rekondisi/Remanufacturing"): **A328-1** mercury contactor/switch · **A328-2** lampu fluoresen (Hg) · **A328-3** larutan printed circuit · **A328-4** caustic stripping/photoresist · **A328-5** sludge produksi · **B328-1** CRT · **B328-4** PCB · **B328-5** limbah kabel logam & insulasinya · **A329-1/-2/-4/-5** analog di Kegiatan 29

**Statistik Indonesia:**
- KLHK 2021 (34 provinsi, 3.227 perusahaan): **33.683,781 ton** e-waste industri terdaftar; Jabar 31.360,29 t, Jatim 768,47 t, DIY 669,87 t. Ekspor 2020–2021: 343.771,65 t (Kanada, Jepang, Korsel, Singapura). ⚠️ Hanya aliran formal terdaftar — ~2 orde di bawah estimasi termasuk rumah tangga
- ★ **Mairizal, Sembada, Tse & Rhamdhani (2021)**, *Journal of Cleaner Production* **293**, 126096 — **~2 juta ton di 2021 (terbesar di Asia Tenggara)**, naik ke **3,2 Mt/tahun dan 10 kg/kapita di 2040**, bernilai **US$14 miliar di 2040**; **~56% e-waste Indonesia dihasilkan di Jawa**
- ★ **Widyarsana & Nurdiani (2024)**, *E3S Web of Conferences* **485**, 05006 — proyeksi **4.204.545 t total di 2040**; **TV mendominasi kedua sektor**. **Komposisi material: 1% berbahaya, 31% logam ferrous, 27% kaca + plastik, 6% non-ferrous, 8% lainnya** ← breakdown fraksi material spesifik-Indonesia, sangat citable
- ★ **DKI Jakarta e-waste service** (ewaste.dinaslhdki.id) memilah operasional lewat **ambang massa: >5 kg = penjemputan; <5 kg = antar sendiri ke dropbox**. Analog Indonesia dari pemisahan 50 cm WEEE, dan biner lokal yang bagus untuk level klaster kasar
- ⚠️ **Tidak ditemukan SNI yang secara spesifik mengklasifikasi jenis produk e-waste.** Kalau mau mengklaim "tidak ada SNI", cek langsung SISPK BSN dulu

### Taksonomi fraksi material (sumbu ortogonal kedua)
Ferrous · Aluminium · Tembaga (dan kabel) · Non-ferrous lain · **PCB** · Plastik (ABS, HIPS, PC/ABS, PP; ber-BFR vs bebas-BFR) · **Kaca CRT** (funnel = timbal, panel = barium/strontium) · Panel LCD/LED + lampu backlight · **Baterai** (Li-ion, NiCd, NiMH, lead-acid) · Refrigerant & oli kompresor · Busa PUR · Toner/tinta · Kayu/lainnya

Komposisi berat WEEE campuran agregat: besi & baja ≈ 50%, plastik ≈ 21%, non-ferrous termasuk logam mulia ≈ 13%, kaca ≈ 5%.

### Kategori bahaya — barang apa membawa bahaya apa
| Zat | Ada di | Efek |
|---|---|---|
| **Timbal (Pb)** | Solder, **kaca funnel CRT**, aki, sebagian formulasi PVC | Gangguan kognitif/perilaku |
| **Merkuri (Hg)** | **TL & CFL**, **backlight CCFL di LCD lama**, saklar tilt, sebagian relay | Gangguan sensorik, kehilangan memori |
| **Kadmium (Cd)** | Resistor peka cahaya, **baterai NiCd**, alloy tahan korosi, fosfor CRT lama | Kerusakan paru; karsinogen |
| **Kromium heksavalen Cr(VI)** | Pelapis logam anti-korosi (sekrup, sasis) | Karsinogen |
| **BFR/PBDE/TBBPA** | Casing plastik **kebanyakan elektronik**, terutama **casing CRT/monitor dan PCB** | Neurologis, disrupsi tiroid; prekursor dioksin saat dibakar |
| **Berilium oksida** | Material antarmuka termal, sebagian konektor | Risiko kanker paru |
| ★ **Sel Li-ion** | Ponsel, laptop, tablet, power bank, sepeda listrik, vape | **Risiko kebakaran thermal-runaway saat pengumpulan & shredding — bahaya operasional #1 di pabrik daur ulang saat ini** |
| **Refrigeran CFC/HCFC/HFC** | **Peralatan temperature-exchange Kategori 1** | Perusakan ozon / GWP |

**RoHS 2011/65/EU membatasi 10 zat:** Pb, Hg, Cd, Cr(VI), PBB, PBDE, + 4 ftalat (DEHP, BBP, DBP, DIBP).

★ **"Hazard tier" yang diusulkan untuk interpretasi klaster** (bisa dipertahankan, memetakan ke A1181): *Tier A (depolusi wajib)* = perangkat CRT, lampu, baterai, peralatan pendingin; *Tier B (ber-BFR/PCB)* = PCB, IT/telekom, layar; *Tier C (bahaya rendah, massa tinggi)* = white goods besar, kabel.

### Global E-waste Monitor 2024 — angka untuk pendahuluan
Baldé, Kuehr, Yamamoto, McDonald, D'Angelo, Althaf, et al. (2024), ITU & UNITAR, Geneva/Bonn.
- **62 juta ton** dihasilkan 2022 (**+82% vs 34 Mt di 2010**); tumbuh 2,6 Mt/tahun
- Proyeksi **82 Mt di 2030** (+33%)
- **Hanya 22,3% terdokumentasi terkumpul & didaur ulang formal** (14 Mt); turun ke **20% di 2030** — generasi naik **lima kali lebih cepat** dari daur ulang terdokumentasi
- Logam di e-waste: **31 miliar kg total**, **19 miliar kg layak dipulihkan**, bernilai **USD 91 miliar** (Cu $19b, Au $15b, Fe $16b); hanya **USD 28 miliar** dipulihkan; **USD 62 miliar** hilang
- **<1%** permintaan rare-earth dipenuhi daur ulang e-waste
- Per kapita: Eropa 17,6 kg, Oseania 16,1 kg, Amerika 14,1 kg; tingkat daur ulang Eropa 42,8%, Afrika <1%

★ **Breakdown per kategori 2022** (tabel kunci untuk membenarkan taksonomi klaster):
| Kategori | Dihasilkan (Mt) | Tingkat daur ulang terdokumentasi |
|---|---|---|
| Small equipment | **20,4** | 12% |
| Large equipment (tanpa PV) | **15,1** | 34% |
| Temperature exchange | **13,3** | 27% |
| Screens & monitors | **5,9** | 25% |
| Small IT & telecom | **4,9** | 22% |
| Lamps | **1,9** | 5% |
| PV panels | **0,6** | 17% |

⚠️ Cross-check ke PDF laporan; angka ini via ringkasan pihak ketiga.

## 8.2 Dataset citra e-waste publik (untuk augmentasi, harus dicantumkan sumbernya)

### Spesifik e-waste
| Dataset | Ukuran | Kelas | Lisensi | URL |
|---|---|---|---|---|
| ★★ **Roboflow "E-Waste Dataset"** | **19.613 citra beranotasi** | **77 kelas** | **CC BY 4.0** | universe.roboflow.com/electronic-waste-detection/e-waste-dataset-r0ojc |
| | **Disusun eksplisit mengelilingi 54 UNU-KEY.** Bbox + polygon. Anonim (tanpa wajah). Dari dataset Roboflow terbuka + Wikimedia Commons. **Padanan terbaik untuk kebutuhan kita** | | | |
| ★ **Roboflow "Balanced E-Waste Dataset"** | **7.216 citra**, ~200/kelas | **37 kelas** | CC BY 4.0 | universe.roboflow.com/electronic-waste-detection/balanced-e-waste-dataset |
| **E-Waste Image Dataset** (Tamrakar) | ~12,4 MB | **10**: PCB, Player, Battery, Microwave, Mobile, Mouse, Printer, Television, Washing Machine, Keyboard | **Apache 2.0** | kaggle.com/datasets/akshat103/e-waste-image-dataset |
| **E-Waste Vision 18** (Gore) | 1,75 GB | **18** (campur level perangkat & komponen) | ⚠️ **"Other" — TIDAK ADA lisensi diberikan; uploader menolak kepemilikan** | kaggle.com/datasets/harshadsgore/... |
| | ⚠️ **Berisiko hukum untuk submission kompetisi.** Daftar kelasnya tetap berguna sebagai referensi taksonomi | | | |
| **Hardware-ID** (Rahman) | **3.600+ citra**, 975 MB | Komponen internal laptop: optical drive, touchpad, speaker, heat sink/fan | **MIT** | kaggle.com/datasets/shafin808s/hardware-id-an-annonated-e-waste-dataset |
| **E-Waste Vision Dataset** (EWasteNet) | **1.053 citra** | **8**: Mobile, TV, Laptop, Keyboard, Mouse, Microwave, Smartwatch, Camera | cek repo | github.com/NifulIslam/EWasteNet-... |
| ★ **RecyBat24** | **2.835** non-augmented | **3**: Pouch (1.740), Prismatic (660), Cylindrical (435) | CC (Scientific Data) | zenodo.org/records/13126689 |
| | Baterai Li-ion di e-waste, difoto di pusat daur ulang nyata, 640×640, 3 kondisi pencahayaan | | | |
| **E-Waste Hub** (axeldavid) | 630 MB | — | ⚠️ **CC BY-NC 4.0** | kaggle.com/datasets/axeldavid/e-waste-dataset-e-waste-hub |
| **E Waste Compressed** (mxtuhin) | 62 MB | — | ★ **CC0** (paling aman) | kaggle.com/datasets/mxtuhin/e-waste-compressed |
| **Dismantled Electronic Components 1–5 mm / 5–10 mm** | 197/276 MB | Komponen dari PCB bekas, **berdasarkan ukuran fisik** | ⚠️ Unknown | kaggle.com/datasets/nayarraunaq/... |

### Waste umum yang memuat kelas e-waste
| Dataset | Ukuran | Catatan |
|---|---|---|
| ★ **TrashBox** | **17.785** | 7 top-level termasuk **e-waste 2.883**. **Subkelas e-waste: electronic chips; laptops & smartphones; appliances; electric wires, cords and cables** — taksonomi 4-arah yang sangat mirip lensa material kita. github.com/nikhilvenkatkumsetty/TrashBox |
| **TrashNet** | 2.527 | 6 kelas, **MIT** |
| **TACO** | 1.500 | 28 kelas (60 sub), **MIT** |
| **ZeroWaste** | 4.503 + 6.212 unlabeled | ⚠️ CC BY-NC 4.0 |
| **MJU-Waste v1.0** | 2.475 | MIT, segmentasi |
| **RealWaste** | 4.752 | 9 kelas, UCI |
| **The Garbage Dataset (GD)** | **12.259** | 10 kelas termasuk **battery**, **CC BY 4.0**, arXiv 2602.10500 |
| ★ **Waste Classification Dataset** (fadlicr7 — penulis Indonesia) | 1,35 GB | **3 kelas: Recyclable / Electronic / Organic** — ★ **skema kelas persis sama dengan dataset penyisihan kita**. **CC0** |
| **New Trash Classification Dataset** (Damar Galih — penulis Indonesia) | 410 MB | 8 kelas seimbang, CC BY 4.0 |

**Agregator utama (kutip di related work):** github.com/AgaMiko/waste-datasets-review — 37+ dataset citra sampah dengan ukuran, jumlah kelas, tipe anotasi, dan lisensi.

### PCB / komponen elektronik
| Dataset | Ukuran | Lisensi |
|---|---|---|
| **PCB-DSLR** | **748 citra** @4928×3280 dari **165 PCB bekas**, **9.313 IC berlabel** | akademik |
| **FICS-PCB** | ~400 citra, 31 sampel PCB, ~30.000 bbox SMD | trust-hub |
| **FPIC** | **261 citra**, 93 board, **>71.000 instance beranotasi** | **CC BY 4.0** |
| **Mahalingam** | 984 citra, 123 board, 12.000+ komponen | cek |
| **DeepPCB** | 1.500 pasang citra, 6 tipe defect | repo |

### Spektral
- ★ **TECNALIA WEEE Hyperspectral Dataset** — 13 citra hiperspektral 1024×1024, **76 band 415–1008 nm**, kelas: background, Copper, Brass, Aluminium, Stainless Steel, White Copper. **Non-komersial.** zenodo.org/records/12565131. **Temuan negatif yang berguna dikutip: adaptasi foundation model DINOv2 tidak memberi perbaikan** (U-Net 76 band terbaik, mIoU 0,73)
- **SpectralWaste** — 852 citra, 6 kelas, CC BY 4.0, zenodo.org/records/11499414

### Non-citra (untuk sudut estimasi nilai)
- ★ **Precious Metal Content in E-Waste** — kaggle.com/datasets/abhaynb/precious-metal-content-in-e-waste, CC BY-SA 4.0. **Join ke label klaster kita untuk menghitung estimasi nilai recovery per klaster**
- E-waste Statistics by Country 2015–2024 (ODbL)

## 8.3 Penelitian terkait

### ★★ Unsupervised/clustering pada citra sampah — literatur tipis = celah kita
1. ★★★ **Huang, Song, Ba, An, Liang, Deng, Liu, Zhang, Zhou (2025).** *"Unsupervised Waste Classification By Dual-Encoder Contrastive Learning and Multi-Clustering Voting (DECMCV)."* arXiv:2503.02241. Encoder ConvNeXt pretrained + ViT untuk generasi sampel positif + **multi-clustering voting**. TrashNet **93,78%**, Huawei Cloud **98,29%**, dan **set dunia-nyata 4.169 citra berlabel hanya 50 sampel**, mengalahkan baseline supervised **+29,85%**. **Sitasi metodologis terpenting kita** — skala dataset nyaris identik (4.169 vs 4.179) dan framing discovery-unsupervised persis sama
2. Caron et al. (2018) DeepCluster, ECCV — referensi kanonik
3. Majchrowska et al. (2022), *Waste Management* 138:274 — skema dua tahap detect-then-classify, menggabungkan 6+ dataset litter publik. **Preseden untuk menggabungkan sumber publik heterogen**
4. ★ Koszarski/Majchrowska et al. (2023), *Engineering Applications of AI* 128:107542 — **struktur label hierarkis** dengan segmentasi weakly-supervised. Mendukung taksonomi klaster multi-level

### Klasifikasi citra e-waste (supervised)
5. ★ **Islam, Jony, Hasan, Sutradhar, Rahman, Islam (2023).** *"EWasteNet: A Two-Stream Data Efficient Image Transformer Approach for E-Waste Classification."* **IEEE ICSECS 2023**, arXiv:2311.12823. Dua stream DeiT: (a) edge Sobel, (b) piramida ASPP + CBAM. **96% akurasi**, <1M parameter
6. ★ **Rajeev, Dharewa, Lakshmi et al. (2025).** *Scientific Reports* **15**, 18151. YOLOv5/v7/v8, YOLOv8 terbaik, F1 puncak 0,63
7. ★ **Oise & Konyeha (2025).** *Discover Artificial Intelligence* **5**(1), 210. Kaggle E-Waste 3.859 citra 12 kelas, **skema hierarkis 4 parent × 3 child** — ★ **preseden langsung untuk taksonomi klaster dua level**. Hybrid EfficientNet+MobileNet → 98%
8. ★ **Nowakowski & Pamuła (2020).** *Waste Management* 109:1–9. Faster R-CNN untuk mengenali **tipe DAN kategori ukuran** e-waste dari foto warga, lalu masuk ke vehicle routing. **Framing terapan yang bagus sekali untuk paper BDC**
9. Kumsetty, Bhat Nekkare, Kamath, Kumar (2022). TrashBox, IEEE FRUCT 31
10. *"Artificial intelligence based classification for waste management: A survey based on taxonomy, classification & future direction."* *Computer Science Review* (2024) — ★ survei untuk paragraf related work

### PCB & estimasi nilai recovery ★★
11. ★★ **Fernandes et al. (2021).** *"Estimating Recycling Return of Integrated Circuits Using Computer Vision on Printed Circuit Boards."* *Applied Sciences* **11**(6), 2808. YOLOv3, **86,77% mAP**. Pipeline WPCB-EFA: estimasi luas IC dari visi → massa via densitas permukaan **2.358 mg/mm²** → logam terpulihkan dari data daur ulang terpublikasi. Hasil: **1.079,18 g massa IC total**, **≥909,94 g terpulihkan**, **rata-rata USD 0,05 return per PCB**; Au/Ag/Pd/Pt ≈ **88% nilai daur ulang PCB**. ★★ **Template untuk mengubah klaster citra jadi uang — persis sudut terapan yang kita mau**
12. ★★ **(2026).** *"Image Similarity Judgment Method for Waste Printed Circuit Boards."* *Sensors* **26**(4), 1224. Lima fitur hand-crafted interpretable — hue, koefisien variasi luas terminal, jumlah garis di area terminal, kompleksitas area PCB, jumlah IC. **10 tipe lot WPCB. 88% akurasi, tanpa training, dan mengalahkan baseline kontrastif self-supervised.** ★★ Baseline/kontras yang sangat kuat untuk paper unsupervised
13. **(2025).** *"Artificial Intelligence Approach for Waste-Printed Circuit Board Recycling: A Systematic Review."* *Computers* **14**(8), 304

### Sorting robotik / disassembly
14. **Tripathi, Biju, Thota, Lingam (IIT Dharwad).** arXiv:2506.07122. **6.180 citra** komponen internal mouse dan charger yang dibongkar, + video 60 FPS simulasi conveyor. YOLOv11 mAP50 70,7 vs Mask R-CNN 39,7

### Sorting level material / spektral
15. **Picon, Galan, Bereciartua-Perez, Benito-del-Valle (2024).** WEEE hyperspectral + deep learning, arXiv:2407.04505
16. ★ **(2025).** *"Advancing WEEE Recycling: A Random Forest Approach to Classifying WEEE Plastics."* *Environments* **12**(2), 68. Prediksi recyclable vs non-recyclable **casing plastik flat-panel display dari metadata saja** (tipe polimer via FTIR, konsentrasi Br via XRF di ambang **2.000 ppm**, manufaktur, tahun, LED/LCD, ukuran). 15.006 sampel; **80–88% akurasi**. ★ **Preseden untuk menginferensi bahaya/recyclability dari atribut permukaan murah** — sangat dekat semangatnya dengan menginferensinya dari citra
17. **(2025).** *"Real-time high-accuracy sorting of imbalanced non-ferrous scraps in ELVs."* *J. Mater. Cycles Waste Manag.* ★ **penanganan ketidakseimbangan kelas** — klaster kita pasti tidak seimbang

### ★★ Indonesia
⚠️ **Temuan kunci: tidak ditemukan satu pun paper peer-reviewed berbahasa Indonesia tentang klasifikasi *citra* e-waste.** Semua kerja CV-on-waste Indonesia menyasar organik/anorganik atau sampah umum. **Ini celah riset yang nyata dan bisa dipertahankan — nyatakan eksplisit di pendahuluan paper.** Argumen kebaruan yang kuat untuk submission BDC.

Kerja Indonesia terdekat (semua sampah *umum*, semua bisa dikutip sebagai konteks domestik):
- **Bintang & Azhar (2024).** *"Implementasi Data Augmentation untuk Klasifikasi Sampah Organik dan Non Organik Menggunakan Inception-V3."* *JISKA* **9**(3):192–204. 25.500 citra, **94% akurasi**
- ★ **Afifah, Arumi, Maimunah, Nugroho (2025).** *"Comparative Evaluation of Preprocessing Methods for MobileNetV1 and V2 in Waste Classification."* *Jurnal RESTI* **9**(3):444–452 (SINTA). **CLAHE + bilateral filtering + MobileNetV1** terbaik: 85% train / 96% val. ★ Kutip untuk pilihan preprocessing
- Prasetio et al. (2024/2025). *"Deteksi Sampah Organik dan Anorganik Menggunakan Model YOLOv8."* *JIPI* **10**(1)
- Beberapa prosiding & skripsi: *Jurnal SIFO Mikroskil*, *JAIC Politeknik Negeri Batam*, *SEMNAS INOTEK UNP Kediri*, *Jurnal Janitra*, *Jurnal Informatika BSI*, repositori UPNVJ
- ⚠️ **"Sistem Bank Sampah Elektronik Menggunakan Deep Learning"**, repositori Telkom University item 239225 — item Indonesia e-waste + DL terdekat yang ditemukan, **halaman HTTP 403, belum terverifikasi. Kejar manual**
- Konteks kebijakan non-CV: *"Evaluasi pengelolaan limbah elektronik di Indonesia"* (*WASS* 1(1)); *"Dampak Sampah Elektronik (E-Waste) Terhadap Lingkungan Hidup"* (Univ. Muhammadiyah Mataram); ★ *"Study on potential household electronic waste generation and community perception in **Batam** city"* (AIP Conf. Proc.); *"E-waste, money and power: Mapping electronic waste flows in Yogyakarta, Indonesia"* (*The Extractive Industries and Society*)

### Tiongkok
⚠️ Literatur Mandarin khusus e-waste + citra tipis di sumber web terbuka; kebanyakan kerja 中文 CV soal 垃圾分类 umum. CNKI/Wanfang (tidak terindeks web-search) kemungkinan memberi lebih banyak.
- 《基于深度学习的垃圾识别与分类研究》 — tesis, Wanfang D03050811
- 《基于YOLOv8的可回收垃圾识别方法研究》 — Hans Publishers

## 8.4 ★★ Framing ekonomi sirkular — hook terapan terbaik

**Cucchiella, D'Adamo, Koh & Rosa (2015).** *"Recycling of WEEEs: An economic assessment of present and future e-waste streams."* *Renewable and Sustainable Energy Reviews*, **51**, 263–272.

| Perangkat | €/unit | €/kg |
|---|---|---|
| LED Notebooks | **51,3** | 14,7 |
| LCD Notebooks | **47,1** | 13,5 |
| CRT Monitors | 25,0 | 1,6 |
| ★ **Cell Phones** | 25,0 | **312,5** |
| LCD Monitors | 22,5 | 4,5 |
| ★ **Smartphones** | 19,0 | **158,3** |
| CRT TVs | 18,0 | 0,7 |
| LCD TVs | 10,3 | 1,0 |
| LED TVs | 10,2 | 1,0 |
| Tablets | 4,4 | 8,8 |
| HDDs | 2,0 | 3,4 |
| PV Panels | 0,04 | 0,0005 |

**Kutipan kunci: "emas adalah material yang menentukan separuh potensi pendapatan."** Dan kolom €/kg adalah statistik pembunuhnya — **satu ponsel bernilai ~300× lebih per kilogram dibanding satu TV CRT**. Kalau klaster kita memisahkan ponsel/laptop dari TV/white goods, itu memetakan langsung ke gradien nilai ~2–3 orde besaran. **Ini "so what" terkuat kita.**

**Urban mining:** Zeng, Mathews & Li (2018), *Environmental Science & Technology* **52**(8):4835 — *"ingot tembaga dan emas murni bisa dipulihkan dari aliran e-waste dengan biaya sebanding dengan penambangan bijih perawan."* ⚠️ Angka "13× lebih murah" yang sering diulang ada di balik paywall — **jangan kutip pengali itu tanpa membaca teks lengkap.**

**Reuse vs repair vs recycle:** *"Repair and Reuse or Recycle: What Is Best for Small WEEE in Australia?"* *Sustainability* **16**(7), 3035 — merekomendasikan memulai repair/reuse dengan **empat tipe produk: mainan, peralatan penyiapan makanan, vacuum cleaner, perkakas rumah tangga**. Kesimpulan kunci: **repair/reuse sendiri, tanpa mengurangi produk yang masuk pasar, hampir tidak menggerakkan volume limbah jangka pendek.**

## 8.5 ★★ Desain taksonomi klaster yang direkomendasikan

Bangun **skema penamaan tiga sumbu** supaya tiap klaster dapat label dengan sitasi di belakangnya, bukan nama karangan:

1. **Sumbu 1 — identitas regulatif/statistik.** Beri tiap klaster satu **UNU-KEY** (54 key) dan gulung ke **kategori WEEE Annex III (1–6)**. Kutip UNU Guidelines + Directive 2012/19/EU. Ini memberi tiap klaster kode, nama hukum, dan tonase dari GEM 2024. Roboflow 77-kelas sudah selaras UNU-KEY → bisa jadi **external validation set untuk kesepakatan klaster–taksonomi**.
2. **Sumbu 2 — fraksi material.** Ferrous / non-ferrous / plastik / PCB / kaca CRT / kabel / baterai. Grounding di kalimat **B107d** Indonesia (CRT + lampu TL + PCB + kawat logam) dan komposisi Indonesia Widyarsana & Nurdiani (31% ferrous, 27% kaca+plastik, 6% non-ferrous, 1% berbahaya).
3. **Sumbu 3 — hazard tier + nilai recovery.** Tier A/B/C, dan €/unit atau €/kg dari Cucchiella et al. Lalu laporkan per klaster estimasi nilai terpulihkan dan flag bahaya. **Itu mengubah hasil clustering unsupervised jadi kebijakan sorting yang bisa dieksekusi — persis yang memenangkan kompetisi jalur terapan.**

**Catatan hukum untuk kompetisi:** utamakan sumber **CC0 / MIT / Apache-2.0 / CC BY 4.0**. Tambahan teraman: **Roboflow E-Waste Dataset (CC BY 4.0, selaras UNU-KEY)**, **Hardware-ID (MIT)**, **akshat103 E-Waste (Apache 2.0)**, **mxtuhin E-Waste Compressed (CC0)**, **TrashNet/TACO/MJU-Waste (MIT)**, **RecyBat24**. **Hindari E-Waste Vision 18** (uploader eksplisit tidak memberi lisensi) dan cek apakah lisensi **NC** (ZeroWaste, E-Waste Hub) kompatibel dengan ketentuan panitia.

---

# LIBRARY: STATUS TERPASANG (container sesi ini, terverifikasi)

| Paket | Status | Isi |
|---|---|---|
| numpy 2.4.4, scipy 1.17.1, scikit-learn 1.8.0, matplotlib 3.10.9, Pillow | ✅ | dasar |
| torch 2.14.0, open_clip 3.3.0, timm 1.0.29 | ✅ | 1.737 tag pretrained, 196 pasangan open_clip |
| umap-learn, pacmap, trimap, openTSNE, phate | ✅ | reduksi dimensi manifold |
| hdbscan | ✅ | + `relative_validity_` = DBCV aproksimasi gratis |
| leidenalg, igraph, infomap, PhenoGraph, parc, scikit-network, networkx | ✅ | clustering graf/komunitas |
| scikit-learn-extra, kmedoids | ✅ | k-medoids (PAM, FasterPAM) |
| **clustpy** | ✅ | DEC, IDEC, DCN, DKM, VaDE, N2D, DeepECT, ACeDeC, DDC, DipDECK, DipEncoder, ENRC, AEC + XMeans, GMeans, PGMeans, DipMeans, ProjectedDipMeans, SkinnyDip, UniDip, DipNSub, DipExt, DipInit, GapStatistic, SpecialK, SubKmeans, LDAKmeans, Diana + **NrKmeans, OrthogonalClustering, ClusteringInOrthogonalSpaces, AutoNR** |
| pyclustering | ✅ | CURE, ROCK, BSAS/MBSAS/TTSAS, SyncNet, X-Means, G-Means |
| spectralnet | ✅ | spectral clustering neural |
| validclust | ✅ | Dunn, COP + wrapper |
| faiss-cpu | ✅ | kNN cepat |
| kneed | ✅ | Kneedle |
| prince | ✅ | MCA/CA/FAMD |
| giotto-tda | ✅ | TDA / Mapper / persistent homology |
| **ClusterEnsembles / ensembleclustering** | ❌ gagal pip | ambil dari github `827916600/ClusterEnsembles`, atau ko-asosiasi manual ~20 baris |
| **skdim** | ❌ gagal pip | estimasi dimensi intrinsik; TwoNN sudah diimplementasi manual di `analisis_ewaste.py` |
| scikit-hubness | belum dicoba | reduksi hubness; MP manual ~15 baris |
| permetrics, s-dbw, cdbw, clusteval, pycvi-lib, reval, krippendorff, statsmodels, pyclustertend, cleanlab, n2d, unidip | belum dipasang | pasang saat dibutuhkan |
| rpy2 + R (clusterCrit 42 indeks, NbClust 30 indeks, fpc, cluster, clValid, sigclust2, mclust) | belum dipasang | jalur untuk indeks eksotis |

---

# REKOMENDASI EKSEKUSI

## Prioritas 1 — murah, berdampak besar, bisa dikerjakan minggu ini
1. **Baseline panel penuh (CPU):** k-means, GMM, Ward, HDBSCAN, Leiden × 8 backbone × {tanpa, L2, PCA-50, UMAP-10} × k 2–30. Paper *Scaling Up* menunjukkan baseline ini berada ~7 poin dari SOTA ImageNet — bukan barang buangan
2. **Diagnostik dimensi tinggi:** TwoNN/MLE/FisherS ID, k-skewness hubness sebelum/sesudah, relative contrast. Semua ~10 baris, semua memberi angka yang bisa dilaporkan
3. **Null calibration:** marginal shuffle + Gaussian cocok-kovarians gaya M3C, B=200. Mengubah setiap indeks jadi klaim dengan p-value
4. **Protokol chooseR** — 100 subsample 80% → jarak ko-clustering → silhouette → aturan threshold CI 95%
5. **Multiverse / specification curve** + heatmap ARI antar spesifikasi

## Prioritas 2 — butuh satu run GPU
6. **Harmonisasi resolusi + re-embed** (Fase 0 roadmap)
7. **Encode bank prompt SigLIP2** di Kaggle → sub-ruang konsep 4 lensa
8. ★ **Adaptor C-RADIOv4** (`siglip2-g`, `dino_v3`, `sam3`) — 3 ruang tambahan gratis dari model yang sudah diunduh
9. **TEMI** (`--precomputed`, 4 GB VRAM, 5–45 mnt) — deep clustering terbaik rasio biaya/manfaat
10. ★ **TURTLE** — satu-satunya metode yang secara native mengonsumsi **beberapa** backbone frozen sekaligus, persis gunanya cache 8-backbone kita, <5 menit
11. **Backbone tambahan:** C-RADIOv4-H, DINOv3 ViT-H+, SigLIP2 g-opt, Franca ViT-G, MSN ViT-L

## Prioritas 3 — pembeda paper
12. **Uji mekanisme di Clevr-4** — kalau proyeksi sub-ruang konsep bekerja, ia harus memulihkan 4 kriteria ortogonal Clevr-4. Uji validitas termurah dan terbersih yang ada
13. **Ensembling gaya ICCE** — konsensus hipergraf Strehl-Ghosh atas head TEMI dan 8 backbone. Ide SOTA terbaru, murah, dan sumbu kebaruan yang bisa dipertahankan
14. **Studi manusia intruder-detection**, 3 rater, 30 item berstrata per klaster, Krippendorff's α
15. **Penamaan klaster otomatis** gaya TAC/DN-CBM (retrieval teks terdekat dari kosakata e-waste) — memberi tiap klaster label semantik + confidence tanpa anotasi
16. **Ekonomi:** join tabel Cucchiella €/kg ke label klaster → estimasi nilai recovery per klaster → treemap. Itu "so what"-nya

## Yang HARUS dihindari
- ❌ Clustering di koordinat UMAP/t-SNE **2-D**
- ❌ Argmax satu indeks validitas lalu menyebutnya jawaban
- ❌ Memakai silhouette untuk mengevaluasi partisi HDBSCAN/spectral (pakai DBCV)
- ❌ NbClust `alllong` di n=4179 tanpa subsampling
- ❌ GMM full-covariance di d=1000
- ❌ Rotasi acak sebagai null untuk indeks berbasis jarak
- ❌ Menganotasi subset validasi sambil browsing klaster (sirkular)
- ❌ Memakai stabilitas sebagai kriteria pemilihan k tanpa menanggapi Ben-David/von Luxburg
- ❌ Dataset dengan lisensi tidak jelas (E-Waste Vision 18) atau NC tanpa cek aturan panitia

---

# HAL YANG BELUM TERVERIFIKASI (jangan dikutip mentah)

1. ISBN Global E-waste Monitor 2024 — ambil dari kolofon PDF
2. Tabel per-kategori GEM 2024 — diperoleh via ringkasan pihak ketiga; cross-check ~hal. 40 PDF
3. Status Basel A1180 pasca-2025 — sumber sekunder bertentangan; baca keputusan BC-15/18
4. Teks verbatim WEEE Annex I / Annex III — EUR-Lex menolak reproduksi; salin sendiri sebelum submit
5. Jumlah citra persis `akshat103/e-waste-image-dataset` (~3.000 sering dilaporkan)
6. Repositori Telkom University "Sistem Bank Sampah Elektronik Menggunakan Deep Learning" — HTTP 403
7. "Urban mining 13× lebih murah dari penambangan perawan" — paywall; jangan kutip pengalinya
8. Kandungan logam per ton handset (130 kg Cu / 3,5 kg Ag / 300–350 g Au) — hanya di sumber pers/blog
9. iMClusts — judul/venue persis tidak terverifikasi; disitasi di literatur sebagai "Ren et al., 2022"
10. Survei *Multiple clusterings: Recent advances and perspectives*, Computer Science Review 2024 — ScienceDirect memblokir fetch; konfirmasi daftar penulis
11. "SpecialK" sebagai metode pemilihan k — ada di ClustPy tapi rujukan kanoniknya belum terverifikasi
12. "MFCVC" — tidak ditemukan di literatur image clustering; kemungkinan tertukar dengan MCA/MLC/GJR-CLIP
13. Tidak ditemukan SNI khusus klasifikasi jenis produk e-waste — cek SISPK BSN langsung sebelum menyatakannya di naskah
