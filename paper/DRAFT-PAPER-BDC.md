# Evaluasi Pengelompokan Perangkat Elektronik Berbasis Multi-View Frozen Embedding dan Analisis Material Visual Menggunakan Apple DMS46

## Abstrak

Foto sampah elektronik dapat memberi informasi tentang jenis perangkat dan material yang terlihat, tetapi perbedaan proses akuisisi juga dapat memengaruhi kelompok yang terbentuk dari embedding. Penelitian ini mengevaluasi pengelompokan perangkat berbasis multi-view frozen embedding dan penggunaan Apple DMS46 untuk membaca profil material visual. Data internal berisi 4.179 citra, yaitu 3.579 citra untuk membentuk cluster dan 600 citra holdout untuk evaluasi material. Sebelum ekstraksi fitur, resize dan re-encoding JPEG seragam menurunkan AUC prediksi asal foto dari embedding SigLIP2 dari 0,993 menjadi 0,875. Embedding dari delapan frozen backbone kemudian digabung melalui PCA dan concatenation, lalu dikelompokkan dengan K-Means. Setelah penelusuran K=4 hingga 24, adaptasi chooseR memilih K=12 dalam rentang lanjutan K=12 hingga 16. Pada K tersebut, rata-rata ARI terhadap medoid dari 100 seed mencapai 0,99747 dengan 95% CI [0,99564–0,99931]. Dalam evaluasi 710 citra Bangladesh dengan pemetaan cluster ke class yang ditetapkan dari data internal, Fusion-L2 memperoleh strict Macro Top-1 sebesar 0,888 dan CI selisih terhadap RADIO [0,029–0,070]. DMS46 mencapai macro AUROC 0,732 pada 1.018 citra Bangladesh, sementara penambahan indikator cluster menurunkan error prediksi deskriptor material pada holdout internal sebesar 11,45% setelah asal foto dan statistik akuisisi diperhitungkan. Hasil ini mendukung konsistensi kelompok perangkat, transfer pada empat class evaluasi, dan hubungan cluster dengan tampilan material. Namun, Fusion-L2 berbeda dari partisi Fusion yang diuji stabilitasnya, sedangkan komposisi material sesungguhnya dan efektivitas pemilahan fisik belum diuji.

Kata kunci: discovery perangkat, frozen embedding, multi-view fusion, stabilitas clustering, external validation.

## PENDAHULUAN

Sampah elektronik atau e-waste merupakan peralatan listrik dan elektronik yang telah dibuang oleh pemiliknya. Global E-waste Monitor 2024 mencatat bahwa dunia menghasilkan 62 juta ton sampah elektronik pada 2022, meningkat 82% dibandingkan 2010. Dari jumlah tersebut, hanya 22,3% yang tercatat dikumpulkan dan didaur ulang melalui sistem resmi yang memenuhi ketentuan pengelolaan lingkungan. Pengelolaan yang tidak sesuai dapat melepaskan zat berbahaya, termasuk merkuri dan bahan penghambat nyala berbromin, ke lingkungan (Baldé et al., 2024). Persoalannya mencakup kehilangan material yang dapat dipulihkan serta risiko kesehatan dan lingkungan.

Pemilahan sampah elektronik melibatkan pengenalan perangkat dan pemeriksaan materialnya. Pedoman Konvensi Basel untuk peralatan komputer menjelaskan bahwa perangkat diperiksa dan dipilah sebelum pembongkaran, kemudian komponennya dipisahkan untuk pengolahan material (Secretariat of the Basel Convention, 2025). Dalam tahapan tersebut, foto dapat membantu mengenali jenis perangkat dan material permukaannya. Komposisi seluruh perangkat dan keamanan penanganannya tetap memerlukan pemeriksaan lain.

Namun, kemiripan foto tidak selalu menunjukkan kemiripan perangkat. Jaipuria et al. (2022) menunjukkan bahwa clustering citra dapat mengikuti pencahayaan, bayangan, dan lingkungan pengambilan gambar. Temuan tersebut menjadi alasan untuk memeriksa isi tiap kelompok dan menguji apakah hasil pengelompokan tetap bermakna pada sumber foto lain.

### Penelitian Terkait

Penelitian sebelumnya telah menggunakan deteksi objek untuk mengenali komponen elektronik. Rajeev et al. (2025), misalnya, membandingkan YOLOv5, YOLOv7, dan YOLOv8 pada tujuh kategori komponen dan melaporkan mAP50 sebesar 0,821 serta mAP50–95 sebesar 0,678 untuk YOLOv8. Deteksi objek memberi lokasi dan kategori komponen, tetapi training memerlukan anotasi sesuai kategori yang ditetapkan. Karena data internal penelitian ini belum memiliki label fungsi perangkat, pertanyaannya adalah kelompok apa yang muncul dari kemiripan visual tanpa menetapkan label tersebut pada setiap citra sejak awal.

Untuk menelusuri kemiripan tersebut tanpa melatih ulang backbone pada data internal, penelitian ini menggunakan frozen embedding. DINOv2 mempelajari fitur melalui self-supervised learning, sedangkan SigLIP2 menggabungkan image-text pretraining dengan objektif visual tambahan (Oquab et al., 2024; Tschannen et al., 2025). Dalam benchmark yang menggabungkan embedding CLIP dan DINOv2, TURTLE memperoleh hasil lebih tinggi daripada K-Means (Gadetsky et al., 2024). Hasil tersebut memberi alasan untuk membandingkan K-Means dengan optimasi multi-view pada data penelitian ini.

Selain pemilihan backbone, kondisi citra yang masuk ke model juga perlu diperhatikan. Ehrlich et al. (2021) menemukan penurunan kinerja pada beberapa tugas vision ketika kompresi JPEG meningkat, sedangkan Jaipuria et al. (2022) menunjukkan bahwa kondisi foto dapat memengaruhi hasil clustering. Kedua temuan ini membantu menafsirkan pemisahan yang muncul pada pengelompokan awal. Model dapat menggunakan petunjuk visual yang tidak berkaitan dengan jenis perangkat, sebagaimana dibahas dalam shortcut learning (Geirhos et al., 2020).

Pada percobaan clustering sebelum input diseragamkan, cluster yang terbentuk terpisah kuat menurut sumber foto. Dalam partisi diagnostik K=16, dua cluster yang sama-sama memuat perangkat komputer bahkan terpisah menurut sumbernya. Temuan ini mendorong exploratory data analysis (EDA) untuk menelusuri penyebab pemisahan tersebut. Sebanyak 2.815 citra berukuran 150 × 150 piksel, sedangkan 1.364 citra lainnya memiliki resolusi lebih tinggi. Probe linier yang memprediksi asal foto dari embedding SigLIP2 menghasilkan AUC 0,993. Artinya, asal foto mudah dikenali dari representasinya, tetapi angka itu saja belum menunjukkan apakah penyebabnya adalah isi perangkat, ukuran citra, atau proses akuisisi lain.

### Rumusan Masalah

1. Apakah input seragam menurunkan AUC prediksi asal foto?

2. Bagaimana kualitas dan stabilitas cluster K-Means Fusion?

3. Apakah pemetaan cluster internal berlaku pada citra Bangladesh?

4. Apakah profil DMS46 membedakan class material dan berkaitan dengan cluster?

### Tujuan Penelitian

1. Mengukur AUC asal foto sebelum dan sesudah input diseragamkan.

2. Membandingkan K-Means Fusion dengan estimator lain, memilih K acuan, dan mengukur stabilitas cluster.

3. Mengukur strict Macro Top-1 pada empat class perangkat Bangladesh.

4. Mengukur AUROC DMS46 dan penurunan error profil material pada holdout internal setelah kontrol akuisisi.

### Manfaat Penelitian

Secara ilmiah, penelitian ini menyediakan pembanding empiris antara K-Means Fusion dan metode yang lebih kompleks pada citra multisumber, disertai pengujian pengaruh asal foto dan stabilitas cluster. Pemetaan cluster internal mencapai strict Macro Top-1 sebesar 0,888 pada empat class Bangladesh, sedangkan prediksi DMS46 membedakan class material dengan macro AUROC 0,732. Kedua informasi tersebut dapat menjadi dasar pengembangan sistem bantu pemeriksaan awal perangkat dan material yang terlihat, meskipun keputusan pemilahan dan nilai materialnya belum diuji.

## METODOLOGI

### Dataset

Data internal berisi 3.979 citra data train berlabel electronic dan 200 citra data test yang diprediksi electronic oleh sistem penyisihan, bukan berdasarkan ground truth panitia. Sebanyak 3.579 citra digunakan untuk membentuk cluster, sedangkan 600 citra disisihkan sebelum interpretasi sebagai holdout internal. PCA, centroid, dan model regresi material di-fit pada 3.579 citra tersebut, lalu dukungan deskriptor material dinilai pada holdout yang tidak ikut fitting. Pembagian ini tidak memerlukan label fungsi perangkat dan dirangkum pada Tabel 1.

[[TABLE1]]

Untuk menilai penerapan cluster pada sumber foto lain, penelitian ini menggunakan Custom Bangladeshi E-Waste Image Dataset for Object Detection and Recognition versi 1 (Afrin & Azmi, 2025), berlisensi CC BY 4.0. Empat dari 2.157 citra memiliki anotasi kosong dan dikecualikan. Evaluasi pengelompokan menggunakan 710 citra dari class baterai, ponsel, PCB, serta gabungan keyboard dan mouse. Keyboard dan Mouse digabung karena cluster keduanya dipetakan ke kelompok evaluasi yang sama, sedangkan 1.443 citra lainnya berada di luar cakupan pemetaan. Evaluasi material menggunakan 1.018 citra berbeda dari class kaca, logam, kertas, dan plastik.

### Preprocessing Citra dan Ekstraksi Fitur

Pemisahan sumber foto yang muncul pada clustering awal ditelusuri melalui EDA. Ukuran citra diperiksa lebih dulu, lalu tabel kuantisasi JPEG diperiksa setelah uji resize belum cukup mengurangi pemisahan. Untuk mengukur seberapa jelas asal foto tercermin dalam representasi, probe regresi logistik memprediksi asal citra dari embedding SigLIP2 melalui cross-validation tiga fold. Sebelum digunakan oleh probe, fitur dinormalisasi L2, direduksi dengan PCA menjadi 128 dimensi pada seluruh data, lalu dinormalisasi kembali. AUC 0,5 setara tebakan acak, sedangkan nilai mendekati satu menunjukkan asal citra mudah dikenali dari representasi tersebut.

Setelah perbedaan kedua korpus diperiksa, citra diproses dengan konfigurasi input yang sama. Setiap citra di-crop menjadi persegi, di-resize dengan LANCZOS menjadi 150 × 150 piksel, lalu di-encode ulang menjadi JPEG quality 75 dengan subsampling 4:2:0 sebelum transform native tiap backbone diterapkan. Label asal citra digunakan untuk evaluasi, bukan untuk menentukan perlakuan tiap korpus.

Fitur dari citra yang telah diproses diekstraksi dengan delapan backbone, yaitu DINOv3, C-RADIOv4, AIMv2, SigLIP2, DINOv2, SigLIP, ConvNeXtV2, dan EVA02. Bobot semuanya tetap saat digunakan pada data internal maupun Bangladesh. ConvNeXtV2 dan EVA02 memang telah menjalani fine tuning ImageNet oleh pengembangnya, tetapi penelitian ini tidak menggunakan bobot fine tuning tahap penyisihan. DINOv2 dan SigLIP disertakan untuk membandingkan generasi backbone, bukan karena manfaat tiap view dalam fusion sudah terbukti terpisah. Identitas checkpoint tersedia di Lampiran B.

Untuk menggabungkan kedelapan view, konfigurasi Fusion menormalisasi L2 setiap embedding dan mereduksinya dengan PCA menjadi 32 dimensi. Hasilnya digabung menjadi 256 dimensi, dinormalisasi L2, direduksi menjadi 100 dimensi, lalu dinormalisasi kembali. PCA menggunakan randomized solver dengan random state 0 tanpa whitening. Varian Fusion-L2 menambahkan normalisasi L2 setelah PCA per view. Konfigurasi dimensi ini belum diuji sebagai pilihan optimum.

[[EQ1]]

Dalam Persamaan (1), Eᵥ adalah embedding view v, N normalisasi L2, dan P transform PCA yang di-fit pada discovery. Fusion menghasilkan ruang fitur yang dikelompokkan K-Means dengan meminimalkan jumlah kuadrat jarak ke centroid (Lloyd, 1982). Prompt teks dan deskriptor material tidak masuk fungsi objektif. Proyeksi t-SNE hanya digunakan untuk visualisasi.

Gambar 1 memperlihatkan pembentukan cluster dari citra internal. Fusion digunakan untuk studi stabilitas dan contoh cluster, sedangkan Fusion-L2 digunakan untuk evaluasi pada Bangladesh dan analisis hubungan cluster dengan material. ARI antara partisi kedua varian pada K=12 mencapai 0,974794, sehingga hasil evaluasi Fusion-L2 tidak otomatis berlaku untuk partisi Fusion.

[[FIG1]]

### Pemilihan Metode dan Jumlah Cluster

Setelah ruang fitur dibentuk, K-Means Fusion dibandingkan dengan beberapa alternatif untuk menilai manfaat metode yang lebih kompleks pada data ini. Pembanding meliputi K-Means RADIO sebagai single-view, PAM pada ruang fused, MCSF dengan spectral loss dan contrastive alignment (Cai et al., 2026), serta TURTLE yang mengoptimalkan partisi multi-view (Gadetsky et al., 2024). MCSF dan TURTLE dijalankan melalui adapter lokal. Kualitas hasil dibaca dari silhouette untuk kekompakan cluster, ARI untuk kesepakatan assignment, dan NMI untuk hubungan cluster dengan asal foto. Adjusted coverage mengukur keragaman konsep fungsi setelah dikurangi hasil pengacakan skor di tiap sumber foto. Konsep tersebut menjadi diagnostik setelah clustering, bukan ground truth.

Jumlah cluster ditelusuri dalam dua tahap. Penelusuran awal mencakup K=4 hingga 24, lalu K=12 hingga 16 diperiksa lebih lanjut menggunakan 100 subset berisi 80% data discovery dengan seed 20260927. Pada setiap subset, PCA dan clustering di-fit ulang. Frekuensi co-clustering dihitung ketika kedua citra masuk subset dan dikurangkan dari satu untuk membentuk jarak consensus. Silhouette pada jarak ini dihitung terhadap medoid partisi penuh, mengikuti prinsip resampling chooseR pada data single-cell (Patterson-Cross et al., 2021).

Untuk memilih K dari rentang lanjutan, implementasi lokal menggunakan 25.000 percentile bootstrap atas median skor per citra. Batas bawah CI tertinggi antarkandidat menjadi threshold, lalu dipilih K dengan cluster terbanyak yang median skornya melampaui batas tersebut. Prosedur ini disebut adaptasi chooseR karena metode aslinya menggunakan distribusi skor per cluster dan CI BCa. Silhouette consensus menilai konsistensi setelah subsampling, sedangkan silhouette embedding menilai kekompakan ruang fitur.

Selain perubahan sampel, pengaruh inisialisasi K-Means diperiksa melalui 100 seed dengan n_init=20 pada setiap K lanjutan. Partisi medoid dipilih dari run yang rata-rata ARI-nya tertinggi terhadap run lain. ARI mengukur kesepakatan assignment, bukan ketepatan label perangkat, sedangkan Jaccard menilai perubahan tiap cluster setelah subsampling. Stabilitas dibandingkan dengan permutation null assignment, tetapi belum dengan null yang mempertahankan kovarians embedding.

### Interpretasi dan Evaluasi

Untuk membantu interpretasi, setiap kelompok diberi nama sementara setelah proses clustering selesai. Penamaan mengacu pada objek yang terlihat dalam citra dekat centroid, dengan skor konsep visual sebagai petunjuk tambahan. Gambar 6 mengikuti assignment medoid K=12 dan menampilkan dua contoh yang mudah dikenali dari 18 anggota terdekat ke centroid pada ruang Fusion 100 dimensi. Karena contoh tersebut berasal dari sekitar pusat cluster, namanya menggambarkan pola visual yang diperiksa, bukan kategori mayoritas yang telah dibuktikan untuk seluruh anggota.

Untuk menilai apakah pengelompokan dapat diterapkan pada sumber foto lain, citra Bangladesh diproses menggunakan transform Fusion-L2 atau RADIO yang di-fit pada data internal. Setiap citra kemudian ditempatkan pada centroid terdekat dari seluruh cluster, seperti ditunjukkan pada Gambar 2. Pemetaan cluster ke empat class perangkat ditetapkan dari profil visual data internal tanpa penyesuaian terhadap label Bangladesh, dan cluster tanpa pemetaan dihitung sebagai error. Strict Macro Top-1, yaitu rata-rata recall keempat class, dibandingkan dengan paired bootstrap pada citra yang sama. Skor tersebut berbeda dari macro-F1 yang memakai pemetaan in-sample.

[[FIG2]]

Selain kelompok perangkat, penelitian ini menganalisis material yang terlihat melalui dense material segmentation Apple DMS46 (Upchurch & Niu, 2022). Prediksi 46 class diringkas menjadi enam kategori, yaitu kaca, logam, kertas/karton, plastik, karet, dan lainnya. Proporsi area piksel yang diprediksi pada setiap kategori membentuk profil material citra, sebagaimana digambarkan pada Gambar 3. Untuk Bangladesh, area dihitung pada gabungan region of interest (ROI) dari anotasi YOLO, dengan polygon diubah menjadi enclosing bounding box. AUROC dan average precision menilai kemampuan profil tersebut membedakan empat class material pada dataset. Argmax tidak menjadi metrik utama karena satu ROI dapat mengandung beberapa material.

[[FIG3]]

Pada data internal, analisis dilanjutkan untuk menilai apakah cluster menambah informasi tentang profil material. Proporsi area lebih dulu ditransformasi dengan akar kuadrat dan dinormalisasi L2. Model baseline menggunakan asal citra serta statistik akuisisi dan latar, sedangkan model full menambahkan indikator cluster. Kedua model di-fit pada discovery dan dinilai pada holdout internal. Perubahan error prediksinya dirumuskan dalam Persamaan (2).

[[EQ2]]

Persamaan (2) membandingkan jumlah kuadrat error prediksi (SSE) model full dan baseline pada target hasil transformasi akar. R² incremental menunjukkan penurunan error relatif terhadap baseline, bukan proporsi total variance yang dijelaskan. CI dihitung melalui bootstrap holdout terstratifikasi web/field, sedangkan permutation null diperoleh dari 999 pengacakan assignment cluster di dalam asal citra. Ketidakpastian ini bersyarat pada model dan partisi yang telah di-fit, sehingga tidak mencakup pemilihan seluruh kandidat.

## HASIL DAN PEMBAHASAN

### Analisis Eksploratif Data dan Preprocessing

Setelah clustering awal memperlihatkan pemisahan menurut sumber foto, EDA menemukan perbedaan ukuran antara kedua korpus. Karena itu, resize diuji lebih dulu. Seperti terlihat pada Gambar 4, langkah ini hanya menurunkan AUC probe SigLIP2 dari 0,993 menjadi 0,983, sehingga asal foto masih mudah dikenali dari embedding. Pemeriksaan kemudian beralih ke tabel kuantisasi JPEG. Sebanyak 2.814 dari 2.815 citra web memiliki tabel identik dengan standar IJG quality 75 dan subsampling 4:2:0, sedangkan tabel foto beresolusi tinggi lebih beragam. Setelah resize disertai re-encoding JPEG seragam, AUC turun menjadi 0,875.

[[FIG4]]

Untuk menilai hasil pada foto dengan jenis objek yang lebih serupa, probe juga diuji pada 380 foto perangkat komputer dari cluster 08 dan 09 pada partisi awal. Keduanya memperlihatkan monitor, keyboard, dan mouse, tetapi terutama berasal dari sumber foto yang berbeda. Setelah resize dan re-encoding JPEG, AUC pasangan ini turun dari 0,999 menjadi 0,523, mendekati nilai acak 0,500. Artinya, asal foto jauh lebih sulit dibedakan di antara citra berkonten sejenis. Namun, AUC seluruh data masih 0,875, sehingga perbedaan antar-sumber belum hilang secara global dan mungkin juga berkaitan dengan perangkat atau latar.

Kedua langkah tersebut digunakan sebelum ekstraksi fitur karena resize saja tidak cukup menurunkan keterbedaan asal foto. Hasil probe tetap tidak menetapkan JPEG sebagai satu-satunya penyebab perbedaan cluster.

### Perbandingan Metode dan Pemilihan K

Setelah input diproses dengan konfigurasi yang sama, delapan keluarga metode ditelusuri. Tabel 2 merangkum delapan estimator terpilih pada K=12 agar K-Means Fusion dapat dibandingkan dengan single-view, estimator centroid lain, dan beberapa metode multi-view pada jumlah cluster yang sama.

[[TABLE2]]

Dari sisi kekompakan, K-Means Fusion memperoleh silhouette 0,305, sedangkan PAM, MCSF, dan evidence accumulation berada pada 0,300–0,302. Namun, evidence accumulation memberi adjusted coverage lebih tinggi (0,369 dibanding 0,355). FGW MB4 lebih lemah pada silhouette dan coverage (0,243 dan 0,150), walaupun NMI asal fotonya lebih rendah daripada K-Means Fusion (0,012 dibanding 0,160). Dengan hasil tersebut, K-Means digunakan sebagai acuan karena kekompakannya pada perbandingan ini, bukan karena unggul pada setiap ukuran. Ketika tiap metode memakai K pilihannya sendiri, evidence accumulation bahkan mencapai silhouette 0,314 dibanding 0,312 untuk K-Means Fusion. Perbedaan implementasi dan dataset membatasi perbandingan angka ini dengan hasil TURTLE pada Gadetsky et al. (2024).

Perbandingan backbone membantu menjelaskan mengapa versi lebih baru tidak otomatis menggantikan versi sebelumnya pada data ini. Pada K=12 dan seed 42, silhouette single-view DINOv3 mencapai 0,286 dibanding DINOv2 sebesar 0,258. Arah perbandingannya berbalik pada SigLIP, yaitu 0,295 dibanding SigLIP2 sebesar 0,278. Sebagai pemeriksaan jumlah view pada metode lain, TURTLE K=12 mencapai 0,248 dengan DINOv3, RADIO, SigLIP2, dan ConvNeXtV2, lalu 0,254 dengan delapan view. Hasil tersebut belum mengisolasi manfaat masing-masing backbone dalam Fusion dengan K-Means. Karena itu, delapan view dilaporkan sebagai konfigurasi yang diuji, bukan kombinasi optimal.

Penelusuran awal berdasarkan silhouette embedding dan indikator semantik setelah clustering mengarah ke K=14. Namun, kedua ukuran itu belum menunjukkan apakah assignment bertahan ketika data berubah. Karena itu, K=12 hingga 16 diperiksa lagi melalui resampling. Adaptasi chooseR memilih K=12 sebagai jumlah cluster acuan. Gambar 5 memisahkan hasil penelusuran awal dari pengujian stabilitas tersebut.

[[FIG5]]

Adaptasi chooseR menghasilkan threshold 0,997354. Pada K=12, tujuh cluster melampauinya, sedangkan K lain memiliki paling banyak lima. K=12 juga memberi rata-rata ARI terhadap medoid sebesar 0,99747 dengan 95% CI [0,99564–0,99931], dan 99 dari 100 run mencapai ARI ≥0,95. Sebagai pembanding, K=14 memperoleh rata-rata ARI 0,97730 dengan 69 run melewati batas tersebut. Pada tingkat cluster, median Jaccard K=12 mencapai 0,9955 dan tidak ada kuantil 2,5% yang jatuh di bawah 0,60.

K=12 berada pada batas bawah pemeriksaan lanjutan, sehingga pemilihannya belum menetapkan jumlah cluster yang benar atau optimum global. Pada penelusuran awal, silhouette naik dari 0,305 pada K=12 menjadi 0,321 pada K=16, tetapi adjusted coverage turun dari 0,355 menjadi 0,258. Pada K=18 hingga 24, silhouette turun dari 0,309 menjadi 0,257 dan adjusted coverage dari 0,222 menjadi 0,132. Penambahan cluster di atas K=16 terutama menghasilkan pengulangan deskriptor konsep perangkat, bukan bukti kategori baru yang stabil. NMI terhadap asal citra pada K=12 masih 0,160, sehingga assignment yang stabil tetap dapat berkaitan dengan asal foto.

### Interpretasi Hasil Clustering

Setelah stabilitas cluster diperiksa, Gambar 6 memperlihatkan pola visual melalui dua contoh dari tiap cluster. Contohnya mencakup PCB, mesin cuci, mouse, laptop, ponsel, printer, perangkat audio, baterai, microwave, TV, dan keyboard. Pada C08, citra dekat centroid justru memperlihatkan tumpukan berbagai perangkat, sehingga cluster ini diberi nama deskriptif “tumpukan perangkat”. Nama tersebut merangkum contoh yang diperiksa, bukan proporsi jenis objek pada seluruh anggota C08.

[[FIG6]]

C02 berisi 293 citra dengan contoh mouse, sedangkan C11 berisi 274 citra dengan contoh keyboard. Median Jaccard keduanya adalah 1,000 dan 0,9954, dengan kuantil 2,5% sebesar 0,9936 dan 0,9008. Keduanya tetap terpisah dalam pengujian subsampling, walaupun sama-sama digunakan untuk memberi input ke komputer. Hasil ini menunjukkan pemisahan visual yang stabil, tetapi belum membuktikan bahwa algoritma memahami hubungan fungsi keduanya.

Pada resolusi yang lebih rinci, cluster layar C10 pada K=12 terutama terpecah menjadi C05 dan C12 pada K=16. Sebanyak 142 dari 145 anggota C05 dan 135 dari 137 anggota C12 berasal dari C10. Contoh dekat centroid memperlihatkan televisi tabung pada C05 dan layar datar pada C12 (Gambar C1). Kedua cluster juga relatif stabil saat sampel berubah, dengan kuantil 2,5% Jaccard sebesar 0,936 dan 0,955. Proporsi foto web masing-masing 0,814 dan 0,839, dibanding 0,832 pada C10. Bukti ini mendukung dua pola visual di pusat cluster, tetapi belum mengukur purity seluruh anggota atau memisahkan televisi dari monitor secara konsisten.

Pemisahan layar pada K=14 kurang stabil dibanding dua cluster layar pada K=16 karena sekitar 40–41% subset menghasilkan Jaccard di bawah 0,60. Pemecahan laptop pada K=16 juga bertahan, tetapi proporsi sumber fotonya berbeda sehingga makna semantiknya belum jelas. Pemeriksaan konsep visual pun belum mendukung stopkontak, steker, atau terminal listrik sebagai cluster tersendiri.

### Validasi Eksternal

Ketika transform dan centroid yang di-fit pada data internal diterapkan pada 710 citra Bangladesh, strict Macro Top-1 Fusion-L2 K=12 mencapai 0,888 (95% CI [0,865–0,913]). RADIO K=12 memperoleh 0,838 (95% CI [0,813–0,864]), sehingga selisih berpasangannya sekitar 0,050 dengan CI [0,029–0,070] (Gambar 7). Untuk Fusion-L2, null permutation menghasilkan kuantil 95% sebesar 0,264 dan p=0,001.

[[FIG7]]

Selisih agregat terutama berasal dari baterai, dengan recall Fusion-L2 sebesar 0,878 dibanding RADIO 0,628. RADIO sedikit lebih tinggi pada ponsel, PCB, serta keyboard dan mouse, sehingga peningkatan Fusion-L2 tidak merata pada seluruh class. Sebanyak 95,8% citra mendapat assignment ke centroid yang memiliki pemetaan kelompok, sedangkan 4,2% sisanya tetap dihitung sebagai error. Seluruh centroid disertakan agar accuracy tidak meningkat hanya karena pencarian dibatasi pada cluster yang telah dipetakan.

Hasil transfer ini terbatas pada empat class evaluasi perangkat karena 1.443 citra Bangladesh di luar cakupan pemetaan tidak masuk perhitungan accuracy. Pengujian pada dataset lain memberi pemeriksaan generalisasi tambahan (Recht et al., 2019), tetapi label Bangladesh hanya menunjukkan kategori objek. Karena itu, skor transfer belum memvalidasi kondisi limbah atau seluruh jenis perangkat pada data internal, dan independensi citra lintas dataset juga belum dipastikan sepenuhnya.

### Analisis Deskriptor Material

Analisis material kemudian memeriksa apakah cluster memberi informasi tambahan tentang profil material setelah asal citra dan statistik akuisisi diperhitungkan. Pada 600 citra holdout internal, penambahan indikator cluster Fusion-L2 K=12 menurunkan SSE sebesar 11,45% relatif terhadap baseline, dengan 95% CI [8,89%–14,09%]. Ketika kategori lainnya dikeluarkan dan proporsi lima material dinormalisasi ulang, penurunan menjadi 15,28% dengan CI [12,10%–18,45%]. Dua citra tanpa area prediksi pada kelima material dikecualikan dari analisis sensitivitas ini, sehingga tersisa 598 citra. Kedua pengujian menghasilkan permutation p=0,001.

Target regresi pada holdout berasal dari prediksi DMS46, sehingga penurunan error tersebut belum memvalidasi material sebenarnya. Untuk memeriksa deskriptor DMS46 pada sumber lain, evaluasi 1.018 citra Bangladesh menghasilkan macro AUROC 0,732 dengan 95% CI [0,710–0,752] dan macro AP 0,501. AUROC yang melampaui 0,500 menunjukkan adanya sinyal pembeda terhadap empat class material sumber, walaupun logam masih memiliki AUROC terendah, yaitu 0,683. Karena dataset tidak menyediakan ground truth per piksel, hasil ini belum menilai ketepatan luas material yang diprediksi.

Pembatasan ke ROI diuji karena anotasi Bangladesh menunjukkan lokasi objek yang dinilai. Namun, macro AUROC ROI sebesar 0,732 dibanding rembg 0,739 menghasilkan selisih berpasangan −0,008 dengan CI [−0,015–0,0004]. Karena interval tersebut mencakup nol, hasil ini belum mendukung peningkatan AUROC melalui ROI.

Profil deskriptif Fusion-L2 pada 4.179 citra membantu membaca hubungan antara kelompok perangkat dan material yang tampak. Pada kelompok printer, rata-rata area prediksi plastik mencapai 54,2% dibanding 33,0% pada seluruh data. Kelompok layar memiliki area prediksi kaca 28,4% dibanding 10,0% secara keseluruhan, sedangkan kategori lainnya masih mencakup rata-rata 48,9% area pada kelompok baterai. Gambar 8 memperlihatkan keluaran DMS46 pada dua citra internal, yaitu printer dan TV, serta dua citra Bangladesh, yaitu keyboard dan PCB. Profil pada citra internal dihitung dari seluruh gambar, sedangkan profil Bangladesh dihitung dari area anotasi objek. Contoh tersebut membantu membaca prediksi model, tetapi tidak mewakili seluruh anggota cluster atau komposisi material sesungguhnya.

[[FIG8]]

### Keterbatasan Penelitian

Meski penyeragaman input menurunkan AUC asal foto, NMI sekitar 0,16 menunjukkan bahwa proses akuisisi masih berkaitan dengan cluster. Pemeriksaan data juga menemukan enam pasangan citra byte-identik yang melintasi discovery dan holdout internal. Pasangan tersebut mengurangi independensi evaluasi holdout, tetapi besar pengaruhnya terhadap skor belum diukur. Cakupan data turut dibatasi oleh 200 citra data test yang masuk berdasarkan prediksi electronic, bukan label panitia, sementara kemungkinan near-duplicate antara data internal dan Bangladesh belum disingkirkan.

Validasi stabilitas 100 seed dan adaptasi chooseR dilakukan pada fused K-Means, sehingga tingkat pengujiannya belum setara dengan semua alternatif. Perbandingan single-view serta set empat dan delapan view juga belum mengisolasi kontribusi tiap backbone dalam fusion final. Di sisi data, ground truth fungsi untuk seluruh citra internal dan material per piksel belum tersedia. Banyak foto memperlihatkan produk utuh atau tampilan katalog, sehingga hasil penelitian belum menunjukkan kondisi limbah maupun keberhasilan pemilahan fisik.

## PENUTUP

Penelitian ini menilai pengelompokan perangkat dari multi-view frozen embedding serta kaitannya dengan material visual yang diprediksi Apple DMS46. Pada data internal, resize dan re-encoding JPEG seragam menurunkan AUC asal citra SigLIP2 dari 0,993 menjadi 0,875. Adaptasi chooseR memilih K=12 dalam rentang K=12 hingga 16, dengan rata-rata ARI terhadap medoid 0,99747 pada 100 seed. K-Means memberi silhouette yang sebanding dengan PAM dan MCSF pada konfigurasi yang dibandingkan. Contoh dekat centroid memperlihatkan pola mouse, keyboard, serta tumpukan perangkat. Pada resolusi lebih tinggi, sebagian pemecahan cluster bertahan saat sampel berubah, sementara yang lain lebih sensitif.

Pada sumber eksternal Bangladesh, Fusion-L2 K=12 mencapai strict Macro Top-1 sebesar 0,888 untuk empat class perangkat, lebih tinggi secara agregat daripada RADIO. Informasi cluster juga menurunkan error prediksi deskriptor material pada holdout internal sebesar 11,45% setelah asal citra dan statistik akuisisi diperhitungkan. Dengan demikian, kelompok perangkat dan profil material visual berpotensi menjadi dua informasi awal untuk pemilahan, meskipun pengujian ini belum memvalidasi persis medoid Fusion, seluruh jenis perangkat pada data internal, atau komposisi material sesungguhnya.

Pengujian berikutnya perlu menyamakan konfigurasi fusion antarevaluasi, menangani duplikat lintas split, dan memperluas cakupan perangkat. Untuk menilai manfaat dalam pemilahan fisik, penelitian lanjutan juga memerlukan foto kondisi limbah dan pemeriksaan material di luar prediksi visual.

## REFERENSI

[[REFERENCES]]

## LAMPIRAN

### Lampiran A Sensitivitas antarresolusi

[[TABLEA]]

Containment adalah proporsi anggota cluster pada K lebih tinggi yang berasal dari satu cluster pada K lebih rendah, bukan purity terhadap ground truth. Pada K=14, dua pecahan layar sensitif terhadap subsampling. Pada K=16, dua pecahan layar lebih stabil dan contoh dekat centroid memperlihatkan bentuk tabung dan layar datar. Proporsi web yang mirip tidak menghilangkan kemungkinan pengaruh akuisisi lain.

### Lampiran B Parameter dan source code inti

[[GITHUB]]

### Lampiran C Contoh pemecahan cluster layar

[[FIGC]]
