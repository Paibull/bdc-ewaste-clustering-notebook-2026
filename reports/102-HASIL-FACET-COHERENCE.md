# Hasil facet coherence pada cluster data-driven

Tanggal eksekusi: 26 September 2026  
Status: **POSITIF — seluruh primary/secondary/joint gate lulus**

## Pertanyaan yang dijawab

Eksperimen menguji apakah cluster data-driven mengandung pola material dan
visible condition di luar perbedaan function dan source. Partisi, `k`,
representation, transformasi, null, dan gate dibekukan di
`101-PREREGISTRASI-FACET-COHERENCE.md` sebelum hasil dihitung.

Tidak ada clustering atau model inference baru pada analisis utama. DMS46,
visible-damage score, dan cluster yang sudah tersimpan digunakan kembali.

## Hasil utama

| Facet | Partial R² setelah function×source | Null mean | Null 97,5% | Permutation p | Korelasi profil holdout | Gate |
|---|---:|---:|---:|---:|---:|---|
| Material composition | **0,0339** | 0,0030 | 0,0043 | **0,001** | **0,6818** | lulus |
| Visible condition | **0,2434** | 0,0039 | 0,0081 | **0,001** | **0,9692** | lulus |
| Joint | **0,0556** | 0,0030 | 0,0043 | **0,001** | **0,7875** | lulus |

Cluster data-driven menambah 3,39% explained residual variance pada descriptor
material dan 24,34% pada visible-damage score setelah one-hot
`function × source` masuk sebagai baseline. Seluruh nilai jauh di atas 999 null
yang mengacak cluster hanya di dalam control strata.

Condition tetap secondary diagnostic: score berasal dari branch G8-C yang
memiliki external signal, tetapi sebelumnya tidak mengungguli existing
condition baseline sebesar promotion margin yang dipersyaratkan.

## Edge cluster–facet

Dari 84 pasangan cluster–facet, 12 memenuhi dua syarat yang dibekukan:
`BH q <= 0,05` dan absolute residual mean `>= 0,20 SD`. Semua 12 mempertahankan
arah yang sama pada 600 locked holdout.

| Cluster | Facet | Residual mean (SD) | CI 95% | Holdout | Pembacaan terbatas |
|---|---|---:|---:|---:|---|
| D00 laptop | glass | +0,233 | [0,145; 0,318] | +0,261 | lebih kaya signal kaca daripada laptop/source sebanding |
| D00 laptop | metal | −0,216 | [−0,309; −0,114] | −0,066 | lebih rendah pada descriptor logam |
| D00 laptop | visible damage | −0,480 | [−0,596; −0,364] | −0,575 | cenderung utuh/kerusakan visual rendah |
| D03 input computer | paper/cardboard | −0,381 | [−0,486; −0,281] | −0,290 | lebih rendah pada signal kertas/kardus |
| D03 input computer | rubber | +0,495 | [0,323; 0,670] | +0,849 | signal karet lebih tinggi |
| D06 mixed waste pile | glass | −0,468 | [−0,561; −0,377] | −0,321 | signal kaca lebih rendah |
| D06 mixed waste pile | metal | +0,264 | [0,164; 0,369] | +0,316 | signal logam lebih tinggi |
| D06 mixed waste pile | paper/cardboard | +0,247 | [0,166; 0,318] | +0,030 | arah sama, efek holdout kecil |
| D06 mixed waste pile | visible damage | +0,869 | [0,785; 0,946] | +0,876 | visible damage paling kuat |
| D08 input computer | other/unknown | −0,229 | [−0,384; −0,081] | −0,037 | arah sama, efek holdout kecil |
| D08 input computer | paper/cardboard | +0,418 | [0,299; 0,531] | +0,167 | signal kertas/kardus lebih tinggi |
| D08 input computer | rubber | −0,453 | [−0,554; −0,353] | −0,458 | memisahkan subtype input dari D03 |

Edge adalah association descriptor, bukan kandungan material kimia dan bukan
accuracy condition.

## Sensitivity discovery-only fit

Audit setelah run menemukan bahwa partisi final historis dihitung transductively
pada 4.179 citra. Karena itu ditambahkan sensitivity post-hoc tanpa mengubah gate:
K-Means `k=12` difit hanya pada 3.579 discovery, lalu 600 holdout diprediksi.

- ARI terhadap partisi historis: `0,9635` pada discovery dan `0,9738` pada holdout.
- Material: partial R² `0,0328`, `p=0,001`, profile correlation `0,6682`.
- Condition: partial R² `0,2399`, `p=0,001`, profile correlation `0,9685`.
- Joint: partial R² `0,0542`, `p=0,001`, profile correlation `0,7754`.

Kesimpulan tidak bergantung pada transductive fitting partisi lama. Sensitivity
ini tetap diberi label post-hoc dan bukan primary gate preregistrasi.

## Keputusan ilmiah

Eksperimen memberi evidence positif bahwa cluster data-driven bukan hanya nama
function yang berulang. Setelah function dan source dikontrol, cluster masih
membawa perbedaan komposisi material serta visible condition yang stabil pada
subset evaluasi.

Hasil ini mendukung framing **faceted discovery**:

1. function/device menjadi structural backbone;
2. material composition dan visible condition menjadi facet yang dapat
   melintasi atau membedakan subtype di dalam struktur tersebut;
3. satu hard taxonomy tidak perlu dipaksa memuat semua karakteristik e-waste.

Klaim yang diizinkan:

> Fixed data-driven clusters exhibit material-composition and visible-condition
> coherence beyond function and acquisition source controls.

Klaim yang tidak diizinkan: material ground truth, condition accuracy pada
seluruh target, kandungan material tersembunyi, atau causal effect formal.

## Peran dalam karya semifinal

Facet coherence dapat menjadi hasil positif pendamping harmonisasi, bukan daftar
eksperimen tambahan. Negative run lain tidak perlu diceritakan satu per satu.
Hanya ablation yang menjawab alternatif langsung yang dipertahankan di badan
karya, misalnya resolution-only, feature-space correction, dan foreground-only.
Eksperimen lain dipindahkan ke appendix atau tidak dibahas.

## Artefak dan reproducibility

- Runner: `semifinal/scripts/analyze_facet_coherence.py`;
- metrics: `semifinal/results/facet_coherence/metrics.json`;
- profile lengkap dan CI: `semifinal/results/facet_coherence/profiles.csv`;
- edge dan corrected p-value: `semifinal/results/facet_coherence/facet_edges.csv`;
- visual: `semifinal/results/facet_coherence/facet_coherence.pdf`;
- provenance dan checksum: `semifinal/results/facet_coherence/metadata.json`.

SHA-256 PDF: `2ea621606ffcaec6c33ac875540e95c8ab915f45e06942c72c641cefe42b0ee8`.
