# Final scientific audit semifinal

Tanggal audit: 25 September 2026 (WIB)  
Status keseluruhan: **menunggu file human validation; seluruh pekerjaan non-manusia selesai**

Dokumen ini mengaudit phase C1, S1, M1, MatSim, DMS46, common comparison, dan
handoff human validation terhadap preregistrasi dan master plan. Tidak ada
eksperimen, tuning, seed fishing, atau perubahan final partition setelah hasil
dibuka.

## Keputusan per branch

| Branch | Bukti utama | Keputusan ilmiah |
|---|---|---|
| C1 condition residual | DINOv2 residual stability `0,9873` < matched-covariance q97.5 `0,9978`; function probe turun `0,9283 -> 0,3504` | **NEGATIVE** untuk condition discovery; stop tanpa tuning |
| S1 global shape | Morphology global largest cluster `71,1%`; aggressive split menurunkan DBCV | **NEGATIVE** sebagai taxonomy shape global |
| S1 targeted shape | CRT/flat-panel device sub-clusters dengan matched-tier source caveat | **EXPLORATORY_POSITIVE**, hanya untuk device family layar |
| M1 independent DMS gate | Tidak ada CSV manusia independen; perbandingan agent-only pooled kappa `0,6949` | **DIAGNOSTIC_ONLY**; bukan human gate |
| MatSim compatibility | Smoke descriptor `[1,512]`, finite, norm `1,0` | **EXPLORATORY_POSITIVE** secara teknis |
| MatSim pilot | Source AUROC `0,7526` > prereg max `0,75` | **NEGATIVE** untuk membuka full run; stop tanpa rescue |
| DMS46 full composition | 4.179/4.179; stability `0,9925`; null lower-tail `p=0,0099`; source AUROC `0,7894` | **EXPLORATORY_POSITIVE / DIAGNOSTIC_ONLY**; belum semantic-validated |
| Common comparison | DMS dispersion `0,3370`; DMS–SigLIP AMI `0,0683`; key/split identik | **DIAGNOSTIC_ONLY**; tidak memilih winner |
| Final material semantic claim | Tiga reviewer belum mengembalikan CSV | **BLOCKED_BY_HUMAN_GATE** |

## Acceptance audit

| Requirement | Evidence saat ini | Status |
|---|---|---|
| C1 prereg, run, null, gate, report | `87-PREREGISTRASI...`, `88-HASIL...`, `results/c1_condition_residual/` | Lulus sebagai branch negatif |
| S1 tanpa inference baru | `89-SHAPE-EVIDENCE.md` dan artefak SSM existing | Lulus |
| MatSim source, checksum, compatibility, pilot, stop | `92-KEPUTUSAN...`, `93-HASIL...`, `results/matsim_pilot/` | Lulus; full run correctly not executed |
| DMS output full 4.179-key | `results/dms46_full/status.json`, `fractions.jsonl`, `metrics.json` | Lulus sebagai exploratory deviation |
| Common protocol | `results/common_comparison/metrics.csv`, `assignments.csv`, `REPORT.md` | Lulus untuk representation yang tersedia |
| Blind human bundle | `semifinal/human_validation_final/final_penilai_1..3.zip` | Siap; belum ada label manusia |
| Validator/scorer | `scripts/score_final_human_validation.py` | Siap; menunggu CSV |
| Final partition tidak berubah | `final_partition_modified=false` pada branch outputs; no replacement in common runner | Lulus |
| Existing regression | `pytest -q semifinal/scripts` | **47 passed** |

## Deviation yang harus disebut eksplisit

M1 human gate awal tidak dikembalikan oleh reviewer independen. Atas waiver
pengguna, DMS46 full tetap dijalankan sebagai exploratory run. Ini **bukan**
kelulusan retroaktif M1 dan tidak mengubah status semantic DMS menjadi human
validated. Karena itu DMS hanya boleh dipakai sebagai compositional diagnostic
dan bahan hipotesis, bukan sebagai ground truth material.

MatSim full tidak dijalankan. Compatibility yang berhasil tidak menghapus
source-gate failure pada pilot. Branch dihentikan sesuai stop rule.

Bundle lama di `semifinal/intruder_study/bundles/` tidak dihitung sebagai final
cross-representation validation karena menampilkan lens/name proxy dan tidak
memuat assignment DMS-versus-SigLIP yang dibekukan pada common comparison.

## Handoff human yang masih diperlukan

Distribusikan hanya tiga ZIP berikut, satu per reviewer:

- `semifinal/human_validation_final/final_penilai_1.zip`
- `semifinal/human_validation_final/final_penilai_2.zip`
- `semifinal/human_validation_final/final_penilai_3.zip`

Setiap reviewer mengembalikan CSV dengan nama yang sudah ditentukan dan
`independence_declaration.txt`. Letakkan CSV di
`semifinal/human_validation_final/`, lalu jalankan:

```powershell
python semifinal/scripts/score_final_human_validation.py
```

Scorer akan menolak file yang missing, duplicate, salah task ID, salah schema,
atau memakai nilai di luar kontrak. Sampai command itu berhasil dengan tiga
CSV manusia, klaim semantic material tetap `BLOCKED_BY_HUMAN_GATE`.

## Checksum dan reproducibility

Common assignment manifest SHA-256:
`731ad6d8ecc3abb012cc8d5259374b9fb9ff2a619d69023638e41ab53150e74e`.

SHA-256 ZIP final human validation:

| File | SHA-256 |
|---|---|
| `final_penilai_1.zip` | `bfcd9aee17b7b33e913e09cf5d9c64375b42708b2f17f4e47fd6c4dc9aa5ce92` |
| `final_penilai_2.zip` | `c2e3ae9f3f08b4786f46eadb710d9d5717c6f7d6d21fef9c3046fac91c6e5320` |
| `final_penilai_3.zip` | `324bc6eba59c7624b5f98e352c6cc44bf964ec69e6128898650d10cc3782a712` |

Bundle contract: 60 tugas per reviewer, 207 image lokal per ZIP, discovery
only, holdout 600 tidak disertakan, dan tidak ada kunci internal di ZIP.

## Kesimpulan handoff

Evidence terkuat untuk karya tetap temuan acquisition/JPEG confound dan
targeted CRT-versus-flat-panel. C1 condition residual gagal gate stability.
DMS46 memberi signal komposisional yang menarik dan berbeda dari SigLIP2, tetapi
belum cukup untuk menyebut cluster sebagai material discovery final tanpa
human validation. MatSim merupakan compatibility success yang berhenti secara
defensible pada source gate.
