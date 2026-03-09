# Chest X-Ray Multi-Label Classification
> Klasifikasi Multi-Label Citra X-Ray untuk Deteksi Pneumonia dan Tuberkulosis (TBC)
---

## Deskripsi

Klasifikasi multi-label citra X-ray dada untuk deteksi 7 abnormalitas paru yang berkaitan dengan **Pneumonia** dan **Tuberkulosis (TB)**, menggunakan kombinasi deep learning dan handcrafted features, beserta LLM untuk pembuatan laporan sederhana.

---

## Target Label

| Label |
|---|
| Fibrosis |
| Infiltration |
| Nodule |
| Mass |
| Consolidation |
| Effusion |
| Atelectasis |

---

## Arsitektur Model

- ResNet-50 + Handcrafted + ANN
- DenseNet-121 + Handcrafted + ANN
- Vision Transformer (ViT)

---


## Dataset

**NIH Chest X-ray14**
- 112.120 citra dari 30.805 pasien
- Download: https://nihcc.app.box.com/v/ChestXray-NIHCC
- Jumlah citra setelah cleaning: 105.039 gambar (7 target label + No Finding)


---

## Referensi

- Ahmed, S. et al. (2023). *Multi-Technique Chest X-Ray Analysis*. Diagnostics, 13(4), 814.
- Wang, X. et al. (2017). *ChestX-ray8: NIH Chest X-ray Dataset*. CVPR.
- Cohen, J.P. et al. (2022). *TorchXRayVision*. MIDL.
