"""
src/dataset.py
──────────────
Fungsi untuk load, filter, dan split dataset NIH Chest X-ray14.

Fungsi utama:
    load_metadata()            → load CSV metadata
    build_image_index()        → scan folder nested → dict {filename: full_path}
    filter_available_images()  → filter baris CSV yang file gambarnya ada
    make_onehot_labels()       → buat kolom one-hot per label target
    split_by_patient()         → train/val/test split berdasarkan Patient ID
    get_class_weights()        → hitung pos_weight untuk weighted BCE loss
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Optional


# ── Load Metadata ─────────────────────────────────────────────────────────────

def load_metadata(csv_path: str) -> pd.DataFrame:
    """
    Load CSV metadata NIH Chest X-ray14.

    Args:
        csv_path: path ke metadata_cleaned.csv atau metadata_with_mask.csv

    Returns:
        DataFrame dengan kolom lengkap
    """
    df = pd.read_csv(csv_path)
    print(f"✅ Metadata loaded: {len(df):,} baris | {df.columns.tolist()}")
    return df


# ── Build Image Index ─────────────────────────────────────────────────────────

def build_image_index(images_dir: str, ext: str = '.png') -> Dict[str, str]:
    """
    Scan folder images secara rekursif dan bangun index:
        {nama_file.png → path_lengkap}

    Mendukung struktur nested seperti:
        raw_image/images_001/images/00000001_000.png
        raw_image/images_002/images/00000002_000.png

    Args:
        images_dir : root folder gambar
        ext        : ekstensi file yang dicari (default: .png)

    Returns:
        dict mapping filename → full path
    """
    print(f"🔍 Scanning folder: {images_dir}")
    index = {}
    for root, _, files in os.walk(images_dir):
        for fname in files:
            if fname.endswith(ext):
                index[fname] = os.path.join(root, fname)

    print(f"✅ Image index selesai: {len(index):,} gambar ditemukan")
    return index


# ── Filter Available Images ───────────────────────────────────────────────────

def filter_available_images(
    df: pd.DataFrame,
    image_index: Dict[str, str],
    image_col: str = 'Image Index'
) -> pd.DataFrame:
    """
    Filter DataFrame — hanya baris yang file gambarnya benar-benar ada.

    Args:
        df          : DataFrame metadata
        image_index : dict hasil build_image_index()
        image_col   : nama kolom yang berisi nama file gambar

    Returns:
        DataFrame yang sudah difilter + kolom 'image_path' (full path)
    """
    df = df.copy()
    df['image_path'] = df[image_col].map(image_index)

    n_total     = len(df)
    df_filtered = df[df['image_path'].notna()].copy()
    n_found     = len(df_filtered)
    n_missing   = n_total - n_found

    print(f"✅ Filter selesai:")
    print(f"   Total di CSV        : {n_total:,}")
    print(f"   File tersedia       : {n_found:,}")
    print(f"   File tidak ditemukan: {n_missing:,}")

    return df_filtered.reset_index(drop=True)


# ── One-Hot Labels ────────────────────────────────────────────────────────────

def make_onehot_labels(
    df: pd.DataFrame,
    target_labels: list,
    finding_col: str = 'Finding Labels'
) -> pd.DataFrame:
    """
    Buat kolom one-hot per label target dari kolom 'Finding Labels'.

    Jika kolom sudah ada (dari notebook cleaning), fungsi ini akan skip
    dan langsung return DataFrame yang sudah ada.

    Args:
        df            : DataFrame metadata
        target_labels : list label target, misal ['Atelectasis', 'Effusion', ...]
        finding_col   : nama kolom berisi label asli (pipe-separated string)

    Returns:
        DataFrame + kolom one-hot per label target + kolom 'No_Finding'
    """
    df = df.copy()

    # Cek apakah kolom sudah ada
    existing = [l for l in target_labels if l in df.columns]
    if len(existing) == len(target_labels):
        print(f"✅ Kolom one-hot sudah ada, skip pembuatan ulang")
        return df

    print(f"⚙️  Membuat kolom one-hot untuk {len(target_labels)} label...")

    for label in target_labels:
        df[label] = df[finding_col].str.contains(
            label, case=True, regex=False
        ).astype(int)

    if 'No_Finding' not in df.columns:
        df['No_Finding'] = (df[finding_col] == 'No Finding').astype(int)

    df['label_count'] = df[target_labels].sum(axis=1)

    print(f"✅ One-hot labels selesai")
    for label in target_labels:
        print(f"   {label:20s}: {df[label].sum():,}")

    return df


# ── Split by Patient ID ───────────────────────────────────────────────────────

def split_by_patient(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    val_ratio: float   = 0.10,
    test_ratio: float  = 0.10,
    random_seed: int   = 42,
    patient_col: str   = 'Patient ID'
) -> pd.DataFrame:
    """
    Split dataset menjadi train/val/test berdasarkan Patient ID.

    PENTING: Split dilakukan per PASIEN (bukan per gambar) untuk
    mencegah data leakage — pasien yang sama tidak boleh ada di
    train sekaligus di val/test.

    Args:
        df          : DataFrame metadata
        train_ratio : proporsi data training
        val_ratio   : proporsi data validasi
        test_ratio  : proporsi data testing
        random_seed : seed untuk reproduktibilitas
        patient_col : nama kolom Patient ID

    Returns:
        DataFrame + kolom 'split' berisi 'train' / 'val' / 'test'
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train + val + test harus = 1.0"

    df = df.copy()

    # Ambil unique Patient ID
    all_patients = df[patient_col].unique()
    n_patients   = len(all_patients)

    print(f"⚙️  Split dataset:")
    print(f"   Total pasien unik : {n_patients:,}")

    # Split pasien: train | (val + test)
    test_val_ratio = val_ratio + test_ratio
    patients_train, patients_temp = train_test_split(
        all_patients,
        test_size=test_val_ratio,
        random_state=random_seed
    )

    # Split (val + test) → val | test
    val_from_temp = val_ratio / test_val_ratio
    patients_val, patients_test = train_test_split(
        patients_temp,
        test_size=(1 - val_from_temp),
        random_state=random_seed
    )

    # Assign split ke DataFrame
    patient_to_split = {}
    for p in patients_train: patient_to_split[p] = 'train'
    for p in patients_val:   patient_to_split[p] = 'val'
    for p in patients_test:  patient_to_split[p] = 'test'

    df['split'] = df[patient_col].map(patient_to_split)

    # Ringkasan
    for split in ['train', 'val', 'test']:
        n_img = (df['split'] == split).sum()
        n_pat = df[df['split'] == split][patient_col].nunique()
        print(f"   {split:5s} : {n_img:6,} gambar | {n_pat:5,} pasien")

    return df


# ── Class Weights ─────────────────────────────────────────────────────────────

def get_class_weights(
    df: pd.DataFrame,
    target_labels: list,
    split: str = 'train'
) -> Dict[str, float]:
    """
    Hitung pos_weight per label untuk weighted Binary Cross-Entropy loss.

    Formula: pos_weight = n_negative / n_positive
    Semakin langka label positif, semakin besar bobotnya.

    Args:
        df            : DataFrame dengan kolom split & one-hot labels
        target_labels : list label target
        split         : split yang digunakan untuk hitung weight ('train')

    Returns:
        dict {label: pos_weight}
    """
    if 'split' in df.columns:
        df_train = df[df['split'] == split]
    else:
        df_train = df

    weights = {}
    print(f"⚙️  Class weights (pos_weight = n_neg / n_pos):")
    for label in target_labels:
        n_pos = df_train[label].sum()
        n_neg = len(df_train) - n_pos
        w     = n_neg / n_pos if n_pos > 0 else 1.0
        weights[label] = round(w, 4)
        print(f"   {label:20s}: pos={n_pos:5,} neg={n_neg:6,} weight={w:.2f}")

    return weights


# ── Summary ───────────────────────────────────────────────────────────────────

def dataset_summary(df: pd.DataFrame, target_labels: list) -> None:
    """Print ringkasan distribusi label dataset."""
    print("=" * 55)
    print("📊 DATASET SUMMARY")
    print("=" * 55)
    print(f"  Total gambar       : {len(df):,}")

    if 'Patient ID' in df.columns:
        print(f"  Unique pasien      : {df['Patient ID'].nunique():,}")

    if 'split' in df.columns:
        for split in ['train', 'val', 'test']:
            n = (df['split'] == split).sum()
            print(f"  {split:5s}              : {n:,}")

    print()
    print("  Label distribution:")
    for label in target_labels:
        if label in df.columns:
            n   = df[label].sum()
            pct = n / len(df) * 100
            print(f"    {label:20s}: {n:6,}  ({pct:.1f}%)")

    if 'No_Finding' in df.columns:
        n   = df['No_Finding'].sum()
        pct = n / len(df) * 100
        print(f"    {'No Finding':20s}: {n:6,}  ({pct:.1f}%)")

    print("=" * 55)
