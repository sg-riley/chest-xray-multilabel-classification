import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Optional


# Load Metadata ===================================================================

def load_metadata(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"Metadata loaded: {len(df):,} baris | {df.columns.tolist()}")
    return df

# Cleaning Metadata ===============================================================
def _has_any_target_label(finding_str: str, targets: list) -> bool:
    labels = [l.strip() for l in finding_str.split('|')]
    return any(label in targets for label in labels)


def clean_metadata(df, target_labels, finding_col='Finding Labels'):
    print("=" * 60)
    print("PROSES CLEANING METADATA")
    print("=" * 60)
    print(f"   Input: {len(df):,} baris")

    mask_disease    = df[finding_col].apply(
        lambda x: _has_any_target_label(x, target_labels))
    mask_no_finding = df[finding_col] == 'No Finding'
    mask_keep       = mask_disease | mask_no_finding

    print(f"   Total DISIMPAN                 : {mask_keep.sum():>7,}")
    print(f"   Total DIBUANG                  : {(~mask_keep).sum():>7,}")

    df_cleaned = df[mask_keep].copy().reset_index(drop=True)

    for label in target_labels:
        df_cleaned[label] = df_cleaned[finding_col].str.contains(
            label, case=True, regex=False).astype('int8')

    df_cleaned['No_Finding'] = (
        df_cleaned[finding_col] == 'No Finding').astype('int8')
    df_cleaned['label_count'] = (
        df_cleaned[target_labels].sum(axis=1).astype('int8'))

    n_multilabel = (df_cleaned['label_count'] > 1).sum()
    print(f"\n   Output          : {len(df_cleaned):,} baris")
    print(f"   Unique patients : {df_cleaned['Patient ID'].nunique():,}")
    print(f"   Multi-label     : {n_multilabel:,} gambar")

    return df_cleaned

# Build Image Index =========================================================

def _has_any_target_label(finding_str: str, targets: list) -> bool:
    labels = [l.strip() for l in finding_str.split('|')]
    return any(label in targets for label in labels)

def build_image_index(images_dir: str, ext: str = '.png') -> Dict[str, str]:
    print(f"  Scanning folder: {images_dir}")
    index = {}
    for root, _, files in os.walk(images_dir):
        for fname in files:
            if fname.endswith(ext):
                index[fname] = os.path.join(root, fname)

    print(f" Image index selesai: {len(index):,} gambar ditemukan")
    return index

