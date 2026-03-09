"""
config.py
─────────
Semua konstanta, path, dan hyperparameter penelitian.
Satu-satunya file yang perlu diubah saat berpindah environment.
"""

import os


class Config:
    # ── Google Drive Base Path ────────────────────────────────────────────────
    # Sesuaikan BASE_DIR dengan lokasi project di Google Drive kamu
    BASE_DIR    = '/content/drive/MyDrive/[DEV]Chest_X_Ray_Model'
    DATASET_DIR = f'{BASE_DIR}/DATASET'

    # ── Path Dataset ──────────────────────────────────────────────────────────
    # Folder raw image NIH — berisi subfolder images_001/images/, dst.
    IMAGES_DIR      = f'{DATASET_DIR}/raw_image'
    MASK_OUTPUT_DIR = f'{DATASET_DIR}/masks'

    # ── Path CSV ──────────────────────────────────────────────────────────────
    CSV_RAW_PATH     = f'{DATASET_DIR}/Data_Entry_2017_v2020.csv'
    CSV_CLEANED_PATH = f'{DATASET_DIR}/metadata_cleaned.csv'
    CSV_MASK_PATH    = f'{DATASET_DIR}/metadata_with_mask.csv'
    CSV_SPLIT_PATH   = f'{DATASET_DIR}/metadata_split.csv'
    FAILED_LOG_PATH  = f'{DATASET_DIR}/failed_segmentation.csv'

    # ── Path Output Model & Log ───────────────────────────────────────────────
    MODEL_OUTPUT_DIR = f'{BASE_DIR}/models'
    LOG_DIR          = f'{BASE_DIR}/logs'
    FIGURES_DIR      = f'{BASE_DIR}/figures'

    # ── Label Target (7 label sesuai proposal) ────────────────────────────────
    TARGET_LABELS = [
        'Atelectasis',   # Atelektasis
        'Effusion',      # Efusi Pleura
        'Fibrosis',      # Fibrosis
        'Infiltration',  # Infiltrat
        'Consolidation', # Konsolidasi
        'Mass',          # Massa
        'Nodule',        # Nodul
    ]
    N_LABELS = len(TARGET_LABELS)

    # Label mapping TB / Pneumonia (sesuai justifikasi klinis Tabel 2.3 proposal)
    TB_LABELS        = ['Fibrosis', 'Infiltration', 'Nodule', 'Mass']
    PNEUMONIA_LABELS = ['Consolidation', 'Effusion', 'Atelectasis']

    # ── Segmentasi ────────────────────────────────────────────────────────────
    MASK_THRESHOLD       = 0.5   # threshold probabilitas → binary mask
    MORPH_KERNEL_SIZE    = 7     # ukuran kernel morfologi cleanup
    LOW_COVERAGE_THRESH  = 5.0   # % — mask dianggap terlalu kecil
    HIGH_COVERAGE_THRESH = 70.0  # % — mask dianggap terlalu besar

    # ── Preprocessing & Augmentasi ────────────────────────────────────────────
    IMAGE_SIZE        = 224      # input size CNN (ResNet/DenseNet) & ViT
    RANDOM_SEED       = 42

    # Parameter augmentasi — konservatif untuk X-ray klinis
    AUG_ROTATION      = 10       # derajat — tidak terlalu ekstrem
    AUG_WIDTH_SHIFT   = 0.05
    AUG_HEIGHT_SHIFT  = 0.05
    AUG_ZOOM          = 0.05
    AUG_HORIZONTAL_FLIP = True
    AUG_VERTICAL_FLIP   = False  # tidak realistis secara klinis

    # ── Handcrafted Feature Extraction ───────────────────────────────────────
    # LBP
    LBP_RADIUS  = 1
    LBP_N_POINTS = 8             # 8 * radius

    # GLCM
    GLCM_DISTANCES = [1]
    GLCM_ANGLES    = [0, 0.785, 1.571, 2.356]  # 0°, 45°, 90°, 135° dalam radian
    GLCM_PROPERTIES = ['contrast', 'dissimilarity', 'homogeneity',
                        'energy', 'correlation']

    # DWT
    DWT_WAVELET = 'db1'
    DWT_LEVEL   = 1

    # ── PCA ───────────────────────────────────────────────────────────────────
    PCA_VARIANCE_RATIO = 0.95    # pertahankan 95% variance

    # ── Split Dataset ─────────────────────────────────────────────────────────
    TRAIN_RATIO = 0.80
    VAL_RATIO   = 0.10
    TEST_RATIO  = 0.10
    # Split dilakukan per Patient ID untuk menghindari data leakage

    # ── Training ──────────────────────────────────────────────────────────────
    BATCH_SIZE    = 32
    EPOCHS        = 100
    LEARNING_RATE = 1e-3
    LR_PATIENCE   = 5            # epoch sebelum ReduceLROnPlateau
    LR_FACTOR     = 0.5
    EARLY_STOP_PATIENCE = 10

    # ── Model Architecture (ANN Classifier) ───────────────────────────────────
    ANN_HIDDEN_LAYERS = [512, 256, 128]
    ANN_DROPOUT_RATES = [0.4, 0.3, 0.2]
    ANN_ACTIVATION    = 'relu'
    OUTPUT_ACTIVATION = 'sigmoid'  # multi-label → sigmoid per label
    LOSS_FUNCTION     = 'binary_crossentropy'

    # ── Vision Transformer ────────────────────────────────────────────────────
    VIT_MODEL_NAME  = 'google/vit-base-patch16-224-in21k'
    VIT_LR          = 1e-5       # fine-tuning pakai LR lebih kecil
    VIT_WEIGHT_DECAY = 0.01

    # ── Evaluasi ──────────────────────────────────────────────────────────────
    CLASSIFICATION_THRESHOLD = 0.5   # threshold prediksi → label positif
    # Target minimum performa (sesuai proposal)
    TARGET_AUC         = 0.85
    TARGET_ACCURACY    = 0.80
    TARGET_SENSITIVITY = 0.80
    TARGET_SPECIFICITY = 0.80
    TARGET_PRECISION   = 0.80

    # ── Device ────────────────────────────────────────────────────────────────
    DEVICE = 'cuda'   # otomatis fallback ke 'cpu' jika tidak ada GPU

    # ── LLM (integrasi laporan teks) ──────────────────────────────────────────
    LLM_MODEL_NAME = 'google/flan-t5-base'
    LLM_MAX_TOKENS = 256

    def __init__(self):
        """Buat semua direktori output jika belum ada."""
        for path in [self.MASK_OUTPUT_DIR, self.MODEL_OUTPUT_DIR,
                     self.LOG_DIR, self.FIGURES_DIR]:
            os.makedirs(path, exist_ok=True)

    def __repr__(self):
        return (f"Config(\n"
                f"  BASE_DIR={self.BASE_DIR}\n"
                f"  TARGET_LABELS={self.TARGET_LABELS}\n"
                f"  IMAGE_SIZE={self.IMAGE_SIZE}\n"
                f"  BATCH_SIZE={self.BATCH_SIZE}\n"
                f"  EPOCHS={self.EPOCHS}\n"
                f")")
