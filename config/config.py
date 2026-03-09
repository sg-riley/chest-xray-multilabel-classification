import os


class Config:
    # Google Drive Base Path
    BASE_DIR    = '/content/drive/MyDrive/[DEV]Chest_X_Ray_Model'
    DATASET_DIR = f'{BASE_DIR}/DATASET'

    # Path Dataset
    IMAGES_DIR      = f'{DATASET_DIR}/raw_image'
    MASK_OUTPUT_DIR = f'{DATASET_DIR}/masked_image'

    # Path CSV
    CSV_RAW_PATH     = f'{DATASET_DIR}/Data_Entry_2017_v2020.csv'
    CSV_CLEANED_PATH = f'{DATASET_DIR}/metadata_cleaned.csv'
    CSV_MASK_PATH    = f'{DATASET_DIR}/metadata_with_mask.csv'
    CSV_SPLIT_PATH   = f'{DATASET_DIR}/metadata_split.csv'
    FAILED_LOG_PATH  = f'{DATASET_DIR}/failed_segmentation.csv'

    # Path Output Model & Log
    MODEL_OUTPUT_DIR = f'{BASE_DIR}/MODELS'
    LOG_DIR          = f'{BASE_DIR}/LOGS'
    FIGURES_DIR      = f'{BASE_DIR}/FIGURES'

    # Label Target
    TARGET_LABELS = [
        'Atelectasis',  
        'Effusion',      
        'Fibrosis',     
        'Infiltration', 
        'Consolidation', 
        'Mass',          
        'Nodule',     
    ]
    N_LABELS = len(TARGET_LABELS)

    # Segmentasi
    MASK_THRESHOLD       = 0.5   
    MORPH_KERNEL_SIZE    = 7     
    LOW_COVERAGE_THRESH  = 5.0   # % — batas minimum mask
    HIGH_COVERAGE_THRESH = 70.0  # % — batas maksimum mask