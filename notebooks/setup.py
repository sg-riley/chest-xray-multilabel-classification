import os, sys
from google.colab import drive

def setup_colab():
    drive.mount('/content/drive', force_remount=True)

    REPO_URL  = 'https://github.com/sg-riley/chest-xray-multilabel-classification.git'
    REPO_NAME = 'chest-xray-multilabel-classification'
    REPO_PATH = f'/content/{REPO_NAME}'

    if os.path.exists(REPO_PATH):
        print("Repo exist — pulling update...")
        os.chdir(REPO_PATH)
        os.system('git pull')
    else:
        print("Cloning repo...")
        os.system(f'git clone {REPO_URL}')

    # Add repo ke sys.path dan set working directory
    sys.path.insert(0, REPO_PATH)
    os.chdir(REPO_PATH)
    print(f"\n Working directory: {os.getcwd()}")

setup_colab()