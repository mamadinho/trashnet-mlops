import yaml

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

MODE = config['mode']
SEED = config['seed']
NUM_CLASSES = config['num_classes']
NUM_WORKERS = config['num_workers']
BATCH_SIZE = config['batch_size']
LR = config['lr']
ARCHITECTURE = config['architecture']
LABELS_NAME = config['labels_name']

FOLD_NUM = config['cross_validation']['fold_num']
EPOCHS_CROSSVAL = config['cross_validation']['epochs_crossval']
IMG_SAVE_DIR = config['cross_validation']['img_save_dir']

EPOCHS_FULL = config['full_training']['epochs_full']

USE_WANDB = config['logging']['use_wandb']
WANDB_PROJECT = config['logging']['wandb_project']
WANDB_DATASET = config['logging']['wandb_dataset']
