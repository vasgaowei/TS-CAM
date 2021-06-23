import os
import torch.backends.cudnn as cudnn

from utils import fix_random_seed, backup_codes, rm


def prepare_env(cfg):
    # fix random seed
    fix_random_seed(cfg.BASIC.SEED)
    # cudnn
    cudnn.benchmark = cfg.CUDNN.BENCHMARK  # Benchmark will impove the speed
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC  #
    cudnn.enabled = cfg.CUDNN.ENABLE  # Enables benchmark mode in cudnn, to enable the inbuilt cudnn auto-tuner

    # backup codes
    if cfg.BASIC.BACKUP_CODES:
        backup_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'backup')
        rm(backup_dir)
        backup_codes(cfg.BASIC.ROOT_DIR, backup_dir, cfg.BASIC.BACKUP_LIST)


