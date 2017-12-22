import numpy as np
import tensorflow as tf

from config import get_config
from utils import prepare_dirs_and_logger, save_config

def main(config):
    prepare_dirs_and_logger(config)
    save_config(config)

    if config.is_train:
        from trainer import Trainer
        if config.dataset == 'line':
            from data_line import BatchManager

        batch_manager = BatchManager(config)
        trainer = Trainer(config, batch_manager)
        trainer.train()
    else:
        if not config.load_pathnet or not config.load_overlapnet:
            raise Exception("[!] You should specify `load_path` to load a pretrained model")
            
        pass

if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
