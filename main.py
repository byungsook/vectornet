import numpy as np
import tensorflow as tf

from config import get_config
from utils import prepare_dirs_and_logger, save_config

def main(config):
    if config.archi == 'path':
        from trainer_path import Trainer
    elif config.archi == 'overlap':
        from trainer_overlap import Trainer
    else:
        raise Exception("[!] You should specify `archi` to load a trainer")

    if config.dataset == 'line':
        from data_line import BatchManager

    prepare_dirs_and_logger(config)

    batch_manager = BatchManager(config)
    trainer = Trainer(config, batch_manager)

    if config.is_train:
        save_config(config)
        trainer.train()
    else:
        if not config.load_path:
            raise Exception("[!] You should specify `load_path` to load a pretrained model")
        trainer.test()

if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
