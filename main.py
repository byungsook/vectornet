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
        elif config.dataset == 'ch':
            from data_ch import BatchManager
        elif config.dataset == 'kanji':
            from data_kanji import BatchManager
        elif config.dataset == 'baseball' or\
             config.dataset == 'cat':
            from data_qdraw import BatchManager

        batch_manager = BatchManager(config)
        trainer = Trainer(config, batch_manager)
        trainer.train()
    else:
        from tester import Tester
        if config.dataset == 'line':
            from data_line import BatchManager
        elif config.dataset == 'ch':
            from data_ch import BatchManager
        elif config.dataset == 'kanji':
            from data_kanji import BatchManager
        elif config.dataset == 'baseball' or\
             config.dataset == 'cat':
            from data_qdraw import BatchManager
        
        batch_manager = BatchManager(config)
        tester = Tester(config, batch_manager)
        tester.test()

if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
