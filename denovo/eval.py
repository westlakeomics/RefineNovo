"""The command line entry point for XuanjiNovo."""
import datetime
import functools
import logging
import os
import re
import shutil
import sys

import argparse
from typing import Optional, Tuple

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import yaml


from . import eval_train
from . import eval_predict


logger = logging.getLogger("XuanjiNovo")
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    logging.info("Initializing training.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="/storage/guotiannanLab/sunyingying/massNet-meta/xuanji_test_20250721/XuanjiNovo/yaml/eval.yaml")
    parser.add_argument("--phase", default="train")
    args = parser.parse_args()

    logging.info(f"load config:  {args.config_path}")

    with open(args.config_path) as f_in:
        config = yaml.safe_load(f_in)
    eval_train.set_seeds(config['seed'])

    if config['init_model_path'] == '':
        model_path = None
    else:
        model_path = config['init_model_path']
 
    logger.info("Evaluate a trained XuanjiNovo model.")
    if args.phase == 'train':
        eval_train.train(config, model_path)
    else:
        eval_predict.train(config, model_path)
    

