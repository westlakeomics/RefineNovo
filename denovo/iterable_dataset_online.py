from __future__ import annotations

import torch
import torch.utils.data as pt_data
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, ConcatDataset
from torch import Tensor
from torch.utils.data import random_split

import os
import numpy as np
import pandas as pd

import pyarrow.parquet as pq
import pickle
import random
from random import sample
import math
import re
from collections import defaultdict
import glob

# import from project
from .dataset import collate_batch, SpectrumDataset

class Dataset(pt_data.Dataset):
    def __init__(self):
        self.spectra = None
        self.precursors = None
        self.peptides = None

    def __getitem__(self, idx):
        return_dict = {"spectra": self.spectra[idx],
                       "precursors": self.precursors[idx],
                       "peptides": self.peptides[idx]}
        return return_dict

    def __len__(self):
        return len(self.peptides)

    def fit_scale(self):
        pass

    
def dict2pt(
    spectra,
    precursors, 
    peptides
) -> tuple[str, Tensor, Tensor, str]:
    """dict to pt_data.Dataset."""
    # gen dataset
    data_set = Dataset()
    data_set.spectra = np.nan_to_num(spectra)
    data_set.precursors = np.nan_to_num(precursors)
    data_set.peptides = peptides
    return data_set


def shuffle_file_list(file_list, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    idx = torch.randperm(len(file_list), generator=generator).numpy()
    file_list = (np.array(file_list)[idx]).tolist()
    return file_list


def collate_numpy_batch(batch_data):
    """Collate batch of samples."""
    one_batch_spectra = torch.tensor(np.array([batch["spectra"] for batch in batch_data]), dtype=torch.float)
    one_batch_precursors = torch.tensor(np.array([batch["precursors"] for batch in batch_data]), dtype=torch.float)
    one_batch_peptides = [batch["peptides"] for batch in batch_data]

    return one_batch_spectra, one_batch_precursors, one_batch_peptides


def create_iterable_dataset(logging,
                            config,
                            parse='train',
                            seed=123):
    """
    Note: If you want to load all data in the memory, please set "read_part" to False.
    Args:
        :param logging: out logging.
        :param config: data from the yaml file.
        :param buffer_size: An integer. the size of file_name buffer.
    :return:
    """
    # update gpu_num
    gpu_num = torch.cuda.device_count() if torch.cuda.is_available() else 1
    logging.info(f"******************gpu_num: {gpu_num};**********")

    
    # 验证阶段
    if ';' in config['test_path']:
        total_test_path = config['test_path'].split(';')
        data_file_list = []
        for test_path in total_test_path:
            test_part_file_list = glob.glob(f'{test_path}/*.parquet')

            if len(test_part_file_list) > 0:
                data_file_list.extend(test_part_file_list)
        logging.info(f"******************{parse} {config['test_path']}, total loaded: {len(data_file_list)};**********")
    elif os.path.isdir(config['test_path']):
        data_file_list = glob.glob(f"{config['test_path']}/*parquet")
        logging.info(f"******************{parse} {config['test_path']}, loaded: {len(data_file_list)};**********")
    else:
        data_file_list = [config['test_path']]
        
    test_dl = IterableDiartDataset(data_file_list,
                                    config,
                                    gpu_num=gpu_num,
                                    shuffle=False,
                                    batch_size=config['predict_batch_size'])
    logging.info(f"{len(test_dl) * config['predict_batch_size']:,} test samples")
    return test_dl


class IterableDiartDataset(IterableDataset):
    """
    Custom dataset class for dataset in order to use efficient
    dataloader tool provided by PyTorch.
    """

    def __init__(self,
                 file_list: list,
                 config,
                 gpu_num=1,
                 shuffle=False,
                 batch_size=64,
                 **kwargs):
        super(IterableDiartDataset).__init__()
        # 文件列表
        self.file_list = file_list
        self.config = config
        
        self.gpu_num = gpu_num
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.seed = config['seed']

    def parse_file(self, file_name):
        # 加载pkl
        df = pq.read_table(file_name).to_pandas()

        # process_peaks
        ds = SpectrumDataset(
            df,
            self.config['n_peaks'],
            self.config['max_length'],
            self.config['min_mz'], 
            self.config['max_mz'],
            self.config['min_intensity'],
            self.config['remove_precursor_tol']
        )
        spectra, precursors, peptides = collate_batch(ds)
        
        dpt = dict2pt(spectra, precursors, peptides)
        dl = DataLoader(dpt,
                        batch_size=self.batch_size,
                        shuffle=self.shuffle,
                        collate_fn=collate_numpy_batch,
                        num_workers=2,
                        drop_last=False,
                        pin_memory=True)
        return dl

    def file_mapper(self, file_list):
        idx = 0
        file_num = len(file_list)
        while idx < file_num:
            yield self.parse_file(file_list[idx])
            idx += 1

    def __iter__(self):
        # 单卡模式下可手动切片
        if self.gpu_num > 1:
            rank = int(os.environ.get('LOCAL_RANK', 0))
            file_itr = self.file_list[rank::self.gpu_num]
        else:
            file_itr = self.file_list
        return self.file_mapper(file_itr)

    def __len__(self):
        if self.gpu_num > 1:
            return math.ceil(len(self.file_list) / self.gpu_num)
        else:
            return len(self.file_list)
        
    def set_epoch(self, epoch):
        self.epoch = epoch