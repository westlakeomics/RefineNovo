"""Training and testing functionality for the de novo peptide sequencing
model."""
import glob
import logging
import os
import glob
import yaml
import argparse
import random
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from datetime import timedelta
import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from torch_imputer.imputer import best_alignment

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

# import from project
from .model import Spec2Pep
from .iterable_dataset_online import create_iterable_dataset


logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 切换DDP端口号
os.environ["MASTER_PORT"] = "12345"

import warnings
warnings.filterwarnings("ignore")

torch.backends.cuda.matmul.allow_tf32 = False


def set_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    # 固定卷积算法
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    
def mkdir_p(dirs):
    """
    make a directory (dir) if it doesn't exist
    """
    if not os.path.exists(dirs):
        try:
            # 递归创建文件夹
            os.makedirs(dirs)
        except:
            pass

    return True, 'OK'
    

class PTModule(pl.LightningModule):
    """PTL wrapper for model."""

    def __init__(
            self,
            config: dict[str, Any],
            model: Spec2Pep,
            
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model

        # loss
        self.ctcloss = torch.nn.CTCLoss(blank=self.model.decoder.get_blank_idx(),
                                        zero_infinity=True)

        self._reset_test_metrics()


    def _forward_step(
        self,
        spectra: torch.Tensor,
        precursors: torch.Tensor,
        sequences: List[str],
        prev = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward learning step.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum, and axis 2 is essentially a 2-tuple specifying
            the m/z-intensity pair for each peak. These should be zero-padded,
            such that all of the spectra in the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.
        sequences : List[str] of length n_spectra
            The partial peptide sequences to predict.
        prev:  The partial peptide sequences

        Returns
        -------
        scores : torch.Tensor of shape (n_spectra, length, n_amino_acids)
            The individual amino acid scores for each prediction.
        tokens : torch.Tensor of shape (n_spectra, length)
            The predicted tokens for each spectrum.
        """
        spectra, spectra_mask = self.model.encoder(spectra, precursors)
        return self.model.decoder(sequences, precursors, spectra, spectra_mask, prev)
    


    def test_step(
            self, batch: tuple[Tensor, Tensor, Tensor, list[str] | Tensor, Tensor], *args: Any
    ) -> torch.Tensor:
        """Single validation step."""
        try:
            batch = next(iter(batch))
            spectra, precursors, peptides = batch
            spectra = spectra.to(self.device)
            precursors = precursors.to(self.device)

            glat_prev = None
            total_loss = torch.tensor([0]).to(self.device)
        
            #change the peak factor with-respect to epoch here
            peek_factor = max(0.93 - self.current_epoch * 0.01, 0.00 ) 
            
            #----curriculum learning sampling---
            with torch.no_grad():
                word_ins_out, tgt_tokens, _ = self._forward_step(spectra, precursors, peptides, glat_prev)
                nonpad_positions = tgt_tokens.ne(self.model.decoder.get_pad_idx())
                target_lens = (nonpad_positions).sum(1)
                pred_tokens = word_ins_out.argmax(-1)
                out_lprobs = F.log_softmax(word_ins_out, dim=-1)
                seq_lens = torch.full(size=(pred_tokens.size()[0],), fill_value=pred_tokens.size()[1]).to(self.device)
                best_aligns = best_alignment(out_lprobs.transpose(0, 1), 
                                             tgt_tokens, 
                                             seq_lens, 
                                             target_lens, 
                                             self.model.decoder.get_blank_idx(),
                                             zero_infinity=True)
                best_aligns_pad = torch.tensor([a for a in best_aligns], 
                                               device=word_ins_out.device)
                oracle_pos = (best_aligns_pad // 2).clip(max=tgt_tokens.shape[1] - 1)
                oracle = tgt_tokens.gather(-1, oracle_pos)
                oracle_empty = oracle.masked_fill(best_aligns_pad % 2 == 0, self.model.decoder.get_blank_idx())
                same_num = ((pred_tokens == oracle_empty)).sum(1)

                # 
                # keep_prob = 1.1
                keep_prob = 0.5
                # keep_prob = 1e-6
                keep_word_mask = (torch.rand(pred_tokens.shape, device=word_ins_out.device) < keep_prob).bool()
                glat_prev = oracle_empty.masked_fill(~keep_word_mask, self.model.decoder.get_mask_idx())


            for step in range(3):
                if step == 0:
                    sequences = glat_prev
                else:
                    pred, truth, _ = self._forward_step(spectra, precursors, peptides, glat_prev)

                    glat_prev = pred.argmax(-1)
                    sequences = glat_prev
                
                for j in range(sequences.shape[0]):
                    loop_pred_seq = self.model.decoder.detokenize(sequences[j])
                    loop_pred_seq = ''.join(loop_pred_seq)

                    self.training_cache.append(
                        (step, peptides[j], loop_pred_seq)
                    )

        except Exception as e:
            logging.info("test_step, error:", e)
        return torch.tensor(0.0)

    def on_test_epoch_end(self) -> None:
        """Log the validation metrics at the end of each epoch."""
        output_filepath = os.path.join(self.config['out_path'], f"test_keep05_pred.csv")

        if len(self.training_cache):
            df = pd.DataFrame(self.training_cache)
            df.to_csv(output_filepath, mode='a+', header=False, index=None)


    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Attempt to load config with checkpoint."""
        self.config = checkpoint["config"]

        
    def _reset_test_metrics(self) -> None:
        self.training_cache = []

        
def train(
        config: dict,
        model_path: str | None = None,
) -> None:
    """Training function."""

    mkdir_p(config["out_path"])    
    logging.info("Loading data")
    
    test_dl = create_iterable_dataset(logging, config, parse='test')

    logging.info(
        f"Data loaded: {len(test_dl):,} test DataLoader"
    )

    # Initialize the model.
    logger.info("Training from checkpoint...")
    model_filename = config["init_model_path"]
    if not os.path.isfile(model_filename):
        logger.error(
            "Could not find the model weights at file %s to continue "
            "training",
            model_filename,
        )
        raise FileNotFoundError(
            "Could not find the model weights to continue training")
    
    ctc_params = dict(model_path=None,  #to change
                    alpha=0, beta=0,
                    cutoff_top_n=100,
                    cutoff_prob= 1.0,
                    beam_width=config["n_beams"],
                    num_processes=4,
                    log_probs_input = False)
    
    model_params = dict(
        PMC_enable = config["PMC_enable"],
        mass_control_tol = config["mass_control_tol"],
        dim_model=config["dim_model"],
        n_head=config["n_head"],
        dim_feedforward=config["dim_feedforward"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        dim_intensity=config["dim_intensity"],
        max_length=config["max_length"],
        residues=config["residues"],
        max_charge=config["max_charge"],
        precursor_mass_tol=config["precursor_mass_tol"],
        isotope_error_range=config["isotope_error_range"],
        n_beams=config["n_beams"],
        tb_summarywriter=config["tb_summarywriter"],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        ctc_dic = ctc_params
    )

    model_filename = config["init_model_path"]
    if not os.path.isfile(model_filename):
        logger.error(
            "Could not find the model weights at file %s to continue "
            "training",
            model_filename,
        )
        raise FileNotFoundError(
            "Could not find the model weights to continue training")
    model = Spec2Pep().load_from_checkpoint(model_filename, **model_params)
    
    # Train on GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    model = model.to(device)

    # Update config
    logging.info(f"gpu num = {torch.cuda.device_count()}")

    strategy = DDPStrategy(gradient_as_bucket_view=True, find_unused_parameters=True)
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        strategy=strategy,
    )

    evaluate = PTModule(config, model)
    trainer.test(evaluate, dataloaders=test_dl)


if __name__ == "__main__":
    """Train the model."""
    logging.info("Initializing training.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="/storage/guotiannanLab/sunyingying/massNet-meta/xuanji_test_20250718/XuanjiNovo/denovo/eval.py")
    args = parser.parse_args()

    logging.info(f"load config:  {args.config_path}")

    with open(args.config_path) as f_in:
        config = yaml.safe_load(f_in)
    set_seeds(config['seed'])

    if config['init_model_path'] == '':
        model_path = None
    else:
        model_path = config['init_model_path']
    train(config, model_path)
