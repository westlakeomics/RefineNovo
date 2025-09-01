from __future__ import annotations

import re
import os

import numpy as np
import pandas as pd
import spectrum_utils.spectrum as sus

import torch
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset
import torch.nn.functional as F

# from constants import PROTON_MASS_AMU
PROTON_MASS_AMU = 1.007276


def safe_to_tensor(array):
    try:
        return torch.tensor(array.copy() if isinstance(array, np.ndarray) else array)
    except Exception as e:
        raise ValueError(f"Failed to convert to tensor: {e}")

class SpectrumDataset(Dataset):
    """Spectrum dataset class supporting `.ipc` and `.csv`."""

    def __init__(
        self,
        df: pd.DataFrame | pl.DataFrame,
        n_peaks: int = 200,
        max_length=50,
        min_mz: float = 50.0,
        max_mz: float = 2500.0,
        min_intensity: float = 0.01,
        remove_precursor_tol: float = 2.0,

    ) -> None:
        super().__init__()
        self.df = df
        self.n_peaks = n_peaks
        self.max_length = max_length
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.remove_precursor_tol = remove_precursor_tol
        self.min_intensity = min_intensity

        if isinstance(df, pd.DataFrame):
            self.data_type = "pd"
        elif isinstance(df, pl.DataFrame):
            self.data_type = "pl"
        else:
            raise Exception(f"Unsupported data type {type(df)}")

    def __len__(self) -> int:
        return int(self.df.shape[0])

    def __getitem__(self, idx: int) -> tuple[Tensor, float, int, Tensor | list[str]]:
        if self.data_type == "pl":
            mz_array = torch.Tensor(self.df[idx, "mz_array"].to_list())
            int_array = torch.Tensor(self.df[idx, "intensity_array"].to_list())
            precursor_mz = self.df[idx, "precursor_mz"]
            precursor_charge = self.df[idx, "raw_charge"]
            peptide = self.df[idx, "peptide_chimerys"]
            pep_type = self.df[idx, "pep_type"]
        elif self.data_type == "pd":
            row = self.df.iloc[idx]
            mz_array = safe_to_tensor(row["mz_array"])
            int_array = safe_to_tensor(row["intensity_array"])
            precursor_mz = row["precursor_mz"]
            precursor_charge = row["raw_charge"]
            peptide = row["peptide_chimerys"]
            pep_type = row["pep_type"]

        spectrum = self._process_peaks(mz_array, int_array, precursor_mz, precursor_charge)
        return spectrum, precursor_mz, precursor_charge, peptide, pep_type

    def _process_peaks(
        self,
        mz_array: np.ndarray,
        int_array: np.ndarray,
        precursor_mz: float,
        precursor_charge: int,
    ) -> torch.Tensor:
        """
        Preprocess the spectrum by removing noise peaks and scaling the peak
        intensities.

        Parameters
        ----------
        mz_array : numpy.ndarray of shape (n_peaks,)
            The spectrum peak m/z values.
        int_array : numpy.ndarray of shape (n_peaks,)
            The spectrum peak intensity values.
        precursor_mz : float
            The precursor m/z.
        precursor_charge : int
            The precursor charge.

        Returns
        -------
        torch.Tensor of shape (n_peaks, 2)
            A tensor of the spectrum with the m/z and intensity peak values.
        """
        spectrum = sus.MsmsSpectrum(
            "",
            precursor_mz,
            precursor_charge,
            mz_array.numpy().astype(np.float64),
            int_array.numpy().astype(np.float32),
        )
        try:
            spectrum.set_mz_range(self.min_mz, self.max_mz)
            if len(spectrum.mz) == 0:
                raise ValueError
            spectrum.remove_precursor_peak(self.remove_precursor_tol, "Da")
            if len(spectrum.mz) == 0:
                raise ValueError
            spectrum.filter_intensity(self.min_intensity, self.n_peaks)
            if len(spectrum.mz) == 0:
                raise ValueError
            spectrum.scale_intensity("root", 1)
            intensities = spectrum.intensity / np.linalg.norm(
                spectrum.intensity
            )
            return torch.tensor(np.array([spectrum.mz, intensities])).T.float()
        except ValueError:
            # Replace invalid spectra by a dummy spectrum.
            return torch.tensor([[0, 1]]).float()   


def padding(data):
    ll = torch.tensor([x.shape[0] for x in data], dtype=torch.long)
    data = nn.utils.rnn.pad_sequence(data, batch_first=True)
    return data


def collate_batch(
    batch: list[tuple[Tensor, float, int, str]]
) -> tuple[Tensor, Tensor, list]:
    """Collate batch of samples."""
    spectra, precursor_mzs, precursor_charges, peptides, pep_types = zip(*batch)

    # Pad spectra
    spectra = padding(spectra)

    # precursors
    precursor_mzs = torch.tensor(precursor_mzs)
    precursor_charges = torch.tensor(precursor_charges)
    precursor_masses = (precursor_mzs - PROTON_MASS_AMU) * precursor_charges
    
    pep_types = torch.tensor(pep_types)
    precursors = torch.vstack([precursor_masses, precursor_charges, pep_types]).T.float()

    # return numpy
    return spectra.numpy(), precursors.numpy(), peptides


def mkdir_p(dirs, delete=True):
    """
    make a directory (dir) if it doesn't exist
    """    
    # 如果文件夹不存在，则递归新建
    if not os.path.exists(dirs):
        try:
            # 递归创建文件夹
            os.makedirs(dirs)
        except:
            pass

    return True, 'OK'
