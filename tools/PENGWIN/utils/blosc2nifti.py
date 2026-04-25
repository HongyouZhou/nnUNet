from challenge_pengwin.utils.utils import load_filenames, MedVol
from challenge_pengwin.border_core_conversion.instance2border_core import instance2border_core_fixed as instance2border_core
from tqdmp import tqdmp
from os.path import join
from pathlib import Path
import blosc2
import numpy as np


def blosc2nifti(load_dir, save_dir, processes):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    names = load_filenames(load_dir)
    tqdmp(blosc2nifti_single, names, processes, desc="blosc2nifti", load_dir=load_dir, save_dir=save_dir)


def blosc2nifti_single(name, load_dir, save_dir):
    seg = blosc2.load_array(urlpath=join(load_dir, f"{name}.b2nd"))
    MedVol(seg).save(join(save_dir, f"{name}.nii.gz"))


if __name__ == '__main__':
    load_dir = "/home/k539i/Documents/network_drives/cluster-data/preprocessed/nnUNet/nnUNet_preprocessed/Dataset2002_Pengwin_challenge_2024/nnUNetPlans_3d_fullres"
    save_dir = "/home/k539i/Documents/network_drives/cluster-data/preprocessed/nnUNet/nnUNet_preprocessed/Dataset2002_Pengwin_challenge_2024/nnUNetPlans_3d_fullres_tmp"
    processes = None

    blosc2nifti(load_dir, save_dir, processes)