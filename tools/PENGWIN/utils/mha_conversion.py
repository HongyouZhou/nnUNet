try:
    import pyrootutils
    pyrootutils.setup_root(__file__, indicator="project-root", pythonpath=True)
except:
    pass

from challenge_pengwin.utils.utils import load_filenames, MedVol
from tqdmp import tqdmp
from os.path import join
from pathlib import Path
import numpy as np


def nifti2mha(load_dir, save_dir, processes):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    names = load_filenames(load_dir)
    tqdmp(nifti2mha_single, names, processes, desc="nifti2mha", load_dir=load_dir, save_dir=save_dir)


def mha2nifti(load_dir, save_dir, processes):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    names = load_filenames(load_dir)
    tqdmp(mha2nifti_single, names, processes, desc="mha2nifti", load_dir=load_dir, save_dir=save_dir)


def nifti2mha_single(name, load_dir, save_dir):
    image = MedVol(join(load_dir, f"{name}.nii.gz"))
    image.save(join(save_dir, f"{name}.mha"))


def mha2nifti_single(name, load_dir, save_dir):
    image = MedVol(join(load_dir, f"{name}.mha"))
    image.save(join(save_dir, f"{name}_0000.nii.gz"))


if __name__ == '__main__':
    load_dir = "/home/k539i/Documents/projects/challenge_pengwin/test/input/images/pelvic-fracture-ct"
    save_dir = "/home/k539i/Documents/projects/challenge_pengwin/test/output/image_nifti"
    processes = 6

    mha2nifti(load_dir, save_dir, processes)