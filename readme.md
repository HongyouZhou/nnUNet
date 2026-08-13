# nnU-Net

nnU-Net is a semantic segmentation framework that automatically adapts its pipeline to a dataset. It analyzes the training data, creates a dataset fingerprint, configures suitable U-Net variants, and provides an end-to-end workflow from preprocessing to training, model selection, and inference.

It is primarily designed for supervised biomedical image segmentation, but it also works well as a strong baseline and development framework for researchers working on new segmentation methods.

## REPAIR service inference contract

For a single-case service invocation, use `inference_dev.py -i <FILE_UPLOAD>`.
Bone-name post-processing masks must be available in the case's
`postprocessing/`, `totalsegmentator/`, or `bone_masks/` directory (or through
the deployment-only `BONE_MASKS_DIR` setting). The results are published as
`<FILE_UPLOAD>/segmentations/*.nii.gz` using the same convention as
`all_komo_cases.zip`, for example `tibia_L.nii.gz`,
`tibia_L_fragment_1.nii.gz`, and `fibula_L.nii.gz`.

An aggregate `<FILE_UPLOAD>/seg.nii.gz` is also written for consumers that
only accept one NIfTI. It uses bone-aware label blocks, irrespective of
whether the uploaded CT is called `ct.nii.gz`, has an nnU-Net `_0000` suffix,
or uses an operation UUID. Repositioning consumes the named masks; Landmarks
may consume the aggregate file.

Laterality is required and can be supplied with `--side L/R` (also accepted as
`--SIDE`), the deployment `SIDE` environment variable, or `config.json`. The
same side-aware names are embedded in the `seg.nii.gz` NIfTI metadata, so label
1 and `segmentations/tibia_L.nii.gz`, for example, cannot drift apart.

The public output contains only postprocessed, bone-named fragment
segmentations. ABB-C labels and the anonymous instance map are internal
intermediate representations and are never published as pipeline artifacts.

If you are looking for nnU-Net v1, use the [v1 branch](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1). If you are migrating from v1, start with the [TLDR migration guide](documentation/tldr_migration_guide_from_v1.md).

![nnU-Net overview](documentation/assets/nnU-Net_overview.png)

## Start Here

- First-time setup: [Installation and setup](documentation/getting-started/installation-and-setup.md)
- First run on your own data: [Getting Started](documentation/getting-started/README.md)
- Task-oriented docs: [How-to Guides](documentation/how-to/README.md)
- Formats, commands, and configuration details: [Reference](documentation/reference/README.md)
- Concepts and rationale: [Explanation](documentation/explanation/README.md)

## Quick Install

Install PyTorch for your hardware first, then install nnU-Net:

```bash
pip install nnunetv2
```

For the full setup, including `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results`, see [Installation and setup](documentation/getting-started/installation-and-setup.md).

## Documentation

Start with the [documentation home](documentation/README.md).

Useful entry points:

- New users: [Getting Started](documentation/getting-started/README.md)
- Dataset preparation: [Prepare a dataset](documentation/how-to/prepare-a-dataset.md)
- Training workflow: [Train models](documentation/how-to/train-models.md)
- Inference workflow: [Run inference](documentation/how-to/run-inference.md)
- Recommended residual encoder presets: [Residual Encoder Presets in nnU-Net](documentation/resenc_presets.md)
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)

## Scope

nnU-Net is built for supervised semantic segmentation. It supports 2D and 3D data, arbitrary channel definitions, multiple image formats, and dataset-specific adaptation of preprocessing and network configuration.

It performs particularly well in training-from-scratch settings such as biomedical datasets, challenge datasets, and non-standard imaging problems where off-the-shelf natural-image pretrained models are often a poor fit.

For a concise overview of the design, see [How nnU-Net works](documentation/explanation/how-nnunet-works.md).

## Citation

Please cite the following paper when using nnU-Net:

```text
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021).
nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.
Nature Methods, 18(2), 203-211.
```

Additional recent work on residual encoder presets and benchmarking:

- [nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation](https://arxiv.org/pdf/2404.09556.pdf)

## Project Notes

- nnU-Net v2 is a complete reimplementation of the original nnU-Net with improved code structure and extensibility.
- Not every dataset creates every configuration. For example, the cascade is only generated when the dataset characteristics justify it.
- Detailed historical changes are summarized in [What is different in v2?](documentation/changelog.md).

# Acknowledgements
<img src="documentation/assets/HI_Logo.png" height="100px" />

<img src="documentation/assets/dkfz_logo.png" height="100px" />

nnU-Net is developed and maintained by the Applied Computer Vision Lab (ACVL) of [Helmholtz Imaging](http://helmholtz-imaging.de)
and the [Division of Medical Image Computing](https://www.dkfz.de/en/mic/index.php) at the
[German Cancer Research Center (DKFZ)](https://www.dkfz.de/en/index.html).
