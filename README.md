# BridgeUQ: Bayesian Aleatoric Uncertainty Quantification for Time-Series Image Registration via Brownian Bridge Prior

![BridgeUQ overview](figure/iclr_v3_updated.png)

*An overview of BridgeUQ integrated with pretrained registration networks (top) and its detailed design (bottom). Given an image sequence and pretrained velocity estimates, BridgeUQ estimates the diffusion variance and samples plausible velocity trajectories to quantify registration uncertainty.*

## Abstract

This paper presents BridgeUQ, a Bayesian framework for aleatoric uncertainty quantification in pretrained deterministic registration models. In contrast to previous approaches that mainly focus on pairwise registration, BridgeUQ models posterior uncertainty over spatiotemporal deformation trajectories, capturing the intrinsic ambiguity of registration solutions across both space and time. We introduce a hierarchical Gaussian process prior in the temporal deformation space, based on a Brownian bridge formulation with an inverse-Gamma hyperprior on the diffusion variance, to govern stochastic sampling around the predicted registration solution. An energy-based likelihood derived from the pretrained registration objective ensures that sampled deformations remain faithful to the observed data. To facilitate efficient inference, we develop an amortized maximum-a-posteriori (MAP) scheme combined with conditional diffusion sampling to approximate the resulting posterior. BridgeUQ can be integrated with pretrained registration models without retraining the registration backbone. We validate the framework on cardiac and brain magnetic resonance imaging datasets, demonstrating anatomically meaningful uncertainty estimates across multiple registration models.

## Overview

BridgeUQ quantifies uncertainty around a pretrained registration solution in velocity space:

- **Pretrained registration:** estimate a velocity trajectory from an input image sequence while keeping the registration backbone fixed.
- **Temporal prior:** use a hierarchical Gaussian process with a Brownian bridge prior to model temporally correlated perturbations around the predicted trajectory.
- **Data consistency:** guide conditional sampling using an energy-based likelihood derived from the registration objective.
- **Uncertainty estimation:** summarize variability across sampled trajectories as spatial uncertainty maps.

The paper evaluates cardiac cine MRI and longitudinal brain MRI. The scripts currently included in this repository focus on cardiac experiments and diffusion-variance estimation variants.

**Current release:** this repository is not yet a standalone reproduction package. The scripts import shared project modules that are not included here, and the data loaders and pretrained registration checkpoints must be supplied separately. In particular, `main.py` implements per-sequence MC-EM with a Langevin E-step and an Adam M-step; it is not the paper's amortized MAP training entry point.

## Requirements

The included files directly use:

- Python with CUDA-enabled PyTorch
- `numpy`, `matplotlib`, and `tqdm`
- Shared registration, Brownian bridge, loss, training, and evaluation modules from the original project
- Preprocessed cardiac sequences and pretrained registration checkpoints compatible with the shared data loader

Install PyTorch for your CUDA environment, then install the directly imported utility packages:

```bash
git clone https://github.com/SwaksharDeb/Bridge_UQ.git
cd Bridge_UQ

pip install numpy matplotlib tqdm
```

Additional dependencies are determined by the shared modules. A complete environment specification is not currently included.

## Configuration

The current scripts use the original project's Python package layout:

- `uncertainty_sde_combined_acdc_v3_iclr`: shared backbone loaders, Brownian bridge implementation, networks, losses, trainer, warping, and evaluation helpers.
- `uncertainty_sde_combined_acdc_v3_iclr_mcem_gd`: the package name used for the files in this repository.

To use the examples below, place this repository under the second package name within a parent directory containing the shared package, and add that parent directory to `PYTHONPATH`. The GEM script also requires `samplers.py` and `sigma_contrast.py` in the second package; these files are not currently included.

The scripts contain an author-specific `sys.path` entry. Replace it with your local project path or configure `PYTHONPATH` accordingly. Configure dataset locations and pretrained checkpoints through the shared `load_backbone_and_data` implementation.

Selected `main.py` options:

| Option | Purpose |
| --- | --- |
| `--backbone` | Registration backbone: `tlrn`, `ltma`, `tm`, `voxelmorph`, or `vmplus` |
| `--batch_size` | Image sequences processed per batch |
| `--em_rounds` | Number of MC-EM rounds |
| `--em_samples` | Sampled trajectories per sequence in each E-step |
| `--sample_chunk` | Number of trajectories sampled together |
| `--gd_iters` | Maximum Adam iterations in each M-step |
| `--alpha`, `--beta` | Inverse-Gamma prior parameters |
| `--num_runs` | Final trajectory samples used to estimate uncertainty |
| `--gamma` | Likelihood energy scale |
| `--guidance_norm` | `batch` or `per_sequence` likelihood-gradient normalization |
| `--output_dir` | Result directory |

The default `batch` normalization makes likelihood guidance depend on the effective batch size. Use `per_sequence` for normalization independent of batch size, and tune `gamma` for the selected convention. Script defaults describe the current experimental variant and should not be assumed to match every setting reported in the paper.

## Run — per-sequence variance estimation

**These examples require the shared modules, data, and checkpoints described above. A fresh clone alone cannot run them.** Run from the parent project directory after arranging the package layout.

```bash
python -m uncertainty_sde_combined_acdc_v3_iclr_mcem_gd.main \
  --backbone tlrn \
  --batch_size 8 \
  --em_rounds 10 \
  --em_samples 500 \
  --sample_chunk 32 \
  --num_runs 100 \
  --output_dir ./results_bridgeuq_mcem
```

For a smaller check of the variance-optimization path:

```bash
python -m uncertainty_sde_combined_acdc_v3_iclr_mcem_gd.main \
  --em_rounds 3 \
  --em_samples 32 \
  --sample_chunk 32 \
  --batch_size 8 \
  --max_batches 1 \
  --num_workers 0 \
  --optimize_only \
  --no_training_vis \
  --vis_every 0 \
  --output_dir ./results_bridgeuq_smoke
```

## Run — amortized variance-network variant

`gem_unet.py` implements a generalized EM variant that trains a shared variance-prediction network using fixed Monte Carlo targets. This is a separate experimental implementation; it should not be treated as an exact reproduction of the paper's amortized training procedure.

After supplying its additional dependencies:

```bash
python -m uncertainty_sde_combined_acdc_v3_iclr_mcem_gd.gem_unet \
  --rounds 300 \
  --batch_size 8 \
  --em_samples 32 \
  --m_steps 5 \
  --langevin_algorithm psgld \
  --langevin_steps 100 \
  --langevin_step_size 0.05 \
  --gamma 5e-6 \
  --reg_weight 0 \
  --output_dir ./results_bridgeuq_gem
```

## Outputs

The per-sequence pipeline saves run settings in `run_config.json`, optimization history in `sigma_history.json`, and per-batch variance artifacts under `s_K/`. Visualization and registration/UQ reports depend on the enabled options and shared evaluation helpers.

The GEM variant saves network checkpoints under `gem_checkpoints/`, training history in `gem_history.json`, and evaluation visualizations under `gem_eval/`. Pass `--export_test` to export test-set variance artifacts under `gem_s_K/`.

## Code structure

| File | Description |
| --- | --- |
| `main.py` | Per-sequence MC-EM with Langevin sampling and an Adam variance update |
| `mcem.py` | Variance-estimation utilities and alternative refinement methods |
| `langevin.py` | Langevin trajectory sampling utilities |
| `gem_unet.py` | Generalized EM training of an amortized variance network |
| `test_mcem.py` | Evaluation of saved variance estimates using shared evaluation helpers |
| `viz.py` | Variance-map visualization utilities |
| `figure/iclr_v3_updated.png` | BridgeUQ overview figure |
