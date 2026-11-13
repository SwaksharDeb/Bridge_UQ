"""
Test-time evaluation of a SAVED per-sequence sigma (s_K/batch_*.pt from main.py).

This is the scoring half of main.py pulled out, the way test_langevin.py is for
main_langevin.py: no EM is run. main.py itself runs unchanged, with its sigma
estimator (mcem_refine / total_derivative_refine) swapped in memory for a loader
of the saved sigma. Everything downstream -- the Langevin UQ draws,
_compute_sparsification_curves / _compute_risk_coverage_curves,
the registration metrics and the summaries -- is main.py's own code. All
main.py flags pass through.

The FINAL chain, as in test_langevin.py: ONE chain per
batch, --burn_in steps discarded, then --num_runs states taken every --thin
steps; ALL those post-burn-in states are the UQ samples (chain length
burn_in + num_runs * thin). main.py's own UQ draws are instead --num_runs
independent chains of --langevin_steps steps, keeping only each final state;
--independent_draws restores that. The step size, init, preconditioning,
noise smoothing etc. are main.py's --langevin_* flags; --langevin_steps and
--langevin_burn_in are not used by the chain (the latter only ever fed the
EM E-step, which is not run here).

Batching. Each saved batch file is replayed as ONE batch with exactly the
subjects it was fitted on. With the default --guidance_norm batch the likelihood
score is divided by the batch size, so re-batching would change the draws;
--batch_size is therefore ignored.

Which sigma. Always the M-step's raw sigma, used exactly as loaded -- no
stretch, no blur: 'scaling_factors_em_raw' when the file has one (older runs
that post-processed their sigma), else 'scaling_factors_raw' (current runs save
the raw sigma there).

Read from the s_K file unless the flag is given explicitly: --gamma,
--guidance_scale, --guidance_norm, --sigma_convention and --num_diffusion_steps.
The sampler is always Langevin; the --langevin_* flags should be the ones sigma
was fitted with.

--border N excludes an N-pixel image border from nAUSE / nAURC: the uncertainty
and per-pixel error maps are cropped right before they are flattened (the
segmentation warp still runs on the full field).

--nause_region ring (default) scores nAUSE / nAURC only on the myocardium of
each target frame: the pixels of its LV ring (lv_segs[frame] > 0.5), with no
cavity and nothing outside the ring. With --error_metric segmentation the error
is the usual ring-mask disagreement, so only errors that lie ON the target ring
count (a warped ring spilling into the cavity or past the outer wall is not
scored). --border is not applied with ring. --nause_region full restores
whole-image scoring.

Each frame's per-pixel uncertainty map is ALWAYS moved with the heart before
it is scored: it is
warped with that frame's registration displacement (VecInt + SpatialTransformer,
the same forward warp that carries the source mask to the frame), treating the
map as attached to frame-0 anatomy. sigma is one map for the whole sequence while
the wall moves, so without this its low band lines up with the wall only at one
phase of the cycle. This only changes the scoring; the draws are unchanged.

Usage:
  python test_mcem.py \
      --s_K_dir <.../uncertainty_sde_combined_acdc_v3_iclr_mcem_gd/s_K> \
      --files batch_000.pt --output_dir ./test_mcem_batch000 \
      --burn_in 300 --thin 1 --num_runs 100 \
      --error_metric segmentation --border 2 \
      --num_workers 0
"""
import argparse
import glob
import inspect
import os
import re
import sys
import textwrap


import torch
import torch.utils.data as tud

import test_fast as TF
import main as M
from mcem import scaling_to_sigma
from langevin import langevin_velocities

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument('--s_K_dir', default='outputs/TLRN/basic_MSE_Penp_img1200Reg0.03_plus_ACDC/visualization/uncertainty_sde_combined_acdc_v3_iclr_mcem_gd/sigma_vis/batch_000/values',
                 help='Directory holding the saved batch_*.pt files.')
pre.add_argument('--files', nargs='+', default=['round_0010.pt'],
                 help='Subset of batch files in --s_K_dir (default: all batch_*.pt). '
                      'Also accepts sigma_vis/batch_XXX/values/round_*.pt files.')
pre.add_argument('--meta_file', default=None,
                 help='s_K batch_*.pt to take subject_paths / gamma / guidance etc. from '
                      'when a --files entry lacks them (round_*.pt). Default: '
                      '<vis dir>/s_K/batch_XXX.pt inferred from the round file path.')
pre.add_argument('--border', type=int, default=2,
                 help='Exclude this many border pixels from nAUSE / nAURC (default: 0).')
pre.add_argument('--nause_region', choices=['ring', 'full'], default='ring',
                 help="Pixels nAUSE / nAURC are computed over. 'ring' (default): the "
                      "target frame's LV ring only (myocardium, no cavity); 'full': "
                      "the whole image (minus --border).")
pre.add_argument('--error_metric', choices=['similarity', 'segmentation'],
                 default='segmentation',
                 help="Per-pixel error used for nAUSE / nAURC (default: segmentation). "
                      "'segmentation': per-pixel disagreement between the warped "
                      "frame-0 LV mask and the frame's LV mask. 'similarity': "
                      "(warped - target)^2. Passed on to main.py's --error_metric.")
# ---- the FINAL chain, as test_langevin.py --------------------------------
pre.add_argument('--burn_in', type=int, default=500,
                 help='Chain steps discarded before sampling (default: 300). The '
                      '--num_runs UQ samples are then ALL the post-burn-in states of '
                      'one chain per batch, --thin steps apart: chain length '
                      'burn_in + num_runs * thin. --langevin_steps is not used.')
pre.add_argument('--thin', type=int, default=1,
                 help='Keep every k-th post-burn-in state (default: 1).')
pre.add_argument('--independent_draws', type=int, default=1000,
                 help="main.py's own UQ draws instead of the chain: this many "
                      "independent chains of --langevin_steps steps, final state of "
                      "each kept (--burn_in / --thin ignored). Default 1000. It sets "
                      "--num_runs unless --num_runs is given explicitly. 0 uses the "
                      "single chain instead (--burn_in / --thin, --num_runs 100).")
ours, rest = pre.parse_known_args()


def _given(flag):
    return any(a == flag or a.startswith(flag + '=') for a in rest)


# ---- Load the saved sigma -------------------------------------------------
names = ours.files or sorted(os.path.basename(p) for p in
                             glob.glob(os.path.join(ours.s_K_dir, 'batch_*.pt')))
if not names:
    raise SystemExit(f"No batch_*.pt files in {ours.s_K_dir}")

_batches = []            # one entry per saved file, replayed as one batch
meta = None
def _round_meta_path(path):
    """sigma_vis/batch_XXX/values/round_*.pt -> s_K/batch_XXX.pt of the same run."""
    if ours.meta_file:
        return ours.meta_file
    m = re.search(r'^(.*)/sigma_vis/(batch_\d+)/values/[^/]+$', os.path.abspath(path))
    if not m:
        raise SystemExit(f"[test_mcem] {path} has no subject_paths; pass --meta_file")
    return os.path.join(m.group(1), 's_K', m.group(2) + '.pt')


for name in names:
    path = name if os.path.isfile(name) else os.path.join(ours.s_K_dir, name)
    d = torch.load(path, map_location='cpu', weights_only=False)
    if 'subject_paths' not in d:
        # round_*.pt: per-round EM sigma + subject_ids only;
        # settings come from the run's s_K file.
        mpath = _round_meta_path(path)
        md = torch.load(mpath, map_location='cpu', weights_only=False)
        if list(md['subject_paths']) != list(d['subject_ids']):
            raise RuntimeError(f"{path}: subject_ids differ from {mpath}'s subject_paths")
        d = {**{k: v for k, v in md.items() if k not in
                ('scaling_factors_raw', 'scaling_factors_em_raw', 'sigma', 'sigma_post')},
             'subject_paths': d['subject_ids'],
             'scaling_factors_em_raw': d['scaling_factors_raw']}
        print(f"[test_mcem] {name}: settings from {mpath}")
    if d.get('scaling_factors_em_raw') is not None:
        s, src = d['scaling_factors_em_raw'], 'scaling_factors_em_raw'
    else:
        s, src = d['scaling_factors_raw'], 'scaling_factors_raw'
    if s.shape[0] != len(d['subject_paths']):
        raise RuntimeError(f"{path}: sigma batch dim {s.shape[0]} != "
                           f"{len(d['subject_paths'])} subject_paths")
    cur = {k: d.get(k) for k in ('gamma', 'guidance_scale', 'guidance_norm',
                                 'sigma_convention', 'num_diffusion_steps')}
    cur['m_step'] = ('closed_form' if 'em_rounds' in d
                     else 'total_derivative' if 'td_steps' in d else None)
    if meta is None:
        meta = cur
    elif cur != meta:
        raise RuntimeError(f"{path}: settings {cur} differ from the other files' {meta}")
    s = s.float()
    _batches.append({'name': name, 's': s, 'keys': list(d['subject_paths']),
                     'post': d.get('sigma_post')})
    print(f"[test_mcem] {name}: {s.shape[0]} subjects, sigma from '{src}'"
          f" (saved sigma_post: {(d.get('sigma_post') or {}).get('range', 'none')})")

# ---- main.py argv: saved settings unless given explicitly -----------------
inject = {'--gamma': meta['gamma'], '--guidance_scale': meta['guidance_scale'],
          '--guidance_norm': meta['guidance_norm'],
          '--sigma_convention': meta['sigma_convention'],
          '--num_diffusion_steps': meta['num_diffusion_steps'],
          '--num_runs': ours.independent_draws if ours.independent_draws > 0 else 100}
for flag, val in inject.items():
    if val is not None and not _given(flag):
        rest += [flag, str(val)]
        print(f"[test_mcem] {flag} {val}  (from s_K)" if flag != '--num_runs'
              else f"[test_mcem] {flag} {val}  (test default)")
if not ours.independent_draws:
    print(f"[test_mcem] chain: burn_in {ours.burn_in}, thin {ours.thin}")
rest += ['--error_metric', ours.error_metric]
print(f"[test_mcem] --error_metric {ours.error_metric}")
# Never write main.py's own s_K output over the files being evaluated.
if not _given('--output_dir'):
    _default_out = './test_mcem_' + '_'.join(
        os.path.splitext(os.path.basename(n))[0] for n in names)
    rest += ['--output_dir', _default_out]
    print(f"[test_mcem] --output_dir {_default_out}  (test default)")
_out = rest[rest.index('--output_dir') + 1] if '--output_dir' in rest else \
    next(a.split('=', 1)[1] for a in rest if a.startswith('--output_dir='))
if os.path.realpath(os.path.join(_out, 's_K')) == os.path.realpath(ours.s_K_dir):
    raise SystemExit("[test_mcem] --output_dir would overwrite the input s_K files.")

sys.argv = [sys.argv[0]] + rest

# ---- Replay each saved file as one batch ----------------------------------
_RealDataLoader = tud.DataLoader
_armed = [False]


def _dataset_indices(dataset, keys):
    """Map saved subject keys to dataset indices. main.py keys a subject by its
    'path' when the loader yields one, else by 'subj_{global index:05d}'."""
    m = [re.fullmatch(r'subj_(\d+)', k) for k in keys]
    if all(m):
        return [int(x.group(1)) for x in m]
    by_path = {str(dataset[i].get('path')): i for i in range(len(dataset))}
    missing = [k for k in keys if k not in by_path]
    if missing:
        raise RuntimeError(f"{len(missing)} saved subject(s) not in the dataset, "
                           f"e.g. {missing[0]}")
    return [by_path[k] for k in keys]


def _replay_loader(dataset, batch_size=None, shuffle=False, **kw):
    if not _armed[0]:
        return _RealDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kw)
    batch_sampler = [_dataset_indices(dataset, b['keys']) for b in _batches]
    print(f"[test_mcem] replaying {len(batch_sampler)} saved batch(es): "
          f"{[len(b) for b in batch_sampler]} subjects")
    return _RealDataLoader(dataset, batch_sampler=batch_sampler, **kw)


_real_load = M.load_backbone_and_data


def _load_then_arm(*a, **kw):
    out = _real_load(*a, **kw)
    _armed[0] = True      # only main.py's own DataLoader call is replayed
    return out


M.load_backbone_and_data = _load_then_arm
tud.DataLoader = _replay_loader

_next = [0]


def _load_sigma(trainer, source_img, target_imgs, v_reg_list, convention=None,
                device='cuda', _td=False, **_):
    b = _batches[_next[0]]
    _next[0] += 1
    s = b['s'].to(device=device, dtype=torch.float32)
    if s.shape != v_reg_list[0].shape:
        raise ValueError(f"{b['name']}: saved sigma {tuple(s.shape)} does not match "
                         f"the batch {tuple(v_reg_list[0].shape)}")
    n_frames = len(v_reg_list)
    sf_bb = M._expand_sf_bb(s, n_frames)[0]
    # closed_form returns sigma in --sigma_convention units, total_derivative s itself
    sigma = s if _td else scaling_to_sigma(s, trainer.brownian_bridge.T, convention)
    v_bar = [v.detach().float() for v in v_reg_list]
    u = [torch.zeros_like(v) for v in v_bar]
    return sf_bb, sigma, v_bar, u, [], torch.zeros(s.shape[0], dtype=torch.long)


M.mcem_refine = _load_sigma
M.total_derivative_refine = lambda trainer, loss_fn, *a, **kw: _load_sigma(
    trainer, *a, _td=True, **kw)

# ---- Final chain: all post-burn-in states of ONE chain per batch ----------
# main.py's UQ loop calls draw_velocities once per run, each call an
# independent chain whose final state is kept. Here the first call of a batch
# runs one chain -- burn_in discarded steps, then num_runs states every thin
# steps (langevin_velocities' collect_every, one call so the walk is
# continuous) -- and the loop's num_runs calls hand those states out in order.
_num_runs = int(next((rest[i + 1] for i, a in enumerate(rest) if a == '--num_runs'),
                     next((a.split('=', 1)[1] for a in rest
                           if a.startswith('--num_runs=')), 100)))
_real_draw = M.draw_velocities
_queue = []


def _chain_draw(sampler, trainer, v_reg_list, scaling_factors_bb, source_img,
                target_imgs, num_samples=1, sigma=None, langevin_kw=None):
    if sampler != 'langevin' or ours.independent_draws:
        return _real_draw(sampler, trainer, v_reg_list, scaling_factors_bb,
                          source_img, target_imgs, num_samples=num_samples,
                          sigma=sigma, langevin_kw=langevin_kw)
    if not _queue:
        # Same sampler kwargs draw_velocities builds.
        kw = dict(langevin_kw or {})
        kw.setdefault('gamma', trainer.gamma)
        kw.setdefault('guidance_scale', trainer.guidance_scale)
        kw.setdefault('reg_weight', getattr(trainer.brownian_bridge, 'reg_weight', 0.1))
        kw.setdefault('inshape', (trainer.img_size, trainer.img_size))
        kw.setdefault('pin_v_endpoints', False)
        for k in ('return_stats', 'return_latent', 'n_steps', 'init_eta'):
            kw.pop(k, None)
        xi = None
        if ours.burn_in > 0:
            _, _, xi = langevin_velocities(
                v_reg_list, sigma, source_img, target_imgs,
                n_steps=ours.burn_in, return_latent=True, **kw)
        thin = max(1, ours.thin)
        _, _, _, collected = langevin_velocities(
            v_reg_list, sigma, source_img, target_imgs,
            n_steps=_num_runs * thin, init_eta=xi, return_latent=True,
            collect_every=thin, **kw)
        _queue.extend([v.detach() for v in s] for s in collected)
        print(f"  [test_mcem] chain: {ours.burn_in} burn-in + {_num_runs * thin} "
              f"steps -> {len(collected)} samples every {thin}")
    v_t_list = _queue.pop(0)
    return v_t_list, [torch.zeros_like(v) for v in v_t_list]


M.draw_velocities = _chain_draw

# ---- Scoring: warp the uncertainty with the heart; LV ring, or image - border
CROP = ""
_WARP = {}

def _warp_to_frame(unc, v_reg, inshape):
    """Carry a [H, W] map from frame 0 into the frame of v_reg (same forward
    warp as the source mask)."""
    dev = v_reg.device
    if dev not in _WARP:
        _WARP[dev] = (TF.VecInt(inshape, TSteps=7).to(dev),
                      TF.SpatialTransformer(inshape).to(dev))
    vec_int, transformer = _WARP[dev]
    with torch.no_grad():
        u = torch.as_tensor(unc, dtype=torch.float32, device=dev)[None, None]
        u, _ = transformer(u, vec_int(v_reg)[-1])
    return u[0, 0].cpu().numpy()

TF._warp_to_frame = _warp_to_frame
CROP += ("        if v_series is None or inshape is None:\n"
         "            raise ValueError('warping the uncertainty needs v_series and inshape')\n"
         "        uncertainty = _warp_to_frame(uncertainty, "
         "v_series[sdef_idx][subject_idx:subject_idx+1], inshape)\n")
print("[test_mcem] uncertainty warped into each frame with the registration displacement")
if ours.nause_region == 'ring':
    CROP += ("        if lv_segs is None or lv_segs.shape[1] <= frame_idx:\n"
             "            continue\n"
             "        _roi = (lv_segs[subject_idx, frame_idx] > 0.5).cpu().numpy()\n"
             "        if not _roi.any():\n"
             "            continue\n"
             "        uncertainty = uncertainty[_roi]\n"
             "        pixel_error = pixel_error[_roi]\n")
    print("[test_mcem] nAUSE / nAURC on the LV ring only (myocardium, no cavity)")
elif ours.border > 0:
    CROP += ("        uncertainty = uncertainty[_BORDER:-_BORDER, _BORDER:-_BORDER]\n"
            "        pixel_error = pixel_error[_BORDER:-_BORDER, _BORDER:-_BORDER]\n")
    TF._BORDER = ours.border
    print(f"[test_mcem] excluding a {ours.border}-pixel border from nAUSE / nAURC")
if CROP:
    ANCHOR = "        eps = 1e-6\n"
    for fn in ('_compute_sparsification_curves', '_compute_risk_coverage_curves'):
        src = inspect.getsource(getattr(TF, fn))
        if src.count(ANCHOR) != 1:
            raise RuntimeError(f"{fn}: expected exactly one '{ANCHOR.strip()}' anchor")
        exec(textwrap.dedent(src.replace(ANCHOR, CROP + ANCHOR)), TF.__dict__)
        setattr(M, fn, getattr(TF, fn))

if __name__ == '__main__':
    M.main()
