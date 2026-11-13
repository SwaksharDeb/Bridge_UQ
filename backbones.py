"""
Backbone dispatch: returns (model, train_loader, test_loader, config) for the
chosen pretrained registration network.

'tlrn' is fully contained in this repository (models/TLRN.py, data_utils.py).

'ltma', 'tm', 'voxelmorph' and 'vmplus' are separate registration projects with
their own model code, configs and checkpoints. They are not bundled here; to use
one, point the matching environment variable at a checkout of that project that
provides `uncertainty_sde_combined_acdc/data_utils.py` with a
`load_model_and_data` function:

    BRIDGEUQ_LTMA_ROOT   BRIDGEUQ_TM_ROOT   BRIDGEUQ_VXM_ROOT
"""

import importlib.util
import os
import sys

from data_utils import OUTPUT_ROOT

_EXTERNAL_ROOT_ENV = {
    'ltma':       'BRIDGEUQ_LTMA_ROOT',
    'tm':         'BRIDGEUQ_TM_ROOT',
    'voxelmorph': 'BRIDGEUQ_VXM_ROOT',
    'vmplus':     'BRIDGEUQ_VXM_ROOT',
}

_BACKBONE_TAGS = {
    'tlrn':       ('TLRN',       'basic_MSE_Penp_img1200Reg0.03_plus_ACDC'),
    'ltma':       ('TLMA_TGrad', 'config_plus_ACDC'),
    'tm':         ('TransMorph', 'basic_plus_ACDC'),
    'voxelmorph': ('VoxelMorph', 'voxelmorph'),
    'vmplus':     ('VoxelMorph', 'vmplus'),
}


# Top-level package names that exist (with different content) inside multiple
# project trees. They get cached in sys.modules once any module of this repo
# imports them, which then shadows the external project's versions when an
# external loader is exec'd. We evict them before each external load.
_SHADOW_NAMES = ('utils', 'models', 'dataload', 'networks', 'metrics')


def _evict_shadowed_modules():
    for name in list(sys.modules):
        if name in _SHADOW_NAMES or any(name.startswith(p + '.') for p in _SHADOW_NAMES):
            del sys.modules[name]


def _import_external_loader(backbone):
    env = _EXTERNAL_ROOT_ENV[backbone]
    project_root = os.environ.get(env)
    abs_path = os.path.join(project_root or '', 'uncertainty_sde_combined_acdc/data_utils.py')
    if not project_root or not os.path.isfile(abs_path):
        raise RuntimeError(
            f"--backbone {backbone} is not bundled with this repository (only "
            f"'tlrn' is). Set {env} to a checkout of that registration project "
            f"containing uncertainty_sde_combined_acdc/data_utils.py "
            f"(currently {env}={project_root!r}).")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    _evict_shadowed_modules()
    spec = importlib.util.spec_from_file_location(f'_{backbone}_data_utils', abs_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_backbone_and_data(backbone: str, combine_acdc: bool):
    backbone = backbone.lower()
    if backbone == 'tlrn':
        from data_utils import load_model_and_data
        out = load_model_and_data(combine_acdc=combine_acdc)
        model, train_loader, test_loader, config = out
        # workdir already set by data_utils (follows --no_combine_acdc too)
        return model, train_loader, test_loader, config

    if backbone in ('voxelmorph', 'vmplus'):
        out = _import_external_loader(backbone).load_model_and_data(
            resmode=backbone, combine_acdc=combine_acdc)
    elif backbone in ('ltma', 'tm'):
        out = _import_external_loader(backbone).load_model_and_data(
            combine_acdc=combine_acdc)
    else:
        raise ValueError(f"Unknown backbone {backbone!r}")

    model, train_loader, test_loader, config = out
    btag, etag = _BACKBONE_TAGS[backbone]
    config.model.params.workdir = os.path.join(OUTPUT_ROOT, btag, etag)
    return model, train_loader, test_loader, config
