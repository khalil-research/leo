from pathlib import Path

from omegaconf import OmegaConf

res_path = Path(__file__).parent.parent.parent.parent / 'resources'

path = OmegaConf.create({
    'module': Path(__file__).parent.parent,
    'resources': res_path,
    'instances': res_path / 'instances',
    'bin': res_path / 'bin',
    'SmacI': res_path / 'SmacI_out',
    'SmacD': res_path / 'SmacD_out',
    'label': res_path / 'labels',
    'dataset': res_path / 'datasets',
    'model_cfg': res_path / 'model_cfg',
    'prediction': res_path / 'predictions',
    'pretrained': res_path / 'pretrained',
    'eval_order': res_path / 'eval_order',
    'model_summary': res_path / 'model_summary'})
