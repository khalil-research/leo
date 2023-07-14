from omegaconf import OmegaConf
from xgboost import XGBRanker

from leo.utils import hashit


class SVMRank:
    """Dummy class for SVMRank model"""

    def __init__(self, cfg=None):
        self.cfg = cfg

    def __str__(self):
        return f'svmrank_c-{self.cfg.c}'

    @property
    def id(self):
        return hashit(str(self))


class GradientBoostingRanker(XGBRanker):
    def __init__(self, cfg=None):
        super(GradientBoostingRanker, self).__init__(
            **OmegaConf.to_container(cfg, resolve=True)
        )
        self.cfg = cfg
        self.id_str = None

    def __str__(self):
        id_str = f"nes-{self.cfg.n_estimators}_"
        id_str += f"md-{self.cfg.max_depth}_"
        id_str += f"rlam-{self.cfg.reg_lambda}_"
        id_str += f"lr-{self.cfg.learning_rate}_"
        id_str += f"gma-{self.cfg.gamma}_"
        id_str += f"mcw-{self.cfg.min_child_weight}_"
        id_str += f"raph-{self.cfg.reg_alpha}_"
        id_str += f"ss-{self.cfg.subsample}_"
        id_str += f"csbt-{self.cfg.colsample_bytree}_"
        id_str += f"gp-{self.cfg.grow_policy}"
        self.id_str = id_str

        return f"xgb_{id_str}"

    @property
    def id(self):
        return hashit(str(self))
