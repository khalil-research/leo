from xgboost import XGBRanker
from omegaconf import OmegaConf


class GradientBoostingRanker(XGBRanker):
    def __init__(self, cfg=None):
        super(GradientBoostingRanker, self).__init__(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg

    @property
    def id(self):
        return f"xgb_nes-{self.cfg.n_estimators}_md-{self.cfg.max_depth}"
