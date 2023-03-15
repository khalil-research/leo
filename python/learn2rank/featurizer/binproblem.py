from operator import itemgetter

import numpy as np

from learn2rank.utils.const import BinproblemStaticOrderings
from .featurizer import Featurizer


class BinproblemFeaturizer(Featurizer):
    def __init__(self, cfg=None, data=None):
        self.norm_const = 100
        self.cfg = cfg
        self.data = data
        # Get features
        self.norm_value = (1 / self.norm_const) * np.asarray(self.data['value'])
        # self.norm_weight = np.mean(self.data['weight'], axis=0) + 1e-10
        self.norm_weight = self.data['weight'] / self.data['n_cons']
        self.n_objs, self.n_vars = self.norm_value.shape
        self.n_cons = self.data['cons_mat'].shape[0]

    def _get_instance_features(self):
        inst_feat = [self.n_objs / 7,
                     self.n_vars / 150,
                     self.n_cons / 20,
                     self.norm_weight.mean(),
                     self.norm_weight.std(),
                     self.norm_weight.min(),
                     self.norm_weight.max()]  # Weight aggregate stats

        # Value double-aggregate stats
        value_mean = self.norm_value.mean(1)
        value_min = self.norm_value.min(1)
        value_max = self.norm_value.max(1)

        inst_feat.extend([value_mean.mean(), value_mean.std(), value_mean.min(), value_mean.max(),
                          value_min.mean(), value_min.std(), value_min.min(), value_min.max(),
                          value_max.mean(), value_max.std(), value_max.min(), value_max.max()])

        # Constraint degree features
        cons_deg = self.data['cons_mat'].mean(1)
        inst_feat.extend([cons_deg.mean(), cons_deg.std(), cons_deg.min(), cons_deg.max()])

        inst_feat = np.asarray(inst_feat)

        return inst_feat

    def _get_heuristic_variable_rank_features(self):
        ranks = []
        for o in BinproblemStaticOrderings:
            if o.name == 'max_weight':
                idx_weight = [(i, w) for i, w in enumerate(self.data['weight'])]
                idx_weight.sort(key=itemgetter(1), reverse=True)
                idx_rank = self._get_rank(idx_weight)

            elif o.name == 'min_weight':
                idx_weight = [(i, w) for i, w in enumerate(self.data['weight'])]
                idx_weight.sort(key=itemgetter(1))
                idx_rank = self._get_rank(idx_weight)

            elif o.name == 'max_avg_value':
                mean_profit = np.mean(self.data['value'], 0)
                idx_profit = [(i, mp) for i, mp in enumerate(mean_profit)]
                idx_profit.sort(key=itemgetter(1), reverse=True)
                idx_rank = self._get_rank(idx_profit)

            elif o.name == 'min_avg_value':
                mean_profit = np.mean(self.data['value'], 0)
                idx_profit = [(i, mp) for i, mp in enumerate(mean_profit)]
                idx_profit.sort(key=itemgetter(1))
                idx_rank = self._get_rank(idx_profit)

            elif o.name == 'max_max_value':
                max_profit = np.max(self.data['value'], 0)
                idx_profit = [(i, mp) for i, mp in enumerate(max_profit)]
                idx_profit.sort(key=itemgetter(1), reverse=True)
                idx_rank = self._get_rank(idx_profit)

            elif o.name == 'min_max_value':
                max_profit = np.max(self.data['value'], 0)
                idx_profit = [(i, mp) for i, mp in enumerate(max_profit)]
                idx_profit.sort(key=itemgetter(1))
                idx_rank = self._get_rank(idx_profit)

            elif o.name == 'max_min_value':
                min_profit = np.min(self.data['value'], 0)
                idx_profit = [(i, mp) for i, mp in enumerate(min_profit)]
                idx_profit.sort(key=itemgetter(1), reverse=True)
                idx_rank = self._get_rank(idx_profit)

            elif o.name == 'min_min_value':
                min_profit = np.min(self.data['value'], 0)
                idx_profit = [(i, mp) for i, mp in enumerate(min_profit)]
                idx_profit.sort(key=itemgetter(1))
                idx_rank = self._get_rank(idx_profit)

            elif o.name == 'max_avg_value_by_weight':
                mean_profit = np.mean(self.data['value'], 0)
                profit_by_weight = [v / w for v, w in zip(mean_profit, self.data['weight'])]
                idx_profit_by_weight = [(i, f) for i, f in enumerate(profit_by_weight)]
                idx_profit_by_weight.sort(key=itemgetter(1), reverse=True)
                idx_rank = self._get_rank(idx_profit_by_weight)

            elif o.name == 'max_max_value_by_weight':
                max_profit = np.max(self.data['value'], 0)
                profit_by_weight = [v / w for v, w in zip(max_profit, self.data['weight'])]
                idx_profit_by_weight = [(i, f) for i, f in enumerate(profit_by_weight)]
                idx_profit_by_weight.sort(key=itemgetter(1), reverse=True)
                idx_rank = self._get_rank(idx_profit_by_weight)

            ranks.append([idx_rank[i] for i in range(self.n_vars)])

        ranks = (1 / self.n_vars) * np.asarray(ranks)
        return ranks

    def _get_variable_features(self):
        v_feat = np.vstack([self.norm_weight,
                            self.norm_value.mean(axis=0),
                            self.norm_value.std(axis=0),
                            self.norm_value.min(axis=0),
                            self.norm_value.max(axis=0),
                            self.norm_value.mean(axis=0) / self.norm_weight,
                            self.norm_value.max(axis=0) / self.norm_weight,
                            self.norm_value.min(axis=0) / self.norm_weight])

        # Constraint degree features
        cm, _v_feat = self.data['cons_mat'], []
        for v in range(self.n_vars):
            cons_deg_v = cm[cm[:, v] == 1].mean(1)
            _v_feat.append([cons_deg_v.mean(), cons_deg_v.std(), cons_deg_v.min(), cons_deg_v.max()])
        _v_feat = np.array(_v_feat)
        # Normalize
        _v_feat = _v_feat / (_v_feat.sum(0) + 1e-10)
        v_feat = np.vstack([v_feat, _v_feat.T])

        # Variable constraint features
        _v_feat = []
        for v in range(self.n_vars):
            cm_v = cm[cm[:, v] == 1].mean(0)
            dot = np.dot(cm_v, self.norm_weight)
            _v_feat.append([cm_v.mean(), cm_v.std(), cm_v.min(), cm_v.max(), dot])
        _v_feat = np.array(_v_feat)
        # Normalize
        _v_feat = _v_feat / (_v_feat.sum(0) + 1e-10)
        v_feat = np.vstack([v_feat, _v_feat.T])

        return v_feat

    @staticmethod
    def _get_rank(sorted_data):
        idx_rank = {}
        for rank, item in enumerate(sorted_data):
            idx_rank[item[0]] = rank

        return idx_rank

    def get(self):
        # Calculate instance features
        feat = {'raw': None, 'inst': None, 'var': None, 'vrank': None}
        raw_feat = np.vstack(([self.norm_value,
                               self.data['cons_mat']]))
        feat['raw'] = raw_feat.T

        inst_feat = self._get_instance_features()
        inst_feat = inst_feat.reshape((1, -1))
        inst_feat = np.repeat(inst_feat, self.n_vars, axis=0)
        feat['inst'] = inst_feat

        # Calculate item features
        var_feat = self._get_variable_features()
        feat['var'] = var_feat.T
        # print(item_feat.shape)

        var_rank_feat = self._get_heuristic_variable_rank_features()
        feat['vrank'] = var_rank_feat.T

        return feat
