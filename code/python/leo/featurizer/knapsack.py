from operator import itemgetter

import numpy as np

from leo.utils.const import KnapsackStaticOrderings
from leo.utils.data import feat_names
from .featurizer import Featurizer


class KnapsackFeaturizer(Featurizer):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.norm_const = self.cfg.norm_const

    def _set_data(self, data):
        self.data = data
        # Get features
        self.norm_value = (1 / self.norm_const) * np.asarray(self.data['value'])
        self.norm_weight = (1 / self.norm_const) * np.asarray(self.data['weight'])
        self.n_objs, self.n_vars = self.norm_value.shape

    def _get_instance_features(self):
        inst_feat = [self.n_objs / 7,
                     self.n_vars / 100,
                     (np.ceil(self.norm_weight.sum()) / 2) / self.n_vars,  # Normalized capacity
                     self.norm_weight.mean(), self.norm_weight.min(), self.norm_weight.max(),
                     self.norm_weight.std()]  # Weight aggregate stats

        # Value double-aggregate stats
        value_mean = self.norm_value.mean(axis=1)
        value_min = self.norm_value.min(axis=1)
        value_max = self.norm_value.max(axis=1)

        inst_feat.extend([value_mean.mean(), value_mean.min(), value_mean.max(), value_mean.std(),
                          value_min.mean(), value_min.min(), value_min.max(), value_min.std(),
                          value_max.mean(), value_max.min(), value_max.max(), value_max.std()])

        inst_feat = np.asarray(inst_feat)

        return inst_feat

    def _get_heuristic_variable_rank_features(self):
        self.n_vars = len(self.data['weight'])

        ranks = []
        for o in KnapsackStaticOrderings:
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
        return np.vstack([self.norm_weight,
                          self.norm_value.mean(axis=0),
                          self.norm_value.min(axis=0),
                          self.norm_value.max(axis=0),
                          self.norm_value.std(axis=0),
                          self.norm_value.mean(axis=0) / self.norm_weight,
                          self.norm_value.max(axis=0) / self.norm_weight,
                          self.norm_value.min(axis=0) / self.norm_weight])

        # return variable_features
        #     item_features = np.vstack([item_features,
        #                                idx_rank_array])

    @staticmethod
    def _get_rank(sorted_data):
        idx_rank = {}
        for rank, item in enumerate(sorted_data):
            idx_rank[item[0]] = rank

        return idx_rank

    def get(self, data=None):
        self._set_data(data)

        # Calculate instance features
        feat = {'raw': None, 'inst': None, 'var': None, 'vrank': None}

        if self.cfg.raw:
            raw_feat = np.vstack((self.norm_value,
                                  self.norm_weight,
                                  np.repeat(self.data['capacity'] / self.norm_const, self.n_vars).reshape(1, -1)))
            feat['raw'] = raw_feat.T
            assert feat['raw'].shape[1] == self.data['n_objs'] + 2

        if self.cfg.context:
            inst_feat = self._get_instance_features()
            inst_feat = inst_feat.reshape((1, -1))
            inst_feat = np.repeat(inst_feat, self.n_vars, axis=0)
            feat['inst'] = inst_feat
            assert feat['inst'].shape[1] == len(feat_names['inst'])

        # Calculate item features
        var_feat = self._get_variable_features()
        feat['var'] = var_feat.T
        # print(item_feat.shape)
        assert feat['var'].shape[1] == len(feat_names['var'])

        var_rank_feat = self._get_heuristic_variable_rank_features()
        feat['vrank'] = var_rank_feat.T
        assert feat['vrank'].shape[1] == len(feat_names['vrank'])

        return feat
