import sklearn.ensemble as skensem
import sklearn.linear_model as sklm
import sklearn.tree as sktree


class LinearRegression(sklm.LinearRegression):
    def __init__(self, cfg=None):
        super(LinearRegression, self).__init__()
        self.cfg = cfg


class Lasso(sklm.Lasso):
    def __init__(self, cfg=None):
        super(Lasso, self).__init__(alpha=cfg.alpha,
                                    fit_intercept=cfg.fit_intercept,
                                    tol=cfg.tol)
        self.cfg = cfg


class Ridge(sklm.Ridge):
    def __init__(self, cfg=None):
        super(Ridge, self).__init__(alpha=cfg.alpha,
                                    fit_intercept=cfg.fit_intercept,
                                    tol=cfg.tol)
        self.cfg = cfg


class DecisionTreeRegressor(sktree.DecisionTreeRegressor):
    def __init__(self, cfg=None):
        super(DecisionTreeRegressor, self).__init__(criterion=cfg.criterion,
                                                    max_depth=cfg.max_depth,
                                                    min_samples_split=cfg.min_samples_split,
                                                    min_samples_leaf=cfg.min_samples_leaf,
                                                    min_weight_fraction_leaf=cfg.min_weight_fraction_leaf,
                                                    max_features=cfg.max_features)
        self.cfg = cfg


class GradientBoostingRegressor(skensem.GradientBoostingRegressor):
    def __init__(self, cfg=None):
        super(GradientBoostingRegressor, self).__init__(loss=cfg.loss,
                                                        learning_rate=cfg.learning_rate,
                                                        n_estimators=cfg.n_estimators,
                                                        subsample=cfg.subsample,
                                                        criterion=cfg.criterion,
                                                        min_samples_split=cfg.min_samples_split,
                                                        min_samples_leaf=cfg.min_samples_leaf,
                                                        min_weight_fraction_leaf=cfg.min_weight_fraction_leaf,
                                                        max_depth=cfg.max_depth,
                                                        min_impurity_decrease=cfg.min_impurity_decrease,
                                                        max_features=cfg.max_features)
        self.cfg = cfg

    def _make_estimator(self, append=True):
        raise NotImplementedError()
