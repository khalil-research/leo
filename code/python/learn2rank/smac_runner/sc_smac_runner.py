from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from .smac_runner import SMACRunner


class SetcoverSMACRunner(SMACRunner):
    def __init__(self, cfg=None):
        super().__init__(cfg=cfg)

        # Hyperparams
        self.weight = None
        self.avg_value = None
        self.max_value = None
        self.min_value = None
        self.avg_value_by_weight = None
        self.max_value_by_weight = None
        self.min_value_by_weight = None
        self.label = None

        self.initialize_config_space()
        self.set_config_space()

    def initialize_config_space(self):
        width = self.cfg.width

        self.weight = UniformFloatHyperparameter("weight", -width.default, width.default)
        self.avg_value = UniformFloatHyperparameter("avg_value", -width.default, width.default)
        self.max_value = UniformFloatHyperparameter("max_value", -width.default, width.default)
        self.min_value = UniformFloatHyperparameter("min_value", -width.default, width.default)
        self.avg_value_by_weight = UniformFloatHyperparameter("avg_value_by_weight", -width.default, width.default)
        self.max_value_by_weight = UniformFloatHyperparameter("max_value_by_weight", -width.default, width.default)
        self.min_value_by_weight = UniformFloatHyperparameter("min_value_by_weight", -width.default, width.default)

        if width.label < 1:
            self.label = UniformFloatHyperparameter("label", 1 - width.label, width.default)
        else:
            self.label = UniformFloatHyperparameter("label", -width.default, width.default)

    def set_config_space(self):
        # Fetch incumbent config and initialize hyperparams
        incb = self.cfg.init_incumbent.split('/')
        problem = self.cfg.problem.name
        size = self.cfg.problem.size

        if incb[0] == 'smac_optimized':
            from learn2rank.prop_wt import optimized as prop_wt_opt
            assert problem in prop_wt_opt and size in prop_wt_opt[problem]
            pwts = prop_wt_opt[problem][size][incb[1]]
        else:
            from learn2rank.prop_wt import static as prop_wt_static
            pwts = prop_wt_static[incb[0]]

        self.weight.default_value = pwts['weight']
        self.avg_value.default_value = pwts['avg_value']
        self.max_value.default_value = pwts['max_value']
        self.min_value.default_value = pwts['min_value']
        self.avg_value_by_weight.default_value = pwts['avg_value_by_weight']
        self.max_value_by_weight.default_value = pwts['max_value_by_weight']
        self.min_value_by_weight.default_value = pwts['min_value_by_weight']
        self.label.default_value = pwts['label']

        # Add hyperparams to config store
        self.cs.add_hyperparameters([self.weight,
                                     self.avg_value,
                                     self.max_value,
                                     self.min_value,
                                     self.avg_value_by_weight,
                                     self.max_value_by_weight,
                                     self.min_value_by_weight,
                                     self.label])
