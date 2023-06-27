from learn2rank.utils import hashit


class SmacOne:
    def __init__(self, cfg=None):
        self.cfg = cfg

    def __str__(self):
        return 'smac'

    @property
    def id(self):
        return hashit(str(self))


class SmacAll:
    def __init__(self, cfg=None):
        self.cfg = cfg

    def __str__(self):
        return 'smac_all'

    @property
    def id(self):
        return hashit(str(self))


class HeuristicOrder:

    def __init__(self, cfg=None):
        self.cfg = cfg

    def __str__(self):
        if self.cfg.name == 'HeuristicWeight':
            return "{}_{}".format(self.cfg.name, self.cfg.sort)
        else:
            return "{}_{}_{}".format(self.cfg.name, self.cfg.agg, self.cfg.sort)

    @property
    def id(self):
        return hashit(str(self))


class Lex:
    def __init__(self, cfg=None):
        self.cfg = cfg

    def __str__(self):
        return 'lex'

    @property
    def id(self):
        return hashit(str(self))


class Random:
    def __init__(self, cfg=None):
        self.cfg = cfg

    def __str__(self):
        return f'rand_{self.cfg.seed}'

    @property
    def id(self):
        return hashit(str(self))
