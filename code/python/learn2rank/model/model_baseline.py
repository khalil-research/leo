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


class MinWeight:
    """Dummy class for Min-weight model"""

    def __init__(self, cfg=None):
        self.cfg = cfg

    def __str__(self):
        return 'minwt'

    @property
    def id(self):
        return hashit(str(self))


class Canonical:
    def __init__(self, cfg=None):
        self.cfg = cfg

    def __str__(self):
        return 'canonical'

    @property
    def id(self):
        return hashit(str(self))
