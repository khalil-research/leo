class SmacOne:
    def __init__(self, cfg=None):
        self.cfg = cfg

    @property
    def id(self):
        return 'smac'


class SmacAll:
    def __init__(self, cfg=None):
        self.cfg = cfg

    @property
    def id(self):
        return 'smac_all'


class MinWeight:
    """Dummy class for Min-weight model"""

    def __init__(self, cfg=None):
        self.cfg = cfg

    @property
    def id(self):
        return 'minwt'


class Canonical:
    def __init__(self, cfg=None):
        self.cfg = cfg

    @property
    def id(self):
        return 'canonical'
