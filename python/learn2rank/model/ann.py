from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MLPConfig:
    layers: str
    act: str
    dp: float
    dp_last: bool


@dataclass
class SetEncoderConfig:
    mlp: MLPConfig
    agg_type: str


@dataclass
class Predictor:
    on: bool
    mlp: MLPConfig


@dataclass
class TFEncoderConfig:
    on: bool
    d_model: int
    n_heads: int
    d_ff: int
    dp: float
    act: str
    n_layers: int


class MLP(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super(MLP, self).__init__()

        self.cfg = cfg
        self.cfg.layers = list(map(int, self.cfg.layers.split(',')))
        self.activation = getattr(nn, self.cfg.act)()
        self.dropout = nn.Dropout(p=self.cfg.dp)

        self.layers_lst = nn.ModuleList()
        for i in range(len(self.cfg.layers) - 1):
            if i < len(self.cfg.layers) - 2 or self.cfg.dp_last:
                self.layers_lst.append(nn.Linear(self.cfg.layers[i], self.cfg.layers[i + 1]))
                self.layers_lst.append(self.activation)
                self.layers_lst.append(self.dropout)
            else:
                self.layers_lst.append(nn.Linear(self.cfg.layers[i], self.cfg.layers[i + 1]))
                self.layers_lst.append(self.activation)

        self.mlp = nn.Sequential(*self.layers_lst)

    def forward(self, x):
        return self.mlp(x)


class SetEncoder(nn.Module):
    def __init__(self, cfg: SetEncoderConfig):
        super(SetEncoder, self).__init__()
        self.cfg = cfg
        self.mlp = MLP(self.cfg.mlp)

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor of shape bs x n_items x d_model
            Input tensor

        Returns
        -------
        x: torch.Tensor of shape bs x ft2_layers[-1]
            Output tensor
        """

        # bs x n_items x ft1_layers[-1]
        x = self.mlp(x)

        # bs x ft1_layers[-1]
        if self.cfg.agg_type == 'sum':
            x = torch.sum(x, 1)
        elif self.cfg.agg_type == 'mean':
            x = torch.mean(x, 1)
        else:
            raise ValueError(self.agg_type)

        return x


class TFEncoder(nn.Module):
    """Transformer-based feature encoder"""

    def __init__(self, cfg: TFEncoderConfig):
        super(TFEncoder, self).__init__()
        self.cfg = cfg

        # Transformer encoder
        _encoder_layer = nn.TransformerEncoderLayer(d_model=self.cfg.d_model,
                                                    nhead=self.cfg.n_heads,
                                                    dim_feedforward=self.cfg.d_ff,
                                                    dropout=self.cfg.dp,
                                                    activation=self.cfg.act)
        self.encoder = nn.TransformerEncoder(_encoder_layer, num_layers=self.cfg.n_layers)

    def forward(self, x):
        return self.encoder(x)


class NeuralRankingMachine(nn.Module):
    def __init__(self, cfg=None):
        super(NeuralRankingMachine, self).__init__()
        assert cfg is not None

        self.cfg = cfg

        # Two-step feature encoding process
        # Step 1: Simple linear embedding
        self.feature_encoder = MLP(self.cfg.feat_enc)
        # Step 2: Apply TransformerEncoder on the linear embedding (Optional)
        self.tf_enc = TFEncoder(self.cfg.tf_enc) if self.cfg.tf_enc.switch else None

        # The learned encoding can be used for multiple tasks
        # Task 1: Rank prediction
        self.rank_predictor = MLP(self.cfg.rp.mlp)
        self.set_encoder = SetEncoder(self.cfg.set_enc) if self.cfg.wp.switch or self.cfg.tp.switch else None
        # Task 2: SMAC Weight prediction (Optional)
        self.weight_predictor = MLP(self.cfg.wp.mlp) if self.cfg.wp.switch else None
        # Task 3: SMAC Time prediction (Optional)
        self.time_predictor = MLP(self.cfg.tp.mlp) if self.cfg.tp.switch else None

    def forward(self, var_feat):
        # Variable feature encoding
        var_enc = self.feature_encoder(var_feat)
        var_enc = self.tf_enc(var_enc) if self.cfg.tf_enc.switch else var_enc

        # Rank prediction
        yp_rank = self.rank_predictor(var_enc)
        yp_rank = torch.squeeze(yp_rank, -1)

        # Weight and time prediction
        yp_weight, yp_time = None, None
        inst_enc = self.set_encoder(var_enc) if self.cfg.wp.switch or self.cfg.tp.switch else None
        if inst_enc is not None:
            yp_weight = self.weight_predictor(inst_enc) if self.cfg.wp.switch else None
            yp_time = self.time_predictor(inst_enc) if self.cfg.tp.switch else None

        return yp_rank, yp_weight, yp_time
