from __future__ import annotations

import torch
import torch.nn as nn

from OPFT import JiT_MODELS, ObservationEmbedder, TimestepEmbedder


class JointJiT(nn.Module):

    def __init__(
        self,
        backbone="jit_small",
        image_size=(128, 128),
        patch_size=(32, 32),
        field_channels=10,
        observation_dim=1050,
        relperm_dim=6,
        global_hidden=256,
    ):
        super().__init__()
        if backbone not in JiT_MODELS:
            raise KeyError(
                f"unknown JiT backbone {backbone!r}; available: {tuple(JiT_MODELS)}"
            )
        self.field_channels = field_channels
        self.observation_dim = observation_dim
        self.relperm_dim = relperm_dim
        self.spatial_model = JiT_MODELS[backbone](
            img_size=image_size,
            patch_size=patch_size,
            in_chans=field_channels,
            obs_dim=observation_dim + relperm_dim,
        )
        self.observation_embedder = ObservationEmbedder(observation_dim, global_hidden)
        self.time_embedder = TimestepEmbedder(global_hidden)
        statistics_dim = 4 * field_channels
        self.relperm_head = nn.Sequential(
            nn.Linear(2 * global_hidden + relperm_dim + statistics_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, relperm_dim),
        )

    @staticmethod
    def _field_statistics(field):
        mean = field.mean(dim=(-2, -1))
        std = field.std(dim=(-2, -1), unbiased=False)
        return torch.cat((mean, std), dim=1)

    def forward(self, field, relperm, t, observation):
        if field.ndim != 4 or field.shape[1] != self.field_channels:
            raise ValueError(f"field must have shape (B, {self.field_channels}, H, W)")
        if relperm.ndim != 2 or relperm.shape[1] != self.relperm_dim:
            raise ValueError(f"relperm must have shape (B, {self.relperm_dim})")
        if observation.ndim != 2 or observation.shape[1] != self.observation_dim:
            raise ValueError(f"observation must have shape (B, {self.observation_dim})")

        spatial_condition = torch.cat((observation, relperm), dim=1)
        field_prediction = self.spatial_model(field, t, spatial_condition)
        global_features = torch.cat(
            (
                self.observation_embedder(observation),
                self.time_embedder(t),
                relperm,
                self._field_statistics(field),
                self._field_statistics(field_prediction),
            ),
            dim=1,
        )
        relperm_prediction = self.relperm_head(global_features)
        return field_prediction, relperm_prediction
