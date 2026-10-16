"""Native MAP-response architecture; initialisation is performed at run time."""
import torch
from torch import nn

class NativeMAPResponseHead(nn.Module):
    """Smooth patient-specific quadratic dose response around predicted usual MAP."""

    def __init__(self, features: int, width: int = 64):
        super().__init__()
        self.state = nn.Sequential(
            nn.Linear(features, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(width, width // 2), nn.GELU(),
        )
        self.map_mean = nn.Linear(width // 2, 1)
        self.parameters_head = nn.Linear(width // 2, 6)
        nn.init.zeros_(self.parameters_head.weight)
        nn.init.zeros_(self.parameters_head.bias)

    def components(self, x: torch.Tensor):
        hidden = self.state(x)
        expected_map = 75.0 + 20.0 * torch.tanh(self.map_mean(hidden).squeeze(1))
        raw = self.parameters_head(hidden).reshape(-1, 2, 3)
        offset = 0.75 * torch.tanh(raw[:, :, 0])
        slope = 1.50 * torch.tanh(raw[:, :, 1])
        curvature = 0.75 * torch.tanh(raw[:, :, 2])
        return expected_map, offset, slope, curvature

    def forward(
        self,
        x: torch.Tensor,
        base_logit: torch.Tensor,
        map_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        expected_map, offset, slope, curvature = self.components(x)
        dose = torch.clamp((map_value - expected_map) / 15.0, -2.0, 2.0)
        response = offset + slope * dose[:, None] + curvature * dose[:, None].square()
        return base_logit + response, expected_map, slope, curvature
