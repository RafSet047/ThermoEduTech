import random
import torch
import numpy as np
from typing import Optional, Tuple, List

class Augmentation:
    def __init__(self, jitter_strength=0.01, scaling_strength=0.1, time_warp_strength=0.2):
        self.jitter_strength = jitter_strength
        self.scaling_strength = scaling_strength
        self.time_warp_strength = time_warp_strength

    def jitter(self, data: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(data) * self.jitter_strength
        return data + noise

    def scaling(self, data: torch.Tensor) -> torch.Tensor:
        scaling_factor = torch.randn(data.size(0), 1) * self.scaling_strength + 1.0
        return data * torch.unsqueeze(scaling_factor, dim=-1)

    def time_warp(self, data: torch.Tensor) -> torch.Tensor:
        time_steps = torch.arange(data.size(0), dtype=torch.float32)
        random_warp = torch.randn_like(time_steps) * self.time_warp_strength
        time_steps += random_warp
        sorted_indices = torch.argsort(time_steps)
        return data[sorted_indices]

    def permutation(self, data: torch.Tensor) -> torch.Tensor:
        perm = torch.randperm(data.size(0))
        return data[perm]

    def window_slicing(self, data: torch.Tensor, slice_ratio=0.9) -> torch.Tensor:
        slice_len = int(data.size(0) * slice_ratio)
        start_idx = np.random.randint(0, data.size(0) - slice_len)
        return data[start_idx:start_idx + slice_len]

    def window_warping(self, data: torch.Tensor, warp_ratio=0.1) -> torch.Tensor:
        data_len = data.size(0)
        warp_len = int(data_len * warp_ratio)
        start_idx = np.random.randint(0, data_len - warp_len)
        end_idx = start_idx + warp_len
        scale = torch.rand(1).item() * 0.5 + 0.5
        data[start_idx:end_idx] = data[start_idx:end_idx] * scale
        return data

    def magnitude_warping(self, data: torch.Tensor) -> torch.Tensor:
        warp = torch.randn(data.size(0), 1) * 0.1 + 1.0
        return data * torch.unsqueeze(warp, dim=-1)

    def frequency_transformation(self, data: torch.Tensor) -> torch.Tensor:
        fft_data = torch.fft.fft(data, dim=0)
        amplitude = torch.abs(fft_data)
        phase = torch.angle(fft_data)
        transformed_amplitude = amplitude * (torch.randn_like(amplitude) * 0.1 + 1.0)
        transformed_data = torch.fft.ifft(transformed_amplitude * torch.exp(1j * phase), dim=0)
        return transformed_data.real

    def noise_injection(self, data: torch.Tensor, noise_level=0.01) -> torch.Tensor:
        noise = torch.randn_like(data) * noise_level
        return data + noise

    def reverse_time(self, data: torch.Tensor) -> torch.Tensor:
        return data.flip(0)

    def apply_augmentation(self, data: torch.Tensor) -> torch.Tensor:
        # Example application: Jitter and Scaling
        callable_methods = [
            self.jitter,
            self.scaling,
            self.time_warp,
            self.permutation,
            self.magnitude_warping,
            self.frequency_transformation,
            self.noise_injection,
            self.reverse_time
        ]
        aug_fn = random.choice(callable_methods)
        return aug_fn(data)

if __name__ == "__main__":
    N = 32
    F = 18
    S = 12
    
    x = torch.randn(N, S, F)

    aug = Augmentation()
    
    print(aug.jitter(x).shape)
    print(aug.scaling(x).shape)
    print(aug.time_warp(x).shape)
    print(aug.permutation(x).shape)
    print(aug.magnitude_warping(x).shape)
    print(aug.frequency_transformation(x).shape)
    print(aug.noise_injection(x).shape)
    print(aug.reverse_time(x).shape)

    print(aug.apply_augmentation(x).shape)