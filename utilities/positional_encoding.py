import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    """
    Implements positional encoding for input data, which adds sinusoidal and cosinusoidal
    functions of varying frequencies to the input to encode positional information.

    Args:
        input_dim (int): The dimensionality of the input data.
        num_frequencies (int, optional): The number of frequency bands to use for encoding. Default is 10.
    """
    def __init__(self, input_dim, num_frequencies=10):
        super(PositionalEncoding, self).__init__()
        self.num_frequencies = num_frequencies
        self.out_dim = input_dim + 2 * input_dim * num_frequencies

    def forward(self, x):
        """
        Applies positional encoding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: The encoded tensor with positional information added,
                          of shape (batch_size, out_dim).
        """
        encodings = [x]
        for i in range(self.num_frequencies):
            for func in [torch.sin, torch.cos]:
                encodings.append(func((2.0 ** i) * x))
        return torch.cat(encodings, dim=-1)

class FourierFeatureTransform(nn.Module):
    """
    Implements a Fourier Feature Transform for input data, which maps the input
    to a higher-dimensional space using random Fourier features.

    Args:
        num_input_channels (int): The number of input channels in the data.
        mapping_size (int, optional): The size of the Fourier feature mapping. Default is 256.
        scale (float, optional): The scaling factor for the random Fourier features. Default is 10.
    """
    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super(FourierFeatureTransform, self).__init__()
        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        B = torch.randn((num_input_channels, mapping_size)) * scale
        B_sort = sorted(B, key=lambda x: torch.norm(x, p=2))
        self._B = torch.stack(B_sort)

    def forward(self, x):
        """
        Applies the Fourier Feature Transform to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, num_input_channels).

        Returns:
            torch.Tensor: The transformed tensor of shape
                          (batch_size, num_input_channels + 2 * mapping_size).
        """
        batches, channels = x.shape
        assert channels == self._num_input_channels, (
            f"Expected input to have {self._num_input_channels} channels (got {channels} channels)"
        )

        res = x @ self._B.to(x.device)
        res = 2 * np.pi * res
        return torch.cat([x, torch.sin(res), torch.cos(res)], dim=1)

class LocalPositionalEncoding(nn.Module):
    """
    Implements a Local Positional Encoding (LPE) mechanism for 3D input data.
    This encoding maps global coordinates to a local grid and computes positional
    encodings using sinusoidal functions modulated by latent coefficients.

    Args:
        grid_resolution (int, optional): The resolution of the 3D grid. Default is 16.
        num_frequencies (int, optional): The number of frequency bands for encoding. Default is 10.
    """
    def __init__(self, grid_resolution=16, num_frequencies=10):
        super(LocalPositionalEncoding, self).__init__()
        self.grid_resolution = grid_resolution
        self.num_frequencies = num_frequencies
        self.grid_size = (grid_resolution, grid_resolution, grid_resolution)

        # Latent coefficients for the grid
        self.latent_grid = nn.Parameter(torch.randn(*self.grid_size, 2 * num_frequencies) * 0.01)

    def forward(self, x):
        """
        Computes the Local Positional Encoding (LPE) for the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3), where each row
                              represents a 3D coordinate normalized to [0, 1].

        Returns:
            torch.Tensor: The computed LPE tensor of shape (batch_size, num_frequencies * 2).
        """
        # Map global coordinates to local grid coordinates
        grid_size = torch.tensor(self.grid_size, device=x.device, dtype=torch.float32)
        x_local = x * grid_size

        # Compute cell indices and local position
        cell_idx = torch.floor(x_local).long()
        local_pos = x_local - cell_idx.float()

        # Compute latent coefficients for the cells
        APE = self.latent_grid[
            cell_idx[:, 0].clamp(0, self.grid_resolution - 1),
            cell_idx[:, 1].clamp(0, self.grid_resolution - 1),
            cell_idx[:, 2].clamp(0, self.grid_resolution - 1)
        ]

        # Compute Local Positional Encoding (LPE)
        frequencies = torch.linspace(1, self.num_frequencies, self.num_frequencies, device=x.device)
        encodings = []
        for i, freq in enumerate(frequencies):
            # Broadcast local_pos to match APE's dimensions
            cos_enc = torch.cos(2 * np.pi * freq * local_pos)
            sin_enc = torch.sin(2 * np.pi * freq * local_pos)

            # Multiply with respective APE values
            encodings.append(cos_enc * APE[..., i:i + 1])
            encodings.append(sin_enc * APE[..., i + self.num_frequencies:i + self.num_frequencies + 1])

        # Concatenate encodings
        lpe = torch.cat(encodings, dim=-1)
        return lpe

class NGPHashEncoding(nn.Module):
    """
    Implements a Neural Graphics Primitives (NGP) Hash Encoding for 3D input data.
    This encoding uses a multi-level hash table to map 3D coordinates to high-dimensional
    feature vectors, enabling efficient representation of spatial data.

    Args:
        input_dim (int, optional): The dimensionality of the input data. Default is 3.
        n_levels (int, optional): The number of levels in the hash table. Default is 16.
        n_features_per_level (int, optional): The number of features per level in the hash table. Default is 2.
        log2_hashmap_size (int, optional): The logarithm (base 2) of the hash map size. Default is 19.
    """
    def __init__(self,
                 input_dim=3,
                 n_levels=16,
                 n_features_per_level=2,
                 log2_hashmap_size=19):
        super(NGPHashEncoding, self).__init__()
        self.hash_table = HashEmbedder3D(
            grid_size=512,
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size
        )
        self.out_dim = n_levels * n_features_per_level

    def forward(self, x):
        """
        Applies the NGP Hash Encoding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, input_dim), where each row
                              represents a 3D coordinate normalized to [0, 1].

        Returns:
            torch.Tensor: The encoded tensor of shape (batch_size, out_dim), containing the
                          concatenated feature vectors from all hash table levels.
        """
        return self.hash_table(x)

class HashEmbedder3D(nn.Module):
    """
    Reimplementation of the hash encoder from:
        - HashNerf: https://github.com/yashbhalgat/HashNeRF-pytorch/blob/main/hash_encoding.py

    to suit the 3D image fitting scenario.

    """
    def __init__(self, grid_size=512, n_levels=16, n_features_per_level=2, log2_hashmap_size=19):
        super(HashEmbedder3D, self).__init__()
        self.grid_size = grid_size
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hashmap_size = 2 ** log2_hashmap_size

        self.embeddings = nn.ModuleList([
            nn.Embedding(self.hashmap_size, n_features_per_level)
            for _ in range(n_levels)
        ])

        for emb in self.embeddings:
            nn.init.uniform_(emb.weight, a=-0.0001, b=0.0001)

    def hash(self, coords):
        primes = [1, 2654435761, 805459861]
        x, y, z = coords.unbind(-1)
        return ((x * primes[0]) ^ (y * primes[1]) ^ (z * primes[2])) % self.hashmap_size

    def forward(self, x):
        # x: (B, 3), assumed normalized to [0, 1]
        x = x * self.grid_size
        x_floor = torch.floor(x).int()
        hashes = self.hash(x_floor)

        features = [emb(hashes) for emb in self.embeddings]
        return torch.cat(features, dim=-1)

