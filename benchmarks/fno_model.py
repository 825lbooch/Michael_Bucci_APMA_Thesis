"""
Fourier Neural Operator (FNO) baseline for antenna S11 prediction.

Architecture: Geometry -> Lift -> Fourier Layers -> Project -> S11(f)

The FNO learns in Fourier space, which is well-suited for smooth frequency responses.
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import List, Tuple, NamedTuple
from functools import partial


class FNOConfig(NamedTuple):
    """Static configuration for FNO (not part of gradient computation)."""
    n_params: int = 6
    n_freq: int = 500
    hidden_dim: int = 64
    n_modes: int = 32
    n_layers: int = 4


def complex_glorot_init(key, shape):
    """Initialize complex weights with Glorot/Xavier initialization."""
    fan_in, fan_out = shape[-2], shape[-1]
    scale = jnp.sqrt(2.0 / (fan_in + fan_out))
    real = random.normal(key, shape) * scale
    key, subkey = random.split(key)
    imag = random.normal(subkey, shape) * scale
    return real + 1j * imag


def init_fno_params(key, config: FNOConfig = None, **kwargs):
    """
    Initialize FNO parameters.

    Args:
        key: JAX random key
        config: FNOConfig namedtuple (or pass kwargs)

    Returns:
        Dictionary of trainable parameters (no config inside)
    """
    if config is None:
        config = FNOConfig(**kwargs)

    n_params = config.n_params
    n_freq = config.n_freq
    hidden_dim = config.hidden_dim
    n_modes = config.n_modes
    n_layers = config.n_layers

    params = {}

    # Geometry encoder: 6 -> hidden_dim
    key, *subkeys = random.split(key, 4)
    params["encoder"] = {
        "W1": random.normal(subkeys[0], (n_params, hidden_dim)) * 0.1,
        "b1": jnp.zeros(hidden_dim),
        "W2": random.normal(subkeys[1], (hidden_dim, hidden_dim)) * 0.1,
        "b2": jnp.zeros(hidden_dim),
    }

    # Frequency embedding (learnable positional encoding)
    key, subkey = random.split(key)
    params["freq_embed"] = random.normal(subkey, (n_freq, hidden_dim)) * 0.1

    # Fourier layers
    params["fourier_layers"] = []
    for i in range(n_layers):
        key, k1, k2, k3 = random.split(key, 4)
        layer = {
            # Spectral weights (complex): (n_modes, hidden_dim, hidden_dim)
            "R": complex_glorot_init(k1, (n_modes, hidden_dim, hidden_dim)),
            # Pointwise linear (bypass)
            "W": random.normal(k2, (hidden_dim, hidden_dim)) * 0.1,
            "b": jnp.zeros(hidden_dim),
        }
        params["fourier_layers"].append(layer)

    # Output projection: hidden_dim -> 2 (real and imaginary)
    key, subkey = random.split(key)
    params["decoder"] = {
        "W1": random.normal(subkey, (hidden_dim, hidden_dim)) * 0.1,
        "b1": jnp.zeros(hidden_dim),
        "W2": random.normal(key, (hidden_dim, 2)) * 0.1,
        "b2": jnp.zeros(2),
    }

    return params, config


def spectral_conv_1d(x, R, n_modes):
    """
    1D spectral convolution.

    Args:
        x: Input tensor, shape (batch, n_freq, hidden_dim)
        R: Complex spectral weights, shape (n_modes, hidden_dim, hidden_dim)
        n_modes: Number of modes to keep (static int)

    Returns:
        Output tensor, shape (batch, n_freq, hidden_dim)
    """
    batch, n_freq, hidden = x.shape

    # FFT along frequency dimension
    x_ft = jnp.fft.rfft(x, axis=1)  # (batch, n_freq//2+1, hidden)

    # Truncate to n_modes (n_modes must be static!)
    x_ft_trunc = jax.lax.dynamic_slice(x_ft, (0, 0, 0), (batch, n_modes, hidden))

    # Complex multiplication with spectral weights
    out_ft = jnp.einsum("bmi,mio->bmo", x_ft_trunc, R)

    # Pad back to full frequency
    pad_size = n_freq // 2 + 1 - n_modes
    out_ft_full = jnp.pad(out_ft, ((0, 0), (0, pad_size), (0, 0)))

    # Inverse FFT
    out = jnp.fft.irfft(out_ft_full, n=n_freq, axis=1)

    return out.real


def fourier_layer(x, layer_params, n_modes):
    """
    Single Fourier layer: spectral conv + pointwise linear + GELU.

    Args:
        x: Input, shape (batch, n_freq, hidden_dim)
        layer_params: Dict with 'R', 'W', 'b'
        n_modes: Number of Fourier modes (static int)

    Returns:
        Output, shape (batch, n_freq, hidden_dim)
    """
    # Spectral convolution path
    x_spectral = spectral_conv_1d(x, layer_params["R"], n_modes)

    # Pointwise linear path (bypass)
    x_linear = jnp.einsum("bfi,io->bfo", x, layer_params["W"]) + layer_params["b"]

    # Combine and activate
    out = jax.nn.gelu(x_spectral + x_linear)

    return out


def fno_forward(params, v, config: FNOConfig):
    """
    FNO forward pass.

    Args:
        params: Model parameters (trainable only)
        v: Geometry parameters, shape (batch, 6)
        config: FNOConfig with static hyperparameters

    Returns:
        S11 predictions, shape (batch, n_freq, 2) for real and imaginary parts
    """
    batch = v.shape[0]
    n_freq = config.n_freq
    n_modes = config.n_modes

    # Encode geometry: (batch, 6) -> (batch, hidden_dim)
    enc = params["encoder"]
    h = jax.nn.gelu(jnp.dot(v, enc["W1"]) + enc["b1"])
    h = jax.nn.gelu(jnp.dot(h, enc["W2"]) + enc["b2"])  # (batch, hidden_dim)

    # Broadcast to frequency dimension and add positional encoding
    h = h[:, None, :] + params["freq_embed"][None, :, :]

    # Apply Fourier layers
    for layer_params in params["fourier_layers"]:
        h = fourier_layer(h, layer_params, n_modes)

    # Decode: (batch, n_freq, hidden_dim) -> (batch, n_freq, 1)
    dec = params["decoder"]
    out = jax.nn.gelu(jnp.einsum("bfi,io->bfo", h, dec["W1"]) + dec["b1"])
    out = jnp.einsum("bfi,io->bfo", out, dec["W2"]) + dec["b2"]

    return out


def make_fno_forward(config: FNOConfig):
    """Create a forward function with config baked in (for use with grad)."""
    @partial(jax.jit)
    def forward(params, v):
        return fno_forward(params, v, config)
    return forward


def count_params(params):
    """Count total number of parameters."""
    total = 0

    # Encoder
    for key in ["W1", "b1", "W2", "b2"]:
        total += params["encoder"][key].size

    # Frequency embedding
    total += params["freq_embed"].size

    # Fourier layers
    for layer in params["fourier_layers"]:
        total += layer["R"].size * 2  # Complex = 2x real
        total += layer["W"].size
        total += layer["b"].size

    # Decoder
    for key in ["W1", "b1", "W2", "b2"]:
        total += params["decoder"][key].size

    return total


if __name__ == "__main__":
    # Test the model
    key = random.PRNGKey(42)
    config = FNOConfig(n_params=6, n_freq=500, hidden_dim=64, n_modes=32, n_layers=4)
    params, config = init_fno_params(key, config)

    print(f"FNO Parameter count: {count_params(params):,}")

    # Test forward pass
    forward_fn = make_fno_forward(config)
    v_test = random.normal(key, (4, 6))
    out = forward_fn(params, v_test)
    print(f"Input shape: {v_test.shape}")
    print(f"Output shape: {out.shape}")

    # Test gradient
    def loss_fn(params):
        return jnp.mean(forward_fn(params, v_test) ** 2)

    grads = jax.grad(loss_fn)(params)
    print("Gradient computation: OK")
