"""
1D U-Net baseline for antenna S11 prediction.

Architecture: Geometry -> Encoder -> Bottleneck -> Decoder (with skip connections) -> S11(f)

Adapted for the operator learning setting where we map geometry params to frequency response.
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import List, Dict, NamedTuple
from functools import partial


class UNetConfig(NamedTuple):
    """Static configuration for U-Net."""
    n_params: int = 6
    n_freq: int = 500
    base_channels: int = 32
    n_levels: int = 3


def init_conv1d(key, in_channels, out_channels, kernel_size):
    """Initialize 1D convolution weights."""
    fan_in = in_channels * kernel_size
    fan_out = out_channels * kernel_size
    scale = jnp.sqrt(2.0 / (fan_in + fan_out))
    W = random.normal(key, (kernel_size, in_channels, out_channels)) * scale
    b = jnp.zeros(out_channels)
    return {"W": W, "b": b}


def conv1d(x, params, stride=1, padding="SAME"):
    """1D convolution."""
    out = jax.lax.conv_general_dilated(
        x,
        params["W"],
        window_strides=(stride,),
        padding=padding,
        dimension_numbers=("NWC", "WIO", "NWC")
    )
    return out + params["b"]


def init_unet_params(key, config: UNetConfig = None, **kwargs):
    """
    Initialize 1D U-Net parameters.

    Returns:
        params: Dictionary of trainable parameters
        config: UNetConfig with static hyperparameters
    """
    if config is None:
        config = UNetConfig(**kwargs)

    n_params = config.n_params
    n_freq = config.n_freq
    base_channels = config.base_channels
    n_levels = config.n_levels

    params = {}

    # Geometry encoder: 6 -> latent_dim
    latent_dim = base_channels * (2 ** n_levels)
    key, k1, k2 = random.split(key, 3)
    params["geo_encoder"] = {
        "W1": random.normal(k1, (n_params, 128)) * 0.1,
        "b1": jnp.zeros(128),
        "W2": random.normal(k2, (128, latent_dim)) * 0.1,
        "b2": jnp.zeros(latent_dim),
    }

    # Initial projection to frequency dimension
    key, subkey = random.split(key)
    params["freq_project"] = {
        "W": random.normal(subkey, (latent_dim, n_freq * base_channels)) * 0.01,
        "b": jnp.zeros(n_freq * base_channels),
    }

    # Encoder blocks (downsampling path)
    params["encoder"] = []
    in_ch = base_channels
    for level in range(n_levels):
        out_ch = base_channels * (2 ** (level + 1))
        key, k1, k2 = random.split(key, 3)
        block = {
            "conv1": init_conv1d(k1, in_ch, out_ch, kernel_size=3),
            "conv2": init_conv1d(k2, out_ch, out_ch, kernel_size=3),
        }
        params["encoder"].append(block)
        in_ch = out_ch

    # Bottleneck
    key, k1, k2 = random.split(key, 3)
    params["bottleneck"] = {
        "conv1": init_conv1d(k1, in_ch, in_ch * 2, kernel_size=3),
        "conv2": init_conv1d(k2, in_ch * 2, in_ch, kernel_size=3),
    }

    # Decoder blocks (upsampling path with skip connections)
    params["decoder"] = []
    for level in range(n_levels - 1, -1, -1):
        skip_ch = base_channels * (2 ** (level + 1))
        out_ch = base_channels * (2 ** level) if level > 0 else base_channels
        in_ch_dec = in_ch + skip_ch  # Concatenated with skip

        key, k1, k2, k3 = random.split(key, 4)
        block = {
            "upsample": init_conv1d(k1, in_ch, in_ch, kernel_size=3),
            "conv1": init_conv1d(k2, in_ch_dec, out_ch, kernel_size=3),
            "conv2": init_conv1d(k3, out_ch, out_ch, kernel_size=3),
        }
        params["decoder"].append(block)
        in_ch = out_ch

    # Output head: 2 channels for real and imaginary
    key, subkey = random.split(key)
    params["output"] = init_conv1d(subkey, base_channels, 2, kernel_size=1)

    return params, config


def encoder_block(x, params):
    """Encoder block: conv -> relu -> conv -> relu -> maxpool."""
    h = jax.nn.relu(conv1d(x, params["conv1"]))
    h = jax.nn.relu(conv1d(h, params["conv2"]))
    # Max pooling with stride 2
    pooled = jax.lax.reduce_window(
        h, -jnp.inf, jax.lax.max, (1, 2, 1), (1, 2, 1), "SAME"
    )
    return h, pooled


def decoder_block(x, skip, params):
    """Decoder block: upsample -> concat skip -> conv -> relu -> conv -> relu."""
    batch, length, channels = x.shape

    # Upsample by 2x
    upsampled = jax.image.resize(x, (batch, length * 2, channels), method="nearest")

    # Match skip connection size
    min_len = min(upsampled.shape[1], skip.shape[1])
    upsampled = upsampled[:, :min_len, :]
    skip = skip[:, :min_len, :]

    # Concatenate with skip connection
    h = jnp.concatenate([upsampled, skip], axis=-1)

    h = jax.nn.relu(conv1d(h, params["conv1"]))
    h = jax.nn.relu(conv1d(h, params["conv2"]))
    return h


def unet_forward(params, v, config: UNetConfig):
    """
    U-Net forward pass.

    Args:
        params: Model parameters
        v: Geometry parameters, shape (batch, 6)
        config: UNetConfig with static hyperparameters

    Returns:
        S11 predictions, shape (batch, n_freq, 2) for real and imaginary parts
    """
    n_freq = config.n_freq
    base_ch = config.base_channels
    batch = v.shape[0]

    # Encode geometry
    geo = params["geo_encoder"]
    h = jax.nn.relu(jnp.dot(v, geo["W1"]) + geo["b1"])
    h = jax.nn.relu(jnp.dot(h, geo["W2"]) + geo["b2"])

    # Project to frequency dimension
    proj = params["freq_project"]
    h = jnp.dot(h, proj["W"]) + proj["b"]
    h = h.reshape(batch, n_freq, base_ch)

    # Encoder path
    skips = []
    for enc_block in params["encoder"]:
        skip, h = encoder_block(h, enc_block)
        skips.append(skip)

    # Bottleneck
    h = jax.nn.relu(conv1d(h, params["bottleneck"]["conv1"]))
    h = jax.nn.relu(conv1d(h, params["bottleneck"]["conv2"]))

    # Decoder path
    for dec_block, skip in zip(params["decoder"], reversed(skips)):
        h = decoder_block(h, skip, dec_block)

    # Ensure output matches n_freq
    if h.shape[1] != n_freq:
        h = jax.image.resize(h, (batch, n_freq, h.shape[2]), method="linear")

    # Output projection
    out = conv1d(h, params["output"])

    return out


def make_unet_forward(config: UNetConfig):
    """Create a forward function with config baked in."""
    @partial(jax.jit)
    def forward(params, v):
        return unet_forward(params, v, config)
    return forward


def count_params(params):
    """Count total number of parameters."""
    total = 0

    def count_dict(d):
        count = 0
        for k, v in d.items():
            if isinstance(v, jnp.ndarray):
                count += v.size
            elif isinstance(v, dict):
                count += count_dict(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        count += count_dict(item)
        return count

    for key, val in params.items():
        if isinstance(val, dict):
            total += count_dict(val)
        elif isinstance(val, list):
            for item in val:
                total += count_dict(item)

    return total


if __name__ == "__main__":
    # Test the model
    key = random.PRNGKey(42)
    config = UNetConfig(n_params=6, n_freq=500, base_channels=32, n_levels=3)
    params, config = init_unet_params(key, config)

    print(f"U-Net Parameter count: {count_params(params):,}")

    # Test forward pass
    forward_fn = make_unet_forward(config)
    v_test = random.normal(key, (4, 6))
    out = forward_fn(params, v_test)
    print(f"Input shape: {v_test.shape}")
    print(f"Output shape: {out.shape}")

    # Test gradient
    def loss_fn(params):
        return jnp.mean(forward_fn(params, v_test) ** 2)

    grads = jax.grad(loss_fn)(params)
    print("Gradient computation: OK")
