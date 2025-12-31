"""
Analytical S11 Model for Patch Antennas

Transmission Line Model approximation for use in residual learning.
The DeepONet learns to correct this analytical baseline.

References:
    Balanis, C.A., "Antenna Theory: Analysis and Design", 4th Ed., Ch. 14
    Pozar, D.M., "Microstrip Antennas", Proc. IEEE, 1992
"""

import jax.numpy as jnp
import numpy as np

# Physical constants
C0 = 3e8  # Speed of light (m/s)


def effective_permittivity(eps_r, h, W):
    """
    Effective dielectric constant for microstrip.

    Args:
        eps_r: Relative permittivity
        h: Substrate height (m)
        W: Patch width (m)

    Returns:
        Effective permittivity
    """
    return (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 * h / W) ** (-0.5)


def fringing_extension(eps_eff, h, W):
    """
    Fringing field extension (delta L).

    Args:
        eps_eff: Effective permittivity
        h: Substrate height (m)
        W: Patch width (m)

    Returns:
        Fringing extension (m)
    """
    return 0.412 * h * ((eps_eff + 0.3) * (W / h + 0.264)) / \
           ((eps_eff - 0.258) * (W / h + 0.8))


def resonant_frequency(L, W, h, eps_r):
    """
    Predicted resonant frequency using transmission line model.

    Args:
        L: Patch length (m)
        W: Patch width (m)
        h: Substrate height (m)
        eps_r: Relative permittivity

    Returns:
        Resonant frequency (Hz)
    """
    eps_eff = effective_permittivity(eps_r, h, W)
    dL = fringing_extension(eps_eff, h, W)
    L_eff = L + 2 * dL
    return C0 / (2 * L_eff * jnp.sqrt(eps_eff))


def edge_resistance(L, W, eps_r):
    """
    Edge input resistance (Balanis Eq. 14-7, approximate form).

    Args:
        L: Patch length (m)
        W: Patch width (m)
        eps_r: Relative permittivity

    Returns:
        Edge resistance (ohms)
    """
    return 90 * (eps_r ** 2 / (eps_r - 1)) * (L / W) ** 2


def input_impedance_at_inset(L, W, inset, eps_r, h, freq):
    """
    Input impedance at inset feed location.

    Uses cavity model: Z_in(y) = R_edge * cos^2(pi * y / L)
    with frequency-dependent reactance from transmission line.

    Args:
        L: Patch length (m)
        W: Patch width (m)
        inset: Inset depth (m)
        eps_r: Relative permittivity
        h: Substrate height (m)
        freq: Frequency (Hz)

    Returns:
        Complex input impedance (ohms)
    """
    # Resonant frequency
    f0 = resonant_frequency(L, W, h, eps_r)

    # Edge resistance
    R_edge = edge_resistance(L, W, eps_r)

    # Resistance at inset (cos^2 taper)
    R_in = R_edge * jnp.cos(jnp.pi * inset / L) ** 2

    # Q factor (approximate, accounts for radiation + substrate losses)
    eps_eff = effective_permittivity(eps_r, h, W)
    Q = C0 / (4 * f0 * h * jnp.sqrt(eps_eff))  # Radiation Q approximation
    Q = jnp.clip(Q, 10, 100)  # Reasonable bounds

    # Frequency detuning
    delta = (freq - f0) / f0

    # Simple RLC model for impedance vs frequency
    # Near resonance: Z ≈ R / (1 + j*2*Q*delta)
    denominator = 1 + 1j * 2 * Q * delta
    Z_in = R_in / denominator

    return Z_in


def analytical_s11(L, W, inset, feedWidth, h, eps_r, freq_GHz, Z0=50.0):
    """
    Analytical S11 prediction using transmission line model.

    Args:
        L: Patch length (mm)
        W: Patch width (mm)
        inset: Inset depth (mm)
        feedWidth: Feed line width (mm) - not used in simple model
        h: Substrate height (mm)
        eps_r: Relative permittivity
        freq_GHz: Frequency array (GHz)
        Z0: Reference impedance (ohms)

    Returns:
        S11 in dB (same shape as freq_GHz)
    """
    # Convert to SI units
    L_m = L / 1000
    W_m = W / 1000
    inset_m = inset / 1000
    h_m = h / 1000
    freq_Hz = freq_GHz * 1e9

    # Compute impedance at each frequency
    Z_in = input_impedance_at_inset(L_m, W_m, inset_m, eps_r, h_m, freq_Hz)

    # Reflection coefficient
    gamma = (Z_in - Z0) / (Z_in + Z0)

    # S11 in dB
    s11_mag = jnp.abs(gamma)
    s11_dB = 20 * jnp.log10(jnp.clip(s11_mag, 1e-6, 1.0))

    return s11_dB


def analytical_s11_batch(params, freq_GHz):
    """
    Batch analytical S11 for multiple antenna geometries.

    Args:
        params: Array of shape (N, 6) with columns [L, W, inset, feedWidth, h, eps_r] in mm
        freq_GHz: Frequency array of shape (N, F) or (F,)

    Returns:
        S11 in dB of shape (N, F)
    """
    N = params.shape[0]

    # Handle frequency array broadcasting
    if freq_GHz.ndim == 1:
        freq_GHz = jnp.tile(freq_GHz, (N, 1))

    # Vectorized computation
    L, W, inset, feedWidth, h, eps_r = params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4], params[:, 5]

    # Expand dims for broadcasting: (N, 1)
    L = L[:, None]
    W = W[:, None]
    inset = inset[:, None]
    h = h[:, None]
    eps_r = eps_r[:, None]

    # Compute analytical S11
    s11_dB = analytical_s11(L, W, inset, feedWidth, h, eps_r, freq_GHz)

    return s11_dB


# NumPy versions for preprocessing (non-JAX)
def analytical_s11_numpy(L, W, inset, feedWidth, h, eps_r, freq_GHz, Z0=50.0):
    """NumPy version for preprocessing."""
    # Convert to SI units
    L_m = L / 1000
    W_m = W / 1000
    inset_m = inset / 1000
    h_m = h / 1000
    freq_Hz = freq_GHz * 1e9

    # Effective permittivity
    eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 * h_m / W_m) ** (-0.5)

    # Fringing extension
    dL = 0.412 * h_m * ((eps_eff + 0.3) * (W_m / h_m + 0.264)) / \
         ((eps_eff - 0.258) * (W_m / h_m + 0.8))

    # Resonant frequency
    L_eff = L_m + 2 * dL
    f0 = C0 / (2 * L_eff * np.sqrt(eps_eff))

    # Edge resistance
    R_edge = 90 * (eps_r ** 2 / (eps_r - 1)) * (L_m / W_m) ** 2

    # Resistance at inset
    R_in = R_edge * np.cos(np.pi * inset_m / L_m) ** 2

    # Q factor
    Q = C0 / (4 * f0 * h_m * np.sqrt(eps_eff))
    Q = np.clip(Q, 10, 100)

    # Frequency detuning
    delta = (freq_Hz - f0) / f0

    # Impedance
    denominator = 1 + 1j * 2 * Q * delta
    Z_in = R_in / denominator

    # Reflection coefficient
    gamma = (Z_in - Z0) / (Z_in + Z0)
    s11_mag = np.abs(gamma)
    s11_dB = 20 * np.log10(np.clip(s11_mag, 1e-6, 1.0))

    return s11_dB


if __name__ == "__main__":
    # Quick test
    import matplotlib.pyplot as plt

    # Example antenna parameters (mm)
    L, W, inset, feedWidth, h, eps_r = 32.0, 40.0, 8.0, 3.0, 1.6, 2.2
    freq = np.linspace(1.5, 3.5, 500)

    s11 = analytical_s11_numpy(L, W, inset, feedWidth, h, eps_r, freq)

    plt.figure(figsize=(10, 4))
    plt.plot(freq, s11)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('S11 (dB)')
    plt.title('Analytical S11 (Transmission Line Model)')
    plt.grid(True, alpha=0.3)
    plt.axhline(-10, color='r', linestyle='--', label='-10 dB threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('analytical_s11_test.png', dpi=150)
    print("Saved test plot to analytical_s11_test.png")

    # Print predicted resonance
    f_res_idx = np.argmin(s11)
    print(f"Predicted resonance: {freq[f_res_idx]:.2f} GHz, S11 = {s11[f_res_idx]:.1f} dB")
