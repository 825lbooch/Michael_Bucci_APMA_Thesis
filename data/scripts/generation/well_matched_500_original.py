"""
Physics-Informed 6D Antenna Dataset Generation
WELL-MATCHED ANTENNAS (500 samples)

FIXED VERSION (Dec 2025):
- Removed aggressive "Physics Corrections" that caused impedance mismatch.
- Reverted to standard Transmission Line Model for L and Inset.
- Maintains geometric validation for MATLAB meshing compatibility.

Target: S11 < -10dB (or at least < -8dB)
"""

import numpy as np
import pandas as pd
import os

# Physical Constants
c = 3e8  # Speed of light (m/s)
Z0_TARGET = 50.0  # Target impedance (ohms)

# =============================================================================
# GEOMETRIC CONSTRAINTS FOR MATLAB SIMULATION
# =============================================================================
# These match the OSCAR/MATLAB constraints to prevent meshing failures
MIN_FEED_WIDTH_MM = 0.55      
MIN_VIA_DIAMETER_MM = 0.2     
MAX_VIA_FEED_RATIO = 0.35     
MIN_SUBSTRATE_H_MM = 0.5      
MAX_SUBSTRATE_H_MM = 3.0      
MIN_EPS_R = 2.2               
MAX_EPS_R = 4.8               

def calculate_microstrip_width(Z0, h, eps_r):
    """
    Synthesizes the required feed width (w) to achieve a specific Z0 (50 ohms)
    given substrate height h and permittivity eps_r.
    """
    A = (Z0 / 60.0) * np.sqrt((eps_r + 1) / 2.0) + \
        ((eps_r - 1) / (eps_r + 1)) * (0.23 + 0.11 / eps_r)
    B = (377.0 * np.pi) / (2.0 * Z0 * np.sqrt(eps_r))
    
    w_h_ratio = 8 * np.exp(A) / (np.exp(2 * A) - 2)
    
    if w_h_ratio > 2:
        w_h_ratio = (2 / np.pi) * (
            B - 1 - np.log(2 * B - 1) + 
            (eps_r - 1) / (2 * eps_r) * (np.log(B - 1) + 0.39 - 0.61 / eps_r)
        )
    
    return w_h_ratio * h

def check_feed_width_valid(w_feed_mm):
    """
    Check if feed width is compatible with MATLAB's via constraints.
    """
    # MATLAB Logic: viaDia = min(0.4 * feedWidth, 0.8mm)
    via_dia_mm = min(0.4 * w_feed_mm, 0.8)
    via_dia_mm = max(via_dia_mm, MIN_VIA_DIAMETER_MM)
    
    # Check ratio
    ratio = via_dia_mm / w_feed_mm
    return ratio <= MAX_VIA_FEED_RATIO

def find_valid_substrate_params(f_target, max_attempts=50):
    """
    Find substrate parameters (h, eps_r) that yield a valid feed width.
    """
    for attempt in range(max_attempts):
        # Bias toward thicker substrates (better bandwidth, wider feeds)
        eps_r = np.random.uniform(MIN_EPS_R, 3.5) # Prefer lower eps_r for efficiency
        h_mm = np.random.uniform(0.8, MAX_SUBSTRATE_H_MM) 
        
        h = h_mm / 1000.0
        w_feed = calculate_microstrip_width(Z0_TARGET, h, eps_r)
        w_feed_mm = w_feed * 1000
        
        # Check constraints
        if w_feed_mm >= MIN_FEED_WIDTH_MM and check_feed_width_valid(w_feed_mm):
            return h_mm, eps_r, w_feed_mm
    
    # Safe Fallback
    return 1.6, 2.2, 4.9 # A generic valid config if random fails

def calculate_patch_dimensions_analytical(f_target, eps_r, h):
    """
    Calculate patch antenna dimensions using standard Transmission Line Model.
    REMOVED the 0.98 multiplier that was shifting frequency too high.
    """
    # 1. Width (W)
    W = (c / (2 * f_target)) * np.sqrt(2 / (eps_r + 1))
    
    # 2. Effective Dielectric Constant (eps_eff)
    eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 * (h / W)) ** (-0.5)
    
    # 3. Fringing Extension (dL)
    dL = 0.412 * h * ((eps_eff + 0.3) * (W / h + 0.264)) / \
         ((eps_eff - 0.258) * (W / h + 0.8))
    
    # 4. Effective Length (L)
    # L_eff = c / (2 * f * sqrt(eps_eff))
    # L = L_eff - 2*dL
    L_eff = c / (2 * f_target * np.sqrt(eps_eff))
    L = L_eff - 2 * dL
    
    return L, W, eps_eff

def calculate_inset_depth_analytical(L, W, eps_r, Z0=50.0):
    """
    Calculate inset depth for 50-ohm impedance matching.
    REMOVED the 1.25 multiplier that caused the -1.3dB mismatch.
    """
    # 1. Edge Resistance (R_in at edge)
    # Standard approximation
    R_edge = 90 * (eps_r ** 2 / (eps_r - 1)) * (L / W) ** 2
    
    # 2. Inset Depth (y0)
    # R_in(y0) = R_edge * cos^2(pi * y0 / L)
    # y0 = (L/pi) * arccos(sqrt(Z0 / R_edge))
    
    if Z0 >= R_edge:
        # If edge impedance is too low (rare), no inset needed, feed at edge
        return 0.0
        
    arg = np.sqrt(Z0 / R_edge)
    arg = np.clip(arg, 0, 1) # Safety clip
    inset_analytical = (L / np.pi) * np.arccos(arg)
    
    # 3. Small Jitter
    # Instead of a massive offset, we add +/- 2% jitter to create data diversity
    # without breaking the impedance match.
    jitter = np.random.uniform(0.98, 1.02)
    inset_final = inset_analytical * jitter
    
    return inset_final

def validate_geometry(L_mm, W_mm, inset_mm, feedWidth_mm):
    """
    Final validation to ensure geometry is physically constructible.
    """
    errors = []
    
    if inset_mm > 0.45 * L_mm:
        errors.append("Inset too deep (>45% of L)")
        
    if inset_mm < 0:
        errors.append("Negative inset")
        
    # Check if notch width (feed + gap) cuts the patch in half? 
    # Usually fine, but checking feed width > W would be bad.
    if feedWidth_mm > W_mm * 0.5:
        errors.append("Feed width > 50% of Patch Width")

    return len(errors) == 0

def generate_wellmatched_antennas(N=500, seed=42):
    np.random.seed(seed)
    
    data = []
    frequencies = []
    
    print(f"Generating {N} WELL-MATCHED samples (Standard Analytical Model)...")
    
    attempts = 0
    while len(data) < N:
        attempts += 1
        
        # 1. Target Freq (S-band: 2.0 - 3.5 GHz)
        f_target = np.random.uniform(2.0e9, 3.5e9)
        
        # 2. Substrate & Feed
        h_mm, eps_r, w_feed_mm = find_valid_substrate_params(f_target)
        h = h_mm / 1000.0
        
        # 3. Patch Dimensions (Standard Physics)
        L, W, eps_eff = calculate_patch_dimensions_analytical(f_target, eps_r, h)
        
        # 4. Inset Depth (Standard Physics + Tiny Jitter)
        inset = calculate_inset_depth_analytical(L, W, eps_r)
        
        # Convert to mm
        L_mm = L * 1000
        W_mm = W * 1000
        inset_mm = inset * 1000
        
        # 5. Validate
        if validate_geometry(L_mm, W_mm, inset_mm, w_feed_mm):
            data.append([L_mm, W_mm, inset_mm, w_feed_mm, h_mm, eps_r])
            frequencies.append(f_target / 1e9)
            
        if len(data) % 100 == 0:
            print(f"  ... {len(data)} generated")
            
    columns = ['L_mm', 'W_mm', 'inset_mm', 'feedWidth_mm', 'h_mm', 'eps_r']
    df = pd.DataFrame(data, columns=columns)
    df['f_target_GHz'] = frequencies
    
    return df

if __name__ == "__main__":
    df = generate_wellmatched_antennas(N=500)
    
    # Save for MATLAB
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'antenna_params_wellmatched_validated.csv')
    
    # Drop the frequency column for the MATLAB input file (it only needs the 6 geometries)
    df_matlab = df[['L_mm', 'W_mm', 'inset_mm', 'feedWidth_mm', 'h_mm', 'eps_r']]
    df_matlab.to_csv(output_path, index=False)
    
    # Save reference file with frequencies (for your own checking)
    ref_path = os.path.join(script_dir, 'antenna_params_wellmatched_validated_with_freq.csv')
    df.to_csv(ref_path, index=False)
    
    print("-" * 50)
    print(f"Successfully generated 500 samples.")
    print(f"Files saved to: {script_dir}")
    print("Corrections applied: Removed artificial 0.98x length factor and 1.25x inset factor.")
    print("-" * 50)