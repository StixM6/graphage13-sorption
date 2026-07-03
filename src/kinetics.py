import numpy as np
import pandas as pd
import traceback
from scipy.optimize import curve_fit

# ==========================================
# 1. THE KINETIC MODEL LIBRARY
# ==========================================

def pso_model(t, M_eq, k):
    """Pseudo-Second-Order (PSO)"""
    return (M_eq**2 * k * t) / (1 + M_eq * k * t)

def avrami_model(t, M_eq, k, n):
    """Avrami"""
    return M_eq * (1 - np.exp(-k * np.power(t, n)))

def langmuir_model(t, M_eq, K):
    """Langmuir"""
    return (M_eq * K * t) / (1 + K * t)

def ldf_model(t, M_eq, k):
    """Linear Driving Force (LDF)"""
    return M_eq * (1 - np.exp(-k * t))

def fickian_model(t, k):
    """Fickian / Intra-particle Diffusion"""
    return k * np.sqrt(t)

def weibull_model(t, M_eq, k, n):
    """Weibull (Statistical spread of energies)"""
    return M_eq * (1 - np.exp(-np.power(k * t, n)))

# Hybrid Examples
def hybrid_pso_avrami(t, M_eq1, k1, M_eq2, k2, n):
    return pso_model(t, M_eq1, k1) + avrami_model(t, M_eq2, k2, n)

def hybrid_langmuir_avrami(t, M_eq1, K1, M_eq2, k2, n):
    return langmuir_model(t, M_eq1, K1) + avrami_model(t, M_eq2, k2, n)


# Dictionary mapping names to functions and parameter counts (k)
MODELS = {
    'PSO': {'func': pso_model, 'k': 2},
    'Avrami': {'func': avrami_model, 'k': 3},
    'Langmuir': {'func': langmuir_model, 'k': 2},
    'LDF': {'func': ldf_model, 'k': 2},
    'Fickian': {'func': fickian_model, 'k': 1},
    'Weibull': {'func': weibull_model, 'k': 3},
    'PSO-Avrami': {'func': hybrid_pso_avrami, 'k': 5},
    'Langmuir-Avrami': {'func': hybrid_langmuir_avrami, 'k': 5}
}

# ==========================================
# 2. THE FITTER & BAYESIAN SELECTION LOGIC
# ==========================================

def fit_stage_kinetics(time_array, mass_array):
    """
    Fits all candidate models to a single stage's time/mass data.
    Computes parameter errors via covariance, calculates BIC, and ranks models.
    """
    # Ensure inputs are clean numpy arrays
    time_array = np.array(time_array, dtype=float)
    mass_array = np.array(mass_array, dtype=float)
    
    n_points = len(time_array)
    max_mass = np.max(mass_array) if len(mass_array) > 0 else 1.0
    results = []

    for name, meta in MODELS.items():
        k_param = meta['k']
        
        # Dynamic initial guesses based on parameter count
        if k_param == 1:
            p0 = [1.0]
        elif k_param == 2:
            p0 = [max_mass, 0.1]
        elif k_param == 3:
            p0 = [max_mass, 0.1, 1.0]
        elif k_param == 5:
            p0 = [max_mass/2, 0.1, max_mass/2, 0.1, 1.0]
            
        # Define explicit bounds matching the exact parameter count to avoid curve_fit broadcast issues
        bounds = ([0.0] * k_param, [np.inf] * k_param)
            
        try:
            # curve_fit returns optimal parameters (popt) and the covariance matrix (pcov)
            popt, pcov = curve_fit(
                meta['func'], time_array, mass_array, 
                p0=p0, bounds=bounds, maxfev=10000
            )
            
            # Extract standard errors from the diagonal of the covariance matrix
            perr = np.sqrt(np.diag(pcov))
            
            # Calculate Residual Sum of Squares (RSS)
            predictions = meta['func'](time_array, *popt)
            rss = np.sum((mass_array - predictions)**2)
            
            # Safeguard against RSS being perfectly 0 (which would break log)
            if rss <= 0:
                rss = 1e-15
            
            # Calculate BIC
            bic = n_points * np.log(rss / n_points) + k_param * np.log(n_points)
            
            # Formatting parameters and uncertainties into strings for clean output
            param_str = ", ".join([f"{val:.4f}±{err:.4f}" for val, err in zip(popt, perr)])
            
            # Extract Avrami exponent (n) if applicable
            avrami_n = None
            if 'Avrami' in name:
                avrami_n = popt[-1] # n is always the last parameter in our definitions
                
            results.append({
                'Model': name,
                'BIC': bic,
                'RSS': rss,
                'Parameters (±SE)': param_str,
                'Avrami n': round(avrami_n, 3) if avrami_n else None,
                'popt': popt
            })
            
        except Exception as e:
            # Model failed to converge (silently skip or print for debugging)
            continue

    if not results:
        return pd.DataFrame()

    # Create DataFrame and calculate Delta BIC
    df_results = pd.DataFrame(results)
    min_bic = df_results['BIC'].min()
    df_results['Delta_BIC'] = df_results['BIC'] - min_bic
    
    # Sort by best fit (lowest BIC)
    df_results = df_results.sort_values('Delta_BIC').reset_index(drop=True)
    
    # Flag statistical ties (Delta BIC < 6 is a good standard rule of thumb)
    df_results['Statistical Tie'] = df_results['Delta_BIC'] < 6.0

    return df_results