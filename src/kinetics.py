import numpy as np
import pandas as pd
import traceback
import warnings
from scipy.optimize import curve_fit, OptimizeWarning

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

# --- Hybrids ---
def hybrid_pso_avrami(t, M_eq1, k1, M_eq2, k2, n):
    return pso_model(t, M_eq1, k1) + avrami_model(t, M_eq2, k2, n)

def hybrid_langmuir_avrami(t, M_eq1, K1, M_eq2, k2, n):
    return langmuir_model(t, M_eq1, K1) + avrami_model(t, M_eq2, k2, n)

def hybrid_weibull_avrami(t, M_eq1, k1, n1, M_eq2, k2, n2):
    return weibull_model(t, M_eq1, k1, n1) + avrami_model(t, M_eq2, k2, n2)

def hybrid_weibull_langmuir(t, M_eq1, k1, n1, M_eq2, k2):
    return weibull_model(t, M_eq1, k1, n1) + langmuir_model(t, M_eq2, k2)

# Dictionary mapped to exactly match Paper 1 abbreviations
MODELS = {
    'P': {'func': pso_model, 'k': 2},
    'A': {'func': avrami_model, 'k': 3},
    'L': {'func': langmuir_model, 'k': 2},
    'LDF': {'func': ldf_model, 'k': 2},
    'F': {'func': fickian_model, 'k': 1},
    'W': {'func': weibull_model, 'k': 3},
    'PA': {'func': hybrid_pso_avrami, 'k': 5},
    'LA': {'func': hybrid_langmuir_avrami, 'k': 5},
    'WA': {'func': hybrid_weibull_avrami, 'k': 6},
    'WL': {'func': hybrid_weibull_langmuir, 'k': 5}
}

# Map showing which parameter index corresponds to M_eq for dynamic bounding
MEQ_INDICES = {
    'F': [], 'LDF': [0], 'L': [0], 'P': [0], 'A': [0], 'W': [0],
    'PA': [0, 2], 'LA': [0, 2], 'WL': [0, 3], 'WA': [0, 3]
}

# Map showing which parameter index corresponds to an exponent (n) to prevent overflows
EXP_INDICES = {
    'F': [], 'LDF': [], 'L': [], 'P': [], 'A': [2], 'W': [2],
    'PA': [4], 'LA': [4], 'WL': [2], 'WA': [2, 5]
}

# ==========================================
# 2. THE FITTER & BAYESIAN SELECTION LOGIC
# ==========================================

def fit_stage_kinetics(time_array, mass_array, m_eq_guess=None, k_guess=None):
    """
    Fits all candidate models to a single stage's time/mass data.
    Computes parameter errors via covariance, calculates BIC, and ranks models.
    Returns a dictionary suitable for high-throughput pipeline ingestion.
    """
    # Suppress non-fatal solver warnings to keep terminal clean
    warnings.simplefilter("ignore", OptimizeWarning)
    warnings.simplefilter("ignore", RuntimeWarning)
    
    # Force minimal time padding to prevent log(0) or 1/t crashing
    time_array = np.maximum(np.array(time_array, dtype=float), 1e-10)
    mass_array = np.array(mass_array, dtype=float)
    
    n_points = len(time_array)
    max_mass = np.max(mass_array) if len(mass_array) > 0 else 1.0
    
    # Safely unpack dynamic pipeline guesses
    meq = m_eq_guess if m_eq_guess is not None else max_mass
    kg = k_guess if k_guess is not None else 0.1

    results = []

    for name, meta in MODELS.items():
        k_param = meta['k']
        
        # 1. Dynamic Guesses based on model architecture
        if name == 'F': p0 = [kg]
        elif name in ['LDF', 'L', 'P']: p0 = [meq, kg]
        elif name in ['A', 'W']: p0 = [meq, kg, 1.0]
        elif name in ['PA', 'LA']: p0 = [meq/2, kg, meq/2, kg, 1.0]
        elif name == 'WL': p0 = [meq/2, kg, 1.0, meq/2, kg]
        elif name == 'WA': p0 = [meq/2, kg, 1.0, meq/2, kg, 1.0]
        else: p0 = [1.0] * k_param

        # 2. Physical Guardrails (Bounding)
        lower_bounds = [1e-7] * k_param
        upper_bounds = [np.inf] * k_param
        
        # Allow M_eq to freely swing positive or negative for Sorption/Desorption
        for idx in MEQ_INDICES.get(name, []):
            lower_bounds[idx] = -np.inf
            upper_bounds[idx] = np.inf
            
        # Cap exponents (n) to 4.0 to physically prevent np.power math overflows and solver hangs
        for idx in EXP_INDICES.get(name, []):
            upper_bounds[idx] = 4.0

        bounds = (lower_bounds, upper_bounds)
            
        try:
            # Added ftol and xtol, and reduced maxfev to strictly prevent solver infinite loops
            popt, pcov = curve_fit(
                meta['func'], time_array, mass_array, 
                p0=p0, bounds=bounds, 
                maxfev=2000, ftol=1e-5, xtol=1e-5
            )
            
            predictions = meta['func'](time_array, *popt)
            rss = np.sum((mass_array - predictions)**2)
            if rss <= 0: rss = 1e-15
            
            # Bayesian Information Criterion
            bic = n_points * np.log(rss / n_points) + k_param * np.log(n_points)
            
            results.append({
                'Model': name,
                'BIC': bic,
                'RSS': rss
            })
            
        except Exception:
            # Model failed to converge, move onto the next candidate
            continue

    if not results:
        return {'best_model': 'Fit Failed', 'delta_bic': np.nan, 'df': pd.DataFrame()}

    # Rank results by Delta BIC
    df_results = pd.DataFrame(results)
    min_bic = df_results['BIC'].min()
    df_results['Delta_BIC'] = df_results['BIC'] - min_bic
    df_results = df_results.sort_values('Delta_BIC').reset_index(drop=True)

    return {
        'best_model': df_results.iloc[0]['Model'],
        'delta_bic': df_results.iloc[1]['Delta_BIC'] if len(df_results) > 1 else np.nan,
        'df': df_results
    }