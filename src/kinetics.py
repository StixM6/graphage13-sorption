import numpy as np
import pandas as pd
import warnings
from scipy.optimize import curve_fit, OptimizeWarning

# ==========================================
# 1. THE KINETIC MODEL LIBRARY
# ==========================================

def fickian_model(t, k):
    """Fickian / intra-particle diffusion. Has no plateau (grows as sqrt(t) forever),
    so it can never win a fit against a full saturating stage. Fit only to the
    early-time portion of a stage (~first 60% of uptake) as a diagnostic of whether
    the initial regime is diffusion-controlled."""
    return k * np.sqrt(t)

def exp_model(t, M_eq, k):
    """Single-exponential relaxation to equilibrium: M_eq*(1-exp(-k*t)).
    This is Paper 1's actual Langmuir kinetics (dtheta/dt = ka*C*(1-theta) - kd*theta
    is a linear ODE that integrates to this form, not to a hyperbola) and it is also
    exactly the model previously coded as LDF, and the n=1 special case of
    stretched_exp_model below. 'L' and 'LDF' are the same function; this replaces both."""
    return M_eq * (1 - np.exp(-k * t))

def hyperbola_model(t, M_eq, k):
    """PSO hyperbola with a signed equilibrium displacement.

    For positive adsorption this is Paper 1's original integrated PSO equation:

        M(t) = M_eq**2 * k2 * t / (1 + M_eq * k2 * t)

    Here ``k`` is the physical PSO rate constant ``k2`` rather than a combined observed
    rate, so the equilibrium capacity remains explicit in the kinetic timescale. For a
    negative zero-based desorption trace, ``M_eq*abs(M_eq)`` supplies the curve's sign
    while ``abs(M_eq)`` keeps the denominator positive. This signed extension is exactly
    the published equation whenever ``M_eq`` is positive and avoids a false pole when it
    is negative.

    The old coded 'Langmuir' was the same hyperbolic family under a different parameter
    name.  Paper 1's stated Langmuir ODE is instead represented by ``exp_model``.
    """
    return (M_eq * abs(M_eq) * k * t) / (1 + abs(M_eq) * k * t)

def stretched_exp_model(t, M_eq, k, n):
    """Stretched exponential: M_eq*(1-exp(-(k*t)^n)). Covers both Paper 1's Avrami
    (eq 4, integrated) and its Weibull (eq 5). With Paper 1's notation, k_avrami=1/a
    and n=b; they are therefore the same curve. This collapse is inherent to Paper 1's
    equations, not an artifact of this codebase."""
    return M_eq * (1 - np.exp(-np.power(k * t, n)))

def elovich_model(t, alpha, beta):
    """Positive Elovich displacement magnitude: ln(1 + alpha*beta*t) / beta.

    ``alpha`` is the initial rate because the derivative at t=0 is alpha. ``beta``
    controls how quickly that rate slows as the occupied surface becomes increasingly
    heterogeneous. The fitter applies the stage direction separately, allowing both
    parameters to remain positive when a zero-based desorption trace is negative.

    Unlike EXP, HYP and SE, Elovich has no finite equilibrium plateau. It is therefore
    included only for desorption, as requested for comparison with Paper 1.
    """
    return np.log1p(alpha * beta * t) / beta

# --- Hybrids ---
def hybrid_hyperbola_stretched_exp(t, M_eq1, k1, M_eq2, k2, n):
    """Two-component: hyperbola + stretched exponential. Covers the old PA, LA and WL
    hybrids, which were all this same 5-parameter model wearing different variable names
    (since P=L and A=W individually)."""
    return hyperbola_model(t, M_eq1, k1) + stretched_exp_model(t, M_eq2, k2, n)

def hybrid_double_stretched_exp(t, M_eq1, k1, n1, M_eq2, k2, n2):
    """Two-component: stretched exponential + stretched exponential. Covers the old WA
    hybrid. NOTE: its two components are mutually interchangeable (swap component 1 and 2
    and the curve is identical), so its parameters are not separately identifiable. Kept
    for completeness/reporting only — do not treat a win by this model as a meaningful
    mechanistic result without inspecting the fit for the swap symmetry."""
    return stretched_exp_model(t, M_eq1, k1, n1) + stretched_exp_model(t, M_eq2, k2, n2)

# Deduped library: one function per functionally distinct family, named for the family.
# Of the original 10 Paper-1-labelled entries, only ~5 (arguably 6) are actually distinct;
# see docstrings above for which old labels each entry now covers.
MODELS = {
    'F': {'func': fickian_model, 'k': 1},
    'EXP': {'func': exp_model, 'k': 2},
    'HYP': {'func': hyperbola_model, 'k': 2},
    'SE': {'func': stretched_exp_model, 'k': 3},
    'HYP_SE': {'func': hybrid_hyperbola_stretched_exp, 'k': 5},
    'SE2': {'func': hybrid_double_stretched_exp, 'k': 6},
    'ELO': {'func': elovich_model, 'k': 2, 'directions': {'desorption'}}
}

# Map showing which parameter index corresponds to M_eq for dynamic bounding
MEQ_INDICES = {
    'F': [], 'EXP': [0], 'HYP': [0], 'SE': [0],
    'HYP_SE': [0, 2], 'SE2': [0, 3], 'ELO': []
}

# Map showing which parameter index corresponds to an exponent (n) to prevent overflows
EXP_INDICES = {
    'F': [], 'EXP': [], 'HYP': [], 'SE': [2],
    'HYP_SE': [4], 'SE2': [2, 5], 'ELO': []
}

# ==========================================
# 2. THE FITTER & BAYESIAN SELECTION LOGIC
# ==========================================

def fit_stage_kinetics(time_array, mass_array, m_eq_guess=None, k_guess=None,
                       direction=None):
    """
    Fits all candidate models to a single stage's time/mass data.
    Computes parameter errors via covariance, calculates BIC, and ranks models.
    ``direction`` should be ``'sorption'`` or ``'desorption'``. If omitted, it is
    inferred from the sign of the final portion of the zero-based mass trace.
    Returns a dictionary suitable for high-throughput pipeline ingestion.
    """
    # Suppress non-fatal solver warnings to keep terminal clean
    warnings.simplefilter("ignore", OptimizeWarning)
    warnings.simplefilter("ignore", RuntimeWarning)
    
    # Force minimal time padding to prevent log(0) or 1/t crashing
    time_array = np.maximum(np.array(time_array, dtype=float), 1e-10)
    mass_array = np.array(mass_array, dtype=float)

    if direction is None:
        tail_size = max(1, min(len(mass_array), max(3, len(mass_array) // 10)))
        tail_displacement = np.median(mass_array[-tail_size:]) if len(mass_array) else 0.0
        stage_direction = 'desorption' if tail_displacement < 0 else 'sorption'
    else:
        stage_direction = str(direction).strip().lower()

    if stage_direction not in {'sorption', 'desorption', 'baseline'}:
        raise ValueError("direction must be 'sorption', 'desorption', or 'baseline'")
    
    n_points = len(time_array)
    if len(mass_array) > 0:
        # Use whichever extreme has the larger magnitude, not just the max: on a
        # zero-offset desorption stage the mass falls toward a *negative* plateau,
        # so np.max alone would return a near-zero point right at the stage start.
        peak, trough = np.max(mass_array), np.min(mass_array)
        max_mass = peak if abs(peak) >= abs(trough) else trough
    else:
        max_mass = 1.0

    # Safely unpack dynamic pipeline guesses
    meq = m_eq_guess if m_eq_guess is not None else max_mass
    kg = k_guess if k_guess is not None else 0.1

    def _hyp_k_guess(m_guess, rate_guess):
        """Convert an exponential rate guess to the physical PSO rate constant.

        EXP reaches half its plateau at ln(2)/k_exp, while HYP reaches half its
        plateau at 1/(|M_eq|*k2). Equating those times gives
        k2=k_exp/(|M_eq|*ln(2)).
        """
        m_safe = max(abs(m_guess), 1e-9)
        return rate_guess / (m_safe * np.log(2))

    results = []

    for name, meta in MODELS.items():
        allowed_directions = meta.get('directions')
        if allowed_directions is not None and stage_direction not in allowed_directions:
            continue

        k_param = meta['k']

        # 1. Dynamic Guesses based on model architecture
        if name == 'F': p0 = [kg]
        elif name == 'EXP': p0 = [meq, kg]
        elif name == 'HYP': p0 = [meq, _hyp_k_guess(meq, kg)]
        elif name == 'SE': p0 = [meq, kg, 1.0]
        elif name == 'HYP_SE': p0 = [
            meq/2, _hyp_k_guess(meq/2, kg), meq/2, kg, 1.0
        ]
        elif name == 'SE2':
            # The two components are swap-symmetric (component 1 <-> component 2 gives
            # an identical curve), so an identical seed for both gives the optimizer no
            # basis to separate them. Bias the split (70/30) and offset each component's
            # k and n so the search starts away from the degenerate symmetric point.
            p0 = [meq*0.7, kg*1.5, 0.8, meq*0.3, kg*0.5, 2.0]
        elif name == 'ELO':
            # Elovich's initial slope is alpha. Match it to the initial slope of the
            # exponential seed (|M_eq|*k), and start beta at the inverse stage size.
            magnitude = max(abs(meq), 1e-7)
            p0 = [max(magnitude * kg, 1e-7), 1.0 / magnitude]
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

        # Elovich is defined as a positive displacement magnitude. Desorption data are
        # stored as negative zero-based changes, so apply the sign outside the model;
        # alpha and beta remain positive and retain their standard interpretation.
        if name == 'ELO':
            fit_func = lambda t, alpha, beta: -elovich_model(t, alpha, beta)
        else:
            fit_func = meta['func']
            
        try:
            # Added ftol and xtol, and reduced maxfev to strictly prevent solver infinite loops
            popt, pcov = curve_fit(
                fit_func, time_array, mass_array,
                p0=p0, bounds=bounds, 
                maxfev=2000, ftol=1e-5, xtol=1e-5
            )
            
            predictions = fit_func(time_array, *popt)
            rss = np.sum((mass_array - predictions)**2)
            if rss <= 0: rss = 1e-15

            # Bayesian Information Criterion
            bic = n_points * np.log(rss / n_points) + k_param * np.log(n_points)

            # Profiled Gaussian log-likelihood (MLE of sigma^2 = RSS/n), stored now so we
            # don't have to refit everything later when Bayes factors are added at M4.
            log_likelihood = -0.5 * n_points * (np.log(2 * np.pi) + np.log(rss / n_points) + 1)

            results.append({
                'Model': name,
                'BIC': bic,
                'RSS': rss,
                'lnL': log_likelihood,
                'N': n_points,
                'k': k_param,
                'popt': popt,
                'pcov': pcov
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
