import numpy as np
import pandas as pd
from scipy.optimize import least_squares

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

PARAMETER_NAMES = {
    'F': ['k'],
    'EXP': ['M_eq', 'k'],
    'HYP': ['M_eq', 'k2'],
    'SE': ['M_eq', 'k', 'n'],
    'HYP_SE': ['M_eq1', 'k1', 'M_eq2', 'k2', 'n'],
    'SE2': ['M_eq1', 'k1', 'n1', 'M_eq2', 'k2', 'n2'],
    'ELO': ['alpha', 'beta'],
}

EARLY_WINDOW_MODELS = {'F', 'ELO'}


def _early_time_window(time_array, mass_array, fraction=0.60, min_points=8):
    """Return the trace up to its first ``fraction`` displacement crossing.

    Fickian and Elovich curves do not plateau. Restricting them to the initial
    displacement asks whether they describe the early regime, instead of
    penalising them for a long-time behaviour they were never designed to model.
    """
    if not 0 < fraction <= 1:
        raise ValueError("early-window fraction must be in (0, 1]")

    target = fraction * abs(mass_array[-1])
    crossings = np.flatnonzero(np.abs(mass_array) >= target)
    stop = crossings[0] + 1 if len(crossings) else len(mass_array)
    stop = min(len(mass_array), max(min_points, stop))
    return time_array[:stop], mass_array[:stop]


def _covariance_diagnostics(jacobian, rss, n_points, n_params, parameters):
    """Estimate covariance and scale-free identifiability diagnostics.

    The covariance is the usual local least-squares approximation. Dividing it
    by the parameter standard deviations gives a correlation matrix that is
    unchanged by a change of units, unlike ``cond(pcov)``.
    """
    covariance = np.full((n_params, n_params), np.nan)
    if n_points > n_params and np.linalg.matrix_rank(jacobian) == n_params:
        covariance = np.linalg.pinv(jacobian.T @ jacobian) * rss / (n_points - n_params)

    with np.errstate(invalid='ignore', divide='ignore'):
        standard_errors = np.sqrt(np.diag(covariance))
        scale = np.outer(standard_errors, standard_errors)
        correlation = covariance / scale
        relative_se = standard_errors / np.abs(parameters)

    if np.isfinite(correlation).any():
        np.fill_diagonal(correlation, 1.0)
    off_diagonal = correlation[~np.eye(n_params, dtype=bool)]
    finite_off_diagonal = np.abs(off_diagonal[np.isfinite(off_diagonal)])
    max_abs_rho = finite_off_diagonal.max() if len(finite_off_diagonal) else 0.0

    return covariance, standard_errors, correlation, relative_se, max_abs_rho

# ==========================================
# 2. THE FITTER & BAYESIAN SELECTION LOGIC
# ==========================================

def fit_stage_kinetics(time_array, mass_array, m_eq_guess=None, k_guess=None,
                       direction=None, early_fraction=0.60):
    """
    Fits all candidate models to a single stage's time/mass data.
    Computes parameter errors via covariance, calculates BIC, and ranks models.
    F and ELO are fitted only through ``early_fraction`` of the displacement and
    are reported as early-regime diagnostics, not whole-stage BIC competitors.
    Every attempted fit is returned, including non-converged fits.
    ``direction`` should be ``'sorption'`` or ``'desorption'``. If omitted, it is
    inferred from the sign of the final portion of the zero-based mass trace.
    Returns a dictionary suitable for high-throughput pipeline ingestion.
    """
    # Force minimal time padding to prevent log(0) or 1/t crashing
    time_array = np.maximum(np.array(time_array, dtype=float), 1e-10)
    mass_array = np.array(mass_array, dtype=float)

    if len(time_array) != len(mass_array) or len(time_array) == 0:
        raise ValueError("time_array and mass_array must be non-empty and equal length")
    finite = np.isfinite(time_array) & np.isfinite(mass_array)
    time_array, mass_array = time_array[finite], mass_array[finite]
    if len(time_array) < 3:
        raise ValueError("at least three finite observations are required")

    if direction is None:
        tail_size = max(1, min(len(mass_array), max(3, len(mass_array) // 10)))
        tail_displacement = np.median(mass_array[-tail_size:]) if len(mass_array) else 0.0
        stage_direction = 'desorption' if tail_displacement < 0 else 'sorption'
    else:
        stage_direction = str(direction).strip().lower()

    if stage_direction not in {'sorption', 'desorption', 'baseline'}:
        raise ValueError("direction must be 'sorption', 'desorption', or 'baseline'")
    
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
        selection_eligible = name not in EARLY_WINDOW_MODELS
        if selection_eligible:
            fit_time, fit_mass = time_array, mass_array
            fit_window = 'full_stage'
        else:
            fit_time, fit_mass = _early_time_window(
                time_array, mass_array, fraction=early_fraction,
                min_points=max(8, k_param + 2),
            )
            fit_window = f'early_{early_fraction:.0%}'

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
            fit = least_squares(
                lambda params: fit_func(fit_time, *params) - fit_mass,
                x0=p0, bounds=bounds,
                max_nfev=2000, ftol=1e-5, xtol=1e-5, gtol=1e-8,
            )
            popt = fit.x
            predictions = fit_func(fit_time, *popt)
            rss = np.sum((fit_mass - predictions)**2)
            if rss <= 0: rss = 1e-15

            fit_n = len(fit_time)
            pcov, perr, correlation, relative_se, max_abs_rho = _covariance_diagnostics(
                fit.jac, rss, fit_n, k_param, popt
            )

            # Bayesian Information Criterion
            bic = fit_n * np.log(rss / fit_n) + k_param * np.log(fit_n)

            # Profiled Gaussian log-likelihood (MLE of sigma^2 = RSS/n), stored now so we
            # don't have to refit everything later when Bayes factors are added at M4.
            log_likelihood = -0.5 * fit_n * (
                np.log(2 * np.pi) + np.log(rss / fit_n) + 1
            )

            identifiable_flags = []
            if max_abs_rho > 0.999:
                identifiable_flags.append('components_not_interpretable')
            elif max_abs_rho > 0.99:
                identifiable_flags.append('poorly_identifiable')
            if np.any(relative_se > 0.5):
                identifiable_flags.append('parameter_undetermined')
            if not np.isfinite(pcov).all():
                identifiable_flags.append('covariance_unavailable')

            results.append({
                'Model': name,
                'Success': bool(fit.success),
                'Status': int(fit.status),
                'Message': str(fit.message),
                'Selection_Eligible': selection_eligible,
                'Fit_Window': fit_window,
                'N_Fit': fit_n,
                'BIC': bic if fit.success else np.nan,
                'RSS': rss,
                'lnL': log_likelihood,
                'N': fit_n,
                'k': k_param,
                'parameter_names': PARAMETER_NAMES[name],
                'popt': popt,
                'pcov': pcov,
                'perr': perr,
                'correlation': correlation,
                'relative_se': relative_se,
                'max_abs_rho': max_abs_rho,
                'identifiability_flags': identifiable_flags,
            })
            
        except Exception as exc:
            # Keep failures visible. A failed model is neither a winner nor a loser.
            results.append({
                'Model': name,
                'Success': False,
                'Status': -1,
                'Message': str(exc),
                'Selection_Eligible': selection_eligible,
                'Fit_Window': fit_window,
                'N_Fit': len(fit_time),
                'BIC': np.nan,
                'RSS': np.nan,
                'lnL': np.nan,
                'N': len(fit_time),
                'k': k_param,
                'parameter_names': PARAMETER_NAMES[name],
                'popt': np.full(k_param, np.nan),
                'pcov': np.full((k_param, k_param), np.nan),
                'perr': np.full(k_param, np.nan),
                'correlation': np.full((k_param, k_param), np.nan),
                'relative_se': np.full(k_param, np.nan),
                'max_abs_rho': np.nan,
                'identifiability_flags': ['non_converged'],
            })

    df_results = pd.DataFrame(results)
    eligible = df_results['Selection_Eligible'] & df_results['Success'] & df_results['BIC'].notna()
    df_results['Delta_BIC'] = np.nan
    if not eligible.any():
        return {'best_model': 'Fit Failed', 'delta_bic': np.nan, 'df': df_results}

    min_bic = df_results.loc[eligible, 'BIC'].min()
    df_results.loc[eligible, 'Delta_BIC'] = df_results.loc[eligible, 'BIC'] - min_bic
    df_results = df_results.sort_values(
        ['Selection_Eligible', 'Success', 'Delta_BIC'],
        ascending=[False, False, True], na_position='last'
    ).reset_index(drop=True)
    ranked = df_results[df_results['Delta_BIC'].notna()]

    return {
        'best_model': ranked.iloc[0]['Model'],
        'delta_bic': ranked.iloc[1]['Delta_BIC'] if len(ranked) > 1 else np.nan,
        'df': df_results
    }


def _kinetic_seed(time_array, mass_array):
    """Estimate an EXP rate seed from the observed half-displacement time."""
    m_eq_guess = mass_array[-1]
    crossings = np.flatnonzero(np.abs(mass_array) >= abs(m_eq_guess) / 2)
    t50 = time_array[crossings[0]] if len(crossings) else time_array[-1] / 2
    t50 = max(float(t50), 0.1)
    return m_eq_guess, np.log(2) / t50


def _format_parameters(row):
    """Format a fitted parameter vector and its one-standard-error uncertainty."""
    values = []
    for name, value, error in zip(row['parameter_names'], row['popt'], row['perr']):
        if np.isfinite(value) and np.isfinite(error):
            values.append(f"{name}={value:.5g} ± {error:.2g}")
        elif np.isfinite(value):
            values.append(f"{name}={value:.5g} ± unavailable")
        else:
            values.append(f"{name}=fit failed")
    return '; '.join(values)


def _component_summary(model_name, parameters):
    """Return exponent and amplitude fractions for the selected model."""
    if model_name == 'SE':
        return parameters[2], 'single component'
    if model_name == 'HYP_SE':
        amplitudes = np.abs(parameters[[0, 2]])
        exponent = parameters[4]
    elif model_name == 'SE2':
        amplitudes = np.abs(parameters[[0, 3]])
        exponent = f"n1={parameters[2]:.4g}; n2={parameters[5]:.4g}"
    else:
        return np.nan, 'single component'

    total = amplitudes.sum()
    weights = amplitudes / total if total > 0 else np.array([np.nan, np.nan])
    return exponent, f"w1={weights[0]:.3f}; w2={weights[1]:.3f}"


def build_stage_kinetics_table(
    segmented_df, dmdt_threshold=0.002, early_fraction=0.60,
    time_col='time_minutes_', mass_col='mass_change_pct',
    dmdt_col='dm_dt_minute_', rh_col='rounded_rh',
):
    """Fit every stage and return the Stage-1 deliverable plus raw fit details.

    ``segmented_df`` should contain one sample. The compact table is intended for
    review; ``fit_details`` preserves covariance matrices, correlations, relative
    errors, solver messages, and failed attempts for audit or later re-thresholding.
    """
    required = {
        'stage_id', 'direction', time_col, mass_col, dmdt_col, rh_col,
    }
    missing = required.difference(segmented_df.columns)
    if missing:
        raise ValueError(f"segmented_df is missing columns: {sorted(missing)}")

    table_rows = []
    fit_details = {}
    for stage_id, stage in segmented_df.groupby('stage_id', sort=True):
        stage = stage.sort_values(time_col)
        time = stage[time_col].to_numpy(dtype=float)
        mass = stage[mass_col].to_numpy(dtype=float)
        time = time - time[0]
        mass = mass - mass[0]
        direction = str(stage['direction'].iloc[0]).lower()
        m_eq_guess, k_guess = _kinetic_seed(time, mass)

        fit_result = fit_stage_kinetics(
            time, mass, m_eq_guess=m_eq_guess, k_guess=k_guess,
            direction=direction, early_fraction=early_fraction,
        )
        detail = fit_result['df']
        fit_details[stage_id] = detail
        flags = []

        end_dmdt = float(stage[dmdt_col].iloc[-1])
        if not np.isfinite(end_dmdt) or abs(end_dmdt) > dmdt_threshold:
            flags.extend(['truncated', 'selection_provisional'])

        failed_models = detail.loc[~detail['Success'], 'Model'].tolist()
        if failed_models:
            flags.append('non_converged:' + ','.join(failed_models))

        if fit_result['best_model'] == 'Fit Failed':
            table_rows.append({
                'stage': stage_id, 'RH': stage[rh_col].iloc[0],
                'direction': direction, 'best_model': 'Fit Failed',
                'delta_BIC_runner_up': np.nan, 'parameters_plus_minus_SE': '',
                'Avrami_n': np.nan, 'component_weights': '',
                'end_dm_dt': end_dmdt,
                'flags': '; '.join(flags + ['no_eligible_model_converged']),
            })
            continue

        winner = detail.loc[detail['Model'] == fit_result['best_model']].iloc[0]
        flags.extend(winner['identifiability_flags'])

        amplitude_indices = MEQ_INDICES.get(fit_result['best_model'], [])
        if len(amplitude_indices) > 1 and any(
            abs(winner['popt'][idx]) <= winner['perr'][idx]
            for idx in amplitude_indices if np.isfinite(winner['perr'][idx])
        ):
            flags.append('redundant_component_amplitude_spans_zero')

        exponent, weights = _component_summary(
            fit_result['best_model'], winner['popt']
        )
        table_rows.append({
            'stage': stage_id,
            'RH': stage[rh_col].iloc[0],
            'direction': direction,
            'best_model': fit_result['best_model'],
            'delta_BIC_runner_up': fit_result['delta_bic'],
            'parameters_plus_minus_SE': _format_parameters(winner),
            'Avrami_n': exponent,
            'component_weights': weights,
            'end_dm_dt': end_dmdt,
            'flags': '; '.join(dict.fromkeys(flags)),
        })

    return pd.DataFrame(table_rows), fit_details
