import unittest

import numpy as np

from src.kinetics import elovich_model, fit_stage_kinetics, hyperbola_model


class HyperbolaModelTests(unittest.TestCase):
    def test_positive_and_negative_plateaus_have_mirrored_shapes(self):
        time = np.array([0.0, 5.0, 10.0, 100_000.0])

        adsorption = hyperbola_model(time, M_eq=8.0, k=0.0125)
        desorption = hyperbola_model(time, M_eq=-8.0, k=0.0125)

        np.testing.assert_allclose(desorption, -adsorption)
        self.assertAlmostEqual(adsorption[2], 4.0)
        self.assertAlmostEqual(desorption[2], -4.0)
        self.assertAlmostEqual(adsorption[-1], 8.0, delta=0.001)
        self.assertAlmostEqual(desorption[-1], -8.0, delta=0.001)

    def test_negative_curve_is_finite_and_monotonic(self):
        time = np.linspace(0.0, 720.0, 721)
        curve = hyperbola_model(time, M_eq=-12.0, k=0.0025)

        self.assertTrue(np.isfinite(curve).all())
        self.assertTrue((np.diff(curve) <= 0.0).all())
        self.assertTrue((curve >= -12.0).all())
        self.assertTrue((curve <= 0.0).all())

    def test_fitter_recovers_a_negative_hyperbolic_stage(self):
        time = np.linspace(0.0, 240.0, 241)
        mass = hyperbola_model(time, M_eq=-9.0, k=0.0025)

        result = fit_stage_kinetics(
            time,
            mass,
            m_eq_guess=-8.5,
            k_guess=np.log(2) * 9.0 * 0.0025,
        )
        hyp_row = result['df'].loc[result['df']['Model'] == 'HYP'].iloc[0]
        fitted_plateau, fitted_rate = hyp_row['popt']

        self.assertAlmostEqual(fitted_plateau, -9.0, places=4)
        self.assertAlmostEqual(fitted_rate, 0.0025, places=6)
        self.assertTrue(np.isfinite(hyp_row['BIC']))


class ElovichModelTests(unittest.TestCase):
    def test_initial_slope_is_alpha(self):
        alpha = 0.4
        beta = 0.8
        dt = 1e-7

        numerical_initial_slope = (
            elovich_model(dt, alpha, beta) - elovich_model(0.0, alpha, beta)
        ) / dt

        self.assertAlmostEqual(numerical_initial_slope, alpha, places=6)

    def test_elovich_is_only_fitted_to_desorption(self):
        time = np.linspace(0.0, 120.0, 121)
        adsorption = 6.0 * (1.0 - np.exp(-0.03 * time))
        desorption = -elovich_model(time, alpha=0.35, beta=0.7)

        adsorption_result = fit_stage_kinetics(
            time, adsorption, m_eq_guess=6.0, k_guess=0.03,
            direction='sorption',
        )
        desorption_result = fit_stage_kinetics(
            time, desorption, m_eq_guess=desorption[-1], k_guess=0.03,
            direction='desorption',
        )

        self.assertNotIn('ELO', adsorption_result['df']['Model'].tolist())
        self.assertIn('ELO', desorption_result['df']['Model'].tolist())

        elo_row = desorption_result['df'].loc[
            desorption_result['df']['Model'] == 'ELO'
        ].iloc[0]
        fitted_alpha, fitted_beta = elo_row['popt']
        self.assertAlmostEqual(fitted_alpha, 0.35, places=4)
        self.assertAlmostEqual(fitted_beta, 0.7, places=4)

    def test_direction_can_be_inferred_from_negative_tail(self):
        time = np.linspace(0.0, 120.0, 121)
        desorption = -elovich_model(time, alpha=0.2, beta=0.5)

        result = fit_stage_kinetics(
            time, desorption, m_eq_guess=desorption[-1], k_guess=0.02,
        )

        self.assertIn('ELO', result['df']['Model'].tolist())


if __name__ == '__main__':
    unittest.main()
