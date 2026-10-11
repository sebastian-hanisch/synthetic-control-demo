"""Presets: Vollständigkeit, gültige Werte, Einrasten der Permalink-Werte."""

import syn_constants as C
import syn_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_ORDER)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_widget_bounds_and_on_the_grid():
    for preset in C.PRESETS.values():
        assert 0 <= preset["scen"] < len(C.SCENARIOS)
        assert 0 <= preset["j"] < len(C.J_CHOICES)
        assert C.ZB_MIN <= preset["zb"] <= C.ZB_MAX
        steps = (preset["zb"] - C.ZB_MIN) / C.ZB_STEP
        assert abs(steps - round(steps)) < 1e-9
        assert 0 <= preset["seed"] <= C.SEED_MAX


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_labels_are_complete():
    assert set(C.SCENARIO_LABELS) == set(C.SCENARIOS)
    assert set(C.ESTIMATOR_LABELS) == set(C.ESTIMATOR_SHORT) == set(C.ESTIMATOR_COLORS) == set(C.ESTIMATORS)


def test_design_constants_are_consistent():
    assert 0 < C.T0 < C.N_PERIODS
    assert C.TREATED not in C.CONTROLS and len(C.CONTROLS) == C.S_ALL - 1
    assert max(C.J_CHOICES) <= len(C.CONTROLS)


def test_permalink_values_are_clamped_and_snapped():
    assert P.clamp_to_spec(P.SETTING_SPECS["zb_slider"], 1.55) == 1.6
    assert P.clamp_to_spec(P.SETTING_SPECS["zb_slider"], 9.0) == 2.0
    assert P.clamp_to_spec(P.SETTING_SPECS["j_radio"], 7) == 2
    assert P.clamp_to_spec(P.SETTING_SPECS["scen_radio"], -3) == 0
