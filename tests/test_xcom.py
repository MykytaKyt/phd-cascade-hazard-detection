import numpy as np
import pandas as pd

from thesis.kinematic.xcom import add_xcom_features, horizontal_margin, omega0


def _frames(com_x_px, ankles):
    rows = []
    for t, (cx, (l, r)) in enumerate(zip(com_x_px, ankles)):
        row = dict(seq="s", frame=t, valid=1, W=100.0, com_x=cx / 100.0, box_h=50.0)
        for j in range(17):
            row[f"kp{j}_x"], row[f"kp{j}_c"] = 0.0, 0.0
        row["kp15_x"], row["kp15_c"] = l, 0.9
        row["kp16_x"], row["kp16_c"] = r, 0.9
        rows.append(row)
    return pd.DataFrame(rows)


def test_horizontal_margin_sign():
    np.testing.assert_allclose(horizontal_margin(np.array([5.0, 12.0]), np.array([0.0, 0.0]), np.array([10.0, 10.0])),
                               [5.0, -2.0])


def test_static_body_has_positive_margin_and_xcom_equals_com():
    df = add_xcom_features(_frames([50.0] * 5, [(40.0, 60.0)] * 5))
    np.testing.assert_allclose(df["xcom_x"], 50.0)
    np.testing.assert_allclose(df["mos_static_x"], 10.0 / 50.0)
    np.testing.assert_allclose(df["mos_xcom_x"], 10.0 / 50.0)


def test_moving_com_leaves_base_of_support_earlier():
    # CoM moves 2 px/frame to the right; XCoM is ahead by v / omega0.
    df = add_xcom_features(_frames([50.0 + 2 * t for t in range(5)], [(40.0, 60.0)] * 5), fps=30.0)
    lead = 2 * 30.0 / omega0()
    np.testing.assert_allclose(df["xcom_x"] - df["com_x"] * 100.0, lead)
    assert (df["mos_xcom_x"] < df["mos_static_x"]).all()
    assert df["mos_xcom_x"].iloc[0] < 0 < df["mos_static_x"].iloc[0]


def test_invisible_ankles_give_nan():
    df = _frames([50.0] * 3, [(40.0, 60.0)] * 3)
    df[["kp15_c", "kp16_c"]] = 0.0
    out = add_xcom_features(df)
    assert out["mos_xcom_x"].isna().all()
