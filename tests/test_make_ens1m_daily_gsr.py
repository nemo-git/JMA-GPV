from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from make_ens1m_daily_gsr import build_gsr_dataset, extraterrestrial_radiation


class DailyGsrTest(unittest.TestCase):
    def test_fao56_example_20s_day_246_is_32_2(self):
        latitude = xr.DataArray([-20.0], dims="latitude")
        doy = xr.DataArray([246], dims="time")
        value = float(extraterrestrial_radiation(latitude, doy).squeeze())
        self.assertTrue(np.isclose(value, 32.2, atol=0.1))

    def test_gsr_uses_percent_cloud_and_preserves_member_dimensions(self):
        times = pd.to_datetime(["2026-07-15"])
        cloud = xr.DataArray(
            np.array([[[[0.0]], [[50.0]], [[100.0]]]]),
            dims=("time", "ensemble", "latitude", "longitude"),
            coords={
                "time": times,
                "ensemble": [-1, 0, 1],
                "latitude": [43.0],
                "longitude": [143.0],
            },
            attrs={"units": "%"},
            name="TCDC_daymean",
        )
        result = build_gsr_dataset(xr.Dataset({"TCDC_daymean": cloud}))
        ra = float(result["Ra"].squeeze())
        np.testing.assert_allclose(
            result["GSR"].values.ravel(),
            ra * np.array([0.75, 0.50, 0.25]),
        )
        self.assertEqual(result["GSR"].dims, cloud.dims)
        self.assertIn("GSR_mean", result)
        self.assertIn("GSR_p50", result)

    def test_polar_day_and_night_are_finite(self):
        latitude = xr.DataArray([89.0], dims="latitude")
        doy = xr.DataArray([172, 355], dims="time")
        values = extraterrestrial_radiation(latitude, doy).values.ravel()
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertGreater(values[0], 40.0)
        self.assertEqual(values[1], 0.0)

    def test_unknown_units_use_pipeline_percent_scale_even_below_one(self):
        cloud = xr.DataArray(
            np.array([[[[1.0]]]]),
            dims=("time", "ensemble", "latitude", "longitude"),
            coords={
                "time": pd.to_datetime(["2026-07-15"]),
                "ensemble": [0],
                "latitude": [43.0],
                "longitude": [143.0],
            },
            attrs={"units": "unknown"},
            name="TCDC_daymean",
        )
        with self.assertWarns(RuntimeWarning):
            result = build_gsr_dataset(xr.Dataset({"TCDC_daymean": cloud}))
        ra = float(result["Ra"].squeeze())
        self.assertTrue(np.isclose(float(result["GSR"].squeeze()), ra * 0.745))


if __name__ == "__main__":
    unittest.main()
