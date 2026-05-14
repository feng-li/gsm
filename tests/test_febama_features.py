import csv
import importlib.util

import numpy as np
import pytest

from gsm.febama import (
    clean_features,
    compute_tsfeatures,
    feature_table,
    read_precomputed_feature_table,
    standardize_features,
)
from gsm.febama.data import LpdFeatures


def test_clean_features_drops_nonfinite_and_constant_columns():
    lpd = np.zeros((3, 2))
    features = np.asarray(
        [
            [1.0, 1.0, np.nan, 1.0],
            [2.0, 1.0, 4.0, np.inf],
            [3.0, 1.0, 5.0, 6.0],
        ]
    )
    data = LpdFeatures(
        lpd=lpd,
        features=features,
        model_names=("a", "b"),
        feature_names=("keep", "constant", "missing", "infinite"),
    )

    cleaned = clean_features(data)

    assert cleaned.feature_names == ("keep",)
    np.testing.assert_allclose(cleaned.feature_mean, np.asarray([2.0]))
    np.testing.assert_allclose(cleaned.feature_sd, np.asarray([1.0]))
    np.testing.assert_allclose(cleaned.features[:, 0], np.asarray([-1.0, 0.0, 1.0]))
    assert cleaned.model_names == ("a", "b")


def test_standardize_features_can_align_columns_by_name():
    features = np.asarray(
        [
            [90.0, 12.0],
            [80.0, 14.0],
        ]
    )

    scaled = standardize_features(
        features,
        feature_mean=np.asarray([10.0, 100.0]),
        feature_sd=np.asarray([2.0, 10.0]),
        feature_names=("b", "a"),
        reference_feature_names=("a", "b"),
    )

    np.testing.assert_allclose(scaled, np.asarray([[1.0, -1.0], [2.0, -2.0]]))


def test_read_precomputed_feature_table_uses_explicit_schema(tmp_path):
    path = tmp_path / "features.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["date", "origin", "y", "f1", "f2", "lpd_a", "lpd_b"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "date": "2000-01-03",
                "origin": "10",
                "y": "0.5",
                "f1": "1.0",
                "f2": "2.0",
                "lpd_a": "-1.0",
                "lpd_b": "-2.0",
            }
        )
        writer.writerow(
            {
                "date": "2000-01-04",
                "origin": "11",
                "y": "-0.2",
                "f1": "3.0",
                "f2": "4.0",
                "lpd_a": "-0.5",
                "lpd_b": "-1.5",
            }
        )

    table = read_precomputed_feature_table(
        path,
        feature_columns=("f1", "f2"),
        lpd_columns=("lpd_a", "lpd_b"),
        response_column="y",
        date_column="date",
        origin_column="origin",
        model_names=("a", "b"),
    )

    assert table.feature_names == ("f1", "f2")
    assert table.date == ("2000-01-03", "2000-01-04")
    assert table.origin == ("10", "11")
    np.testing.assert_allclose(table.features, np.asarray([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_allclose(table.response, np.asarray([0.5, -0.2]))
    assert table.lpd_features is not None
    assert table.lpd_features.feature_names == ("f1", "f2")
    np.testing.assert_allclose(table.lpd_features.lpd, np.asarray([[-1.0, -2.0], [-0.5, -1.5]]))


def test_feature_table_is_available_as_schema():
    assert "x_acf1" in feature_table
    assert "entropy" in feature_table
    assert len(feature_table) == 15


@pytest.mark.skipif(
    importlib.util.find_spec("tsfeatures") is None,
    reason="tsfeatures is not installed",
)
def test_compute_tsfeatures_uses_installed_package_for_requested_columns():
    y = np.sin(np.arange(40, dtype=float) / 3.0)

    features = compute_tsfeatures(y, frequency=1, feature_names=("x_acf1", "entropy"))

    assert set(features) == {"x_acf1", "entropy"}
    assert np.isfinite(list(features.values())).all()
