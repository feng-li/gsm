from pathlib import Path

import numpy as np

from gsm.config import sp500_gaussian_mixture_setting
from gsm.data import Dataset, load_csv_dataset, subset_dataset


def test_load_sp500_calendar_csv():
    dataset = load_csv_dataset(
        Path("data") / "sp500_1990-2009_calendar.csv",
        response_column="Returns",
        add_constant=True,
    )

    assert dataset.y_name == "Returns"
    assert dataset.X.shape == (4845, 11)
    assert dataset.y.shape == (4845, 1)
    assert dataset.x_names[0] == "Const"
    assert dataset.x_names[-1] == "Time"
    assert "Const" not in Path("data/sp500_1990-2009_calendar.csv").read_text().splitlines()[0]
    assert dataset.date is not None
    assert dataset.date[8] == "1990-01-12"


def test_sp500_gaussian_setting_is_data_source_agnostic():
    setting = sp500_gaussian_mixture_setting()

    assert not hasattr(setting, "data_file_name")
    assert setting.n_components == 3
    assert setting.covs == ((0,), tuple(range(10)))
    assert setting.covs_mix == tuple(range(10))
    assert setting.add_constant is True
    assert setting.standardize == 2


def test_subset_dataset_preserves_metadata():
    dataset = Dataset(
        y=np.arange(5)[:, None],
        X=np.arange(10).reshape(5, 2),
        y_name="y",
        x_names=("a", "b"),
        date=("d0", "d1", "d2", "d3", "d4"),
    )

    subset = subset_dataset(dataset, np.asarray([1, 3]))

    np.testing.assert_array_equal(subset.y.reshape(-1), np.asarray([1, 3]))
    np.testing.assert_array_equal(subset.X, np.asarray([[2, 3], [6, 7]]))
    assert subset.y_name == "y"
    assert subset.x_names == ("a", "b")
    assert subset.date == ("d1", "d3")
