from pathlib import Path

from gsm.config import sp500_gaussian_mixture_setting
from gsm.data import load_csv_dataset


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


def test_sp500_gaussian_setting_uses_sp500_csv():
    setting = sp500_gaussian_mixture_setting()

    assert setting.data_file_name == "sp500_1990-2009_calendar.csv"
    assert setting.n_components == 3
    assert setting.covs == ((0,), tuple(range(10)))
    assert setting.covs_mix == tuple(range(10))
    assert setting.add_constant is True
    assert setting.standardize == 2
