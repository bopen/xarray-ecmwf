import xarray_ecmwf


def test_version() -> None:
    assert xarray_ecmwf.__version__ != "999"
