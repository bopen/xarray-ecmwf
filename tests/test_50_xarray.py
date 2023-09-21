import xarray as xr


def test_xarray_plugin() -> None:
    assert "ecmwf" in set(xr.backends.list_engines())
