Xarray backend to map an ECMWF style request to a service onto an Xarray Dataset

This Open Source project is build by B-Open - https://www.bopen.eu

## Features and limitations

xarray-ecmwf is a Python library and [Xarray](https://docs.xarray.dev) backend with the following functionalities:

- opens an ECMWF style request as a Xarray Dataset connected to the remote services
  - the [Climate Data Store](https://cds.climate.copernicus.eu) via [cdsapi](https://github.com/ecmwf/cdsapi): ERA5, Seasonal forecasts
  - the [Athospheric Data Store](https://ads.atmosphere.copernicus.eu) via cdsapi
  - the [ECMWF Open data](https://www.ecmwf.int/en/forecasts/datasets/open-data) via [ecmwf-opendata](https://github.com/ecmwf/ecmwf-opendata): High resolution forecasts, ensemble forecast
- allows lazy loading the data and well integrated with [Dask](https://www.dask.org) and [Dask.distributed](https://distributed.dask.org)
- allows chunking the input request according to a configurable splitting strategy. Allowed strategies:
  - by one month
  - by one day
- supports requests returning a single GRIB file, via [cfgrib](https://github.com/ecmwf/cfgrib)

## Usage

```python
>>> import xarray as xr
>>> request = {
...     "dataset": "reanalysis-era5-pressure-levels",
...     "product_type": "reanalysis",
...     "variable": ["temperature"],
...     "year": ["2002"],
...     "month": ["01"],
...     "day": ["15"],
...     "time": ["00:00"],
... }
>>> ds = xr.open_dataset(request, engine="ecmwf")
>>> ds
<xarray.Dataset>
Dimensions:        (isobaricInhPa: 6, latitude: 721, longitude: 1440)
Coordinates:
    number         int64 ...
    time           datetime64[ns] ...
    step           timedelta64[ns] ...
  * isobaricInhPa  (isobaricInhPa) float64 1e+03 850.0 700.0 500.0 400.0 300.0
    valid_time     datetime64[ns] ...
  * latitude       (latitude) float64 90.0 89.75 89.5 ... -89.5 -89.75 -90.0
  * longitude      (longitude) float64 0.0 0.25 0.5 0.75 ... 359.2 359.5 359.8
Data variables:
    t              (isobaricInhPa, latitude, longitude) float32 ...
Attributes:
    GRIB_edition:            1
    GRIB_centre:             ecmf
    GRIB_centreDescription:  European Centre for Medium-Range Weather Forecasts
    GRIB_subCentre:          0
    Conventions:             CF-1.7
    institution:             European Centre for Medium-Range Weather Forecasts
    history:                 ...

```

## Workflow for developers/contributors

For best experience create a new conda environment (e.g. DEVELOP) with Python 3.10:

```
conda create -n DEVELOP -c conda-forge python=3.10
conda activate DEVELOP
```

Before pushing to GitHub, run the following commands:

1. Update conda environment: `make conda-env-update`
1. Install this package: `pip install -e .`
1. Sync with the latest [template](https://github.com/ecmwf-projects/cookiecutter-conda-package) (optional): `make template-update`
1. Run quality assurance checks: `make qa`
1. Run tests: `make unit-tests`
1. Run the static type checker: `make type-check`
1. Build the documentation (see [Sphinx tutorial](https://www.sphinx-doc.org/en/master/tutorial/)): `make docs-build`

## License

```
Copyright 2023, B-Open Solutions srl.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
