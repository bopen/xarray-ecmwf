FROM continuumio/miniconda3

WORKDIR /src/xarray-ecmwf

COPY environment.yml /src/xarray-ecmwf/

RUN conda install -c conda-forge gcc python=3.11 \
    && conda env update -n base -f environment.yml

COPY . /src/xarray-ecmwf

RUN pip install --no-deps -e .
