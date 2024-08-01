import numpy as np
import pytest
from numpy.testing import assert_almost_equal

import xesmf as xe


def test_grid_global():
    ds = xe.util.grid_global(1.5, 1.5)
    refshape = (120, 240)
    refshape_b = (121, 241)

    assert ds['lon'].values.shape == refshape
    assert ds['lat'].values.shape == refshape
    assert ds['lon_b'].values.shape == refshape_b
    assert ds['lat_b'].values.shape == refshape_b

    # Issue #181 (https://github.com/pangeo-data/xESMF/issues/181)
    d_lon = 360 / 4320
    d_lat = 180 / 2160
    ds = xe.util.grid_global(d_lon, d_lat)
    assert ds.lon.max() <= 180

    ds = xe.util.grid_global(1.5, 1.5, lon1=180)
    assert ds['lon_b'].isel(x_b=-1)[-1] == 180

    ds = xe.util.grid_global(1.5, 1.5, lon1=360)
    assert ds['lon_b'].isel(x_b=-1)[-1] == 360


def test_grid_global_bad_resolution():
    with pytest.warns(UserWarning):
        xe.util.grid_global(1.5, 1.23)

    with pytest.warns(UserWarning):
        xe.util.grid_global(1.23, 1.5)


def test_cell_area():
    ds = xe.util.grid_global(2.5, 2)
    area = xe.util.cell_area(ds)

    # total area of a unit sphere
    assert_almost_equal(area.sum(), np.pi * 4)


def test_simple_tripolar_grid():
    lon, lat = xe.util.simple_tripolar_grid(360, 180, lat_cap=60, lon_cut=-300.0)

    assert lon.min() >= -300.0
    assert lon.max() <= 360.0 - 300.0
    assert lat.min() >= -90
    assert lat.max() <= 90

    lon, lat = xe.util.simple_tripolar_grid(180, 90, lat_cap=60, lon_cut=-300.0)

    assert lon.min() >= -300.0
    assert lon.max() <= 360.0 - 300.0
    assert lat.min() >= -90
    assert lat.max() <= 90
