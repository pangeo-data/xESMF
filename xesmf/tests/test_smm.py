import numpy as np
import sparse as sps
import xarray as xr

import xesmf as xe


def test_add_nans_to_weights():
    """testing adding Nans to empty rows in sparse matrix"""
    # create input sparse matrix with one empty row (j=2)
    coords = np.array([[0, 3, 1, 0], [0, 3, 1, 2]])
    data = np.array([4.0, 5.0, 7.0, 9.0])
    Matin = sps.COO(coords, data, shape=(4, 4))

    # this is what is expected to come out (Nan added at i=0, j=2)
    coords = np.array([[0, 3, 1, 0, 2], [0, 3, 1, 2, 0]])
    data = np.array([4.0, 5.0, 7.0, 9.0, np.nan])
    expected = sps.COO(coords, data, shape=(4, 4))

    Matout = xe.smm.add_nans_to_weights(xr.DataArray(Matin, dims=('in', 'out')))
    assert np.allclose(expected.todense(), Matout.data.todense(), equal_nan=True)

    # Matrix without empty rows should return the same
    coords = np.array([[0, 3, 1, 0, 2], [0, 3, 1, 2, 1]])
    data = np.array([4.0, 5.0, 7.0, 9.0, 10.0])
    Matin = sps.COO(coords, data, shape=(4, 4))

    Matout = xe.smm.add_nans_to_weights(xr.DataArray(Matin, dims=('in', 'out')))
    assert np.allclose(Matin.todense(), Matout.data.todense())


def test_gen_mask_from_weights():
    """testing creating mask out of weight matrix Nans"""
    # Create input and output Dataset
    ds_in = xe.util.grid_2d(20, 40, 1, 20, 30, 1)
    ds_out = xe.util.grid_2d(20, 40, 2, 20, 30, 2)

    # Create random mask for ds_out
    mask = np.random.randint(low=0, high=2, size=(5, 10), dtype=np.int32)
    ds_out['mask'] = xr.DataArray(data=mask, dims=['lat', 'lon'])

    # Create remapping weights
    Weights = xe.Regridder(ds_in, ds_out, method='bilinear').weights

    # Generate mask from weights
    maskwgts = xe.smm.gen_mask_from_weights(Weights, 5, 10)

    # Assert equality between both masks
    assert np.array_equal(mask, maskwgts, equal_nan=False)
