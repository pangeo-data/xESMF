import numpy as np
import pytest
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


def test_mask_source_indices():
    # Create a small sparse matrix
    coords = np.array([[0, 1, 2, 3, 4, 8], [0, 2, 4, 10, 11, 19]])
    data = np.array([0.3, 0.5, 0.2, 0.26, 0.7, 0.25])
    shape = (10, 20)
    W = sps.COO(coords, data, shape=shape)
    weights = xr.DataArray(W, dims=('out_dim', 'in_dim'))

    masked = xe.smm.mask_source_indices(weights, source_indices_to_mask=[4, 11])

    # Only selected entries should remain
    expected_coords = np.array([[0, 1, 3, 8], [0, 2, 10, 19]])
    expected_data = np.array([0.3, 0.5, 0.26, 0.25])

    assert np.array_equal(masked.data.coords, expected_coords)
    assert np.allclose(masked.data.data, expected_data)


def test_gen_mask_from_weights():
    """testing creating mask out of weight matrix Nans"""
    # Create input and output Dataset
    ds_in = xe.util.grid_2d(20, 40, 1, 20, 30, 1)
    ds_out = xe.util.grid_2d(20, 40, 2, 20, 30, 2)

    # Create random mask for ds_out
    mask = np.random.randint(low=0, high=2, size=(5, 10), dtype=np.int32)
    ds_out['mask'] = xr.DataArray(data=mask, dims=['y', 'x'])

    # Create remapping weights
    Weights = xe.Regridder(ds_in, ds_out, method='bilinear').weights

    # Generate mask from weights
    maskwgts = xe.smm.gen_mask_from_weights(Weights, 5, 10)

    # Assert equality between both masks
    assert np.array_equal(mask, maskwgts, equal_nan=False)


def test_post_apply_target_mask_to_weights():
    # Create a small sparse weights matrix with shape (9 target, 4 source)
    #  coords = [[target_indices], [source_indices]]
    coords = np.array([[0, 1, 1, 2, 3, 3, 4, 5], [0, 0, 1, 1, 2, 3, 2, 3]])
    data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.45, 0.7, 0.8])
    shape = (6, 4)
    W_sparse = sps.COO(coords, data, shape=shape)
    weights = xr.DataArray(W_sparse, dims=('out_dim', 'in_dim'))

    # Define a 3x3 mask for target (flattened size = 9):
    #   If all goes to plan, weights of cells 3 and 4 (i.e. index 2 and 3)
    #   will be set to 0.
    target_mask_2d = np.array([[True, False], [True, True], [False, True]])

    # Apply mask
    masked_weights = xe.smm.post_apply_target_mask_to_weights(weights, target_mask_2d)

    # Check results
    np.testing.assert_array_equal(masked_weights.data.data, np.array([0.1, 0.2, 0.3, 0.7, 0.8]))
    np.testing.assert_array_equal(
        masked_weights.data.coords, np.array([[0, 1, 1, 4, 5], [0, 0, 1, 2, 3]])
    )


def test_post_apply_target_mask_to_weights_exceptions():
    # Create a weights DataArray & mask
    coords = np.array([[0, 1], [0, 1]])
    data = np.array([0.5, 0.5])
    shape = (2, 2)
    W_sparse = sps.COO(coords, data, shape=shape)
    weights = xr.DataArray(W_sparse, dims=('out_dim', 'in_dim'))
    valid_mask = np.array([[True, False]])

    # Mask not array-like
    with pytest.raises(
        TypeError,
        match="Argument 'target_mask_2d' must be array-like and convertible to a numeric/boolean array",
    ):
        xe.smm.post_apply_target_mask_to_weights(weights, 'not_array_like')

    # Shape mismatch
    wrong_shape_mask = np.array([[True, False, True]])
    with pytest.raises(
        ValueError, match='Mismatch: weight matrix has 2 target cells, but mask has 3 elements'
    ):
        xe.smm.post_apply_target_mask_to_weights(weights, wrong_shape_mask)

    # Mask not 2D
    wrong_shape_mask = np.array([[[True]], [[True]]])
    with pytest.raises(
        ValueError, match="Argument 'target_mask_2d' must be 2D, got shape \\(2, 1, 1\\)"
    ):
        xe.smm.post_apply_target_mask_to_weights(weights, wrong_shape_mask)

    # That should work
    xe.smm.post_apply_target_mask_to_weights(weights, valid_mask)
