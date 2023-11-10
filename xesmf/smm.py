"""
Sparse matrix multiplication (SMM) using scipy.sparse library.
"""
import warnings
from pathlib import Path
from typing import Any, Tuple

import numba as nb  # type: ignore[import]
import numpy as np
import numpy.typing as npt
import sparse as sps  # type: ignore[import]
import xarray as xr


def read_weights(
    weights: str | Path | xr.Dataset | xr.DataArray | sps.COO | dict[str, Any],
    n_in: int,
    n_out: int,
) -> xr.DataArray:
    """
    Read regridding weights into a DataArray (sparse COO matrix).

    Parameters
    ----------
    weights : str, Path, xr.Dataset, xr.DataArray, sparse.COO
        Weights generated by ESMF. Can be a path to a netCDF file generated by ESMF, an xr.Dataset,
        a dictionary created by `ESMPy.api.Regrid.get_weights_dict` or directly the sparse
        array as returned by this function.

    n_in, n_out : integers
        ``(N_out, N_in)`` will be the shape of the returning sparse matrix.
        They are the total number of grid boxes in input and output grids::

            N_in = Nx_in * Ny_in
            N_out = Nx_out * Ny_out

        We need them because the shape cannot always be inferred from the
        largest column and row indices, due to unmapped grid boxes.

    Returns
    -------
    xr.DataArray
        A DataArray backed by a sparse.COO array, with dims ('out_dim', 'in_dim')
        and size (n_out, n_in).
    """
    if isinstance(weights, (str, Path, xr.Dataset, dict)):
        return _parse_coords_and_values(weights, n_in, n_out)

    if isinstance(weights, sps.COO):
        return xr.DataArray(weights, dims=('out_dim', 'in_dim'), name='weights')

    if isinstance(weights, xr.DataArray):  # type: ignore[no-untyped-def]
        return weights

    raise ValueError(f'Weights of type {type(weights)} not understood.')


def _parse_coords_and_values(
    indata: str | Path | xr.Dataset | dict[str, Any],
    n_in: int,
    n_out: int,
) -> xr.DataArray:
    """Creates a sparse.COO array from weights stored in a dict-like fashion.

    Parameters
    ----------
    indata: str, Path, xr.Dataset or dict
        A dictionary as returned by ESMF.Regrid.get_weights_dict
        or an xarray Dataset (or its path) as saved by xESMF.
    n_in : int
        The number of points in the input grid.
    n_out : int
        The number of points in the output grid.

    Returns
    -------
    sparse.COO
        Sparse array in the COO format.
    """

    if isinstance(indata, (str, Path, xr.Dataset)):
        if not isinstance(indata, xr.Dataset):
            if not Path(indata).exists():
                raise IOError(f'Weights file not found on disk.\n{indata}')
            ds_w = xr.open_dataset(indata)  # type: ignore[no-untyped-def]
        else:
            ds_w = indata

        if not {'col', 'row', 'S'}.issubset(ds_w.variables):
            raise ValueError(
                'Weights dataset should have variables `col`, `row` and `S` storing the indices '
                'and values of weights.'
            )

        col = ds_w['col'].values - 1  # type: ignore[no-untyped-def]
        row = ds_w['row'].values - 1  # type: ignore[no-untyped-def]
        s = ds_w['S'].values  # type: ignore[no-untyped-def]

    elif isinstance(indata, dict):  # type: ignore
        if not {'col_src', 'row_dst', 'weights'}.issubset(indata.keys()):
            raise ValueError(
                'Weights dictionary should have keys `col_src`, `row_dst` and `weights` storing '
                'the indices and values of weights.'
            )
        col = indata['col_src'] - 1
        row = indata['row_dst'] - 1
        s = indata['weights']

    crds = np.stack([row, col])
    return xr.DataArray(sps.COO(crds, s, (n_out, n_in)), dims=('out_dim', 'in_dim'), name='weights')


def check_shapes(
    indata: npt.NDArray[Any],
    weights: npt.NDArray[Any],
    shape_in: Tuple[int, int],
    shape_out: Tuple[int, int],
) -> None:
    """Compare the shapes of the input array, the weights and the regridder and raises
    potential errors.

    Parameters
    ----------
    indata : array
        Input array with the two spatial dimensions at the end,
        which should fit shape_in.
    weights : array
        Weights 2D array of shape (out_dim, in_dim).
        First element should be the product of shape_out.
        Second element should be the product of shape_in.
    shape_in : 2-tuple of int
        Shape of the input of the Regridder.
    shape_out : 2-tuple of int
        Shape of the output of the Regridder.

    Raises
    ------
    ValueError
        If any of the conditions is not respected.
    """
    # COO matrix is fast with F-ordered array but slow with C-array, so we
    # take in a C-ordered and then transpose)
    # (CSR or CRS matrix is fast with C-ordered array but slow with F-array)
    if hasattr(indata, 'flags') and not indata.flags['C_CONTIGUOUS']:
        warnings.warn('Input array is not C_CONTIGUOUS. ' 'Will affect performance.')

    # Limitation from numba : some big-endian dtypes are not supported.
    try:
        nb.from_dtype(indata.dtype)  # type: ignore
        nb.from_dtype(weights.dtype)  # type: ignore
    except (NotImplementedError, nb.core.errors.NumbaError):  # type: ignore
        warnings.warn(
            'Input array has a dtype not supported by sparse and numba.'
            'Computation will fall back to scipy.'
        )

    # get input shape information
    shape_horiz = indata.shape[-2:]

    if shape_horiz != shape_in:
        raise ValueError(
            f'The horizontal shape of input data is {shape_horiz}, different from that '
            f'of the regridder {shape_in}!'
        )

    if shape_in[0] * shape_in[1] != weights.shape[1]:
        raise ValueError('ny_in * nx_in should equal to weights.shape[1]')

    if shape_out[0] * shape_out[1] != weights.shape[0]:
        raise ValueError('ny_out * nx_out should equal to weights.shape[0]')


def apply_weights(
    weights: sps.COO,
    indata: npt.NDArray[Any],
    shape_in: Tuple[int, int],
    shape_out: Tuple[int, int],
) -> npt.NDArray[Any]:
    """
    Apply regridding weights to data.

    Parameters
    ----------
    weights : sparse COO matrix
        Regridding weights.
    indata : numpy array of shape ``(..., n_lat, n_lon)`` or ``(..., n_y, n_x)``.
        Should be C-ordered. Will be then transposed to F-ordered.
    shape_in, shape_out : tuple of two integers
        Input/output data shape.
        For rectilinear grid, it is just ``(n_lat, n_lon)``.

    Returns
    -------
    outdata : numpy array of shape ``(..., shape_out[0], shape_out[1])``.
        Extra dimensions are the same as `indata`.
        If input data is C-ordered, output will also be C-ordered.
    """
    extra_shape = indata.shape[0:-2]

    # Limitation from numba : some big-endian dtypes are not supported.
    indata_dtype = indata.dtype
    try:
        nb.from_dtype(indata.dtype)  # type: ignore
        nb.from_dtype(weights.dtype)  # type: ignore
    except (NotImplementedError, nb.core.errors.NumbaError):  # type: ignore
        indata = indata.astype('<f8')  # On the fly conversion

    # Dot product
    outdata = np.tensordot(
        indata,
        weights,
        axes=((indata.ndim - 2, indata.ndim - 1), (weights.ndim - 2, weights.ndim - 1)),
    )

    # Ensure same dtype as the input.
    outdata = outdata.astype(indata_dtype)

    # Ensure output shape is what is expected
    outdata = outdata.reshape(*extra_shape, shape_out[0], shape_out[1])
    return outdata


def add_nans_to_weights(weights: xr.DataArray) -> xr.DataArray:
    """Add NaN in empty rows of the regridding weights sparse matrix.

    By default, empty rows in the weights sparse matrix are interpreted as zeroes. This can become problematic
    when the field being interpreted has legitimate null values. This function inserts NaN values in each row to
    make sure empty weights are propagated as NaNs instead of zeros.

    Parameters
    ----------
    weights : DataArray backed by a sparse.COO array
        Sparse weights matrix.

    Returns
    -------
    DataArray backed by a sparse.COO array
        Sparse weights matrix.
    """

    # Taken from @trondkr and adapted by @raphaeldussin to use `lil`.
    # lil matrix is better than CSR when changing sparsity
    m = weights.data.to_scipy_sparse().tolil()
    # replace empty rows by one NaN value at element 0 (arbitrary)
    # so that remapped element become NaN instead of zero
    for krow in range(len(m.rows)):
        m.rows[krow] = [0] if m.rows[krow] == [] else m.rows[krow]
        m.data[krow] = [np.NaN] if m.data[krow] == [] else m.data[krow]

    # update regridder weights (in COO)
    weights = weights.copy(data=sps.COO.from_scipy_sparse(m))  # type: ignore
    return weights


def _combine_weight_multipoly(  # type: ignore
    weights: xr.DataArray,
    areas: npt.NDArray[np.integer[Any]],
    indexes: npt.NDArray[np.integer[Any]],
) -> xr.DataArray:
    """Reduce a weight sparse matrix (csc format) by combining (adding) columns.

    This is used to sum individual weight matrices from multi-part geometries.

    Parameters
    ----------
    weights : DataArray
        Usually backed by a sparse.COO array, with dims ('out_dim', 'in_dim')
    areas : np.array
        Array of destination areas, following same order as weights.
    indexes : array of integers
        Columns with the same 'index' will be summed into a single column at this
        index in the output matrix.

    Returns
    -------
    sparse matrix (CSC)
        Sum of weights from individual geometries.
    """

    sub_weights = weights.rename(out_dim='subgeometries')

    # Create a sparse DataArray with the mesh areas
    # This ties the `out_dim` (the dimension for the original geometries) to the
    # subgeometries dimension (the exploded polygon exteriors and interiors).
    crds = np.stack([indexes, np.arange(len(indexes))])
    a = xr.DataArray(
        sps.COO(crds, areas, (indexes.max() + 1, len(indexes)), fill_value=0),
        dims=('out_dim', 'subgeometries'),
        name='area',
    )

    # Weight the regridding weights by the area of the destination polygon and sum over sub-geometries
    out = (sub_weights * a).sum(dim='subgeometries')

    # Renormalize weights along in_dim
    wsum = out.sum('in_dim')

    # Change the fill_value to 1
    wsum = wsum.copy(
        data=sps.COO(wsum.data.coords, wsum.data.data, shape=wsum.data.shape, fill_value=1)
    )

    return out / wsum
