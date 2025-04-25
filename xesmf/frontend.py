"""
Frontend for xESMF, exposed to users.
"""

import warnings

import cf_xarray as cfxr
import numpy as np
import sparse as sps
import xarray as xr
from shapely.geometry import LineString
from xarray import DataArray, Dataset

from .backend import Grid, LocStream, Mesh, add_corner, esmf_regrid_build, esmf_regrid_finalize
from .smm import (
    _combine_weight_multipoly,
    _parse_coords_and_values,
    add_nans_to_weights,
    apply_weights,
    check_shapes,
    read_weights,
)
from .util import LAT_CF_ATTRS, LON_CF_ATTRS, split_polygons_and_holes

try:
    import dask.array as da

    dask_array_type = (da.Array,)  # for isinstance checks
except ImportError:
    dask_array_type = ()


def subset_regridder(
    ds_out, ds_in, method, in_dims, out_dims, locstream_in, locstream_out, periodic, **kwargs
):
    """Compute subset of weights"""
    kwargs.pop('filename', None)  # Don't save subset of weights
    kwargs.pop('reuse_weights', None)

    # Renaming dims to original names for the subset regridding
    if locstream_in:
        ds_in = ds_in.rename({'x_in': in_dims[0]})
    else:
        ds_in = ds_in.rename({'y_in': in_dims[0], 'x_in': in_dims[1]})

    if locstream_out:
        ds_out = ds_out.rename({'x_out': out_dims[1]})
    else:
        ds_out = ds_out.rename({'y_out': out_dims[0], 'x_out': out_dims[1]})

    regridder = Regridder(
        ds_in, ds_out, method, locstream_in, locstream_out, periodic, parallel=False, **kwargs
    )
    return regridder.w


def as_2d_mesh(lon, lat):
    if (lon.ndim, lat.ndim) == (2, 2):
        assert lon.shape == lat.shape, 'lon and lat should have same shape'
    elif (lon.ndim, lat.ndim) == (1, 1):
        lon, lat = np.meshgrid(lon, lat)
    else:
        raise ValueError('lon and lat should be both 1D or 2D')

    return lon, lat


def _get_lon_lat(ds):
    """Return lon and lat extracted from ds."""
    if ('lat' in ds and 'lon' in ds) or ('lat' in ds.coords and 'lon' in ds.coords):
        # Old way.
        return ds['lon'], ds['lat']
    # else : cf-xarray way
    try:
        lon = ds.cf['longitude']
        lat = ds.cf['latitude']
    except (KeyError, AttributeError, ValueError):
        # KeyError if cfxr doesn't detect the coords
        # AttributeError if ds is a dict
        raise ValueError('dataset must include lon/lat or be CF-compliant')

    return lon, lat


def _get_lon_lat_bounds(ds):
    """Return bounds of lon and lat extracted from ds."""
    if 'lat_b' in ds and 'lon_b' in ds:
        # Old way.
        return ds['lon_b'], ds['lat_b']
    # else : cf-xarray way
    if 'longitude' not in ds.cf.coordinates:
        # If we are here, _get_lon_lat() didn't fail, thus we should be able to guess the coords.
        ds = ds.cf.guess_coord_axis()
    try:
        lon_bnds = ds.cf.get_bounds('longitude')
        lat_bnds = ds.cf.get_bounds('latitude')
    except KeyError:  # bounds are not already present
        if ds.cf['longitude'].ndim > 1:
            # We cannot infer 2D bounds, raise KeyError as custom "lon_b" is missing.
            raise KeyError('lon_b')
        lon_name = ds.cf['longitude'].name
        lat_name = ds.cf['latitude'].name
        ds = ds.cf.add_bounds([lon_name, lat_name])
        lon_bnds = ds.cf.get_bounds('longitude')
        lat_bnds = ds.cf.get_bounds('latitude')

    # Convert from CF bounds to xESMF bounds.
    # order=None is because we don't want to assume the dimension order for 2D bounds.
    lon_b = cfxr.bounds_to_vertices(lon_bnds, ds.cf.get_bounds_dim_name('longitude'), order=None)
    lat_b = cfxr.bounds_to_vertices(lat_bnds, ds.cf.get_bounds_dim_name('latitude'), order=None)
    return lon_b, lat_b


def ds_to_ESMFgrid(ds, need_bounds=False, periodic=None, append=None):
    """
    Convert xarray DataSet or dictionary to ESMF.Grid object.

    Parameters
    ----------
    ds : xarray DataSet or dictionary
        Contains variables ``lon``, ``lat``,
        and optionally ``lon_b``, ``lat_b`` if need_bounds=True.

        Shape should be ``(n_lat, n_lon)`` or ``(n_y, n_x)``,
        as normal C or Python ordering. Will be then tranposed to F-ordered.

    need_bounds : bool, optional
        Need cell boundary values?

    periodic : bool, optional
        Periodic in longitude?

    Returns
    -------
    grid
        ESMF.Grid object
    shape
        Shape of the grid
    dim_names
        Dimension names of the grid

    """
    # use np.asarray(dr) instead of dr.values, so it also works for dictionary

    lon, lat = _get_lon_lat(ds)
    if hasattr(lon, 'dims'):
        if lon.ndim == 1:
            dim_names = lat.dims + lon.dims
        else:
            dim_names = lon.dims
    else:
        dim_names = None
    lon, lat = as_2d_mesh(np.asarray(lon), np.asarray(lat))

    if 'mask' in ds:
        mask = np.asarray(ds['mask'])
    else:
        mask = None

    # tranpose the arrays so they become Fortran-ordered
    if mask is not None:
        grid = Grid.from_xarray(lon.T, lat.T, periodic=periodic, mask=mask.T)
    else:
        grid = Grid.from_xarray(lon.T, lat.T, periodic=periodic, mask=None)

    if need_bounds:
        lon_b, lat_b = _get_lon_lat_bounds(ds)
        lon_b, lat_b = as_2d_mesh(np.asarray(lon_b), np.asarray(lat_b))
        add_corner(grid, lon_b.T, lat_b.T)

    return grid, lon.shape, dim_names


def ds_to_ESMFlocstream(ds):
    """
    Convert xarray DataSet or dictionary to ESMF.LocStream object.

    Parameters
    ----------
    ds : xarray DataSet or dictionary
        Contains variables ``lon``, ``lat``.

    Returns
    -------
    locstream : ESMF.LocStream object

    """

    lon, lat = _get_lon_lat(ds)
    if hasattr(lon, 'dims'):
        dim_names = lon.dims
    else:
        dim_names = None
    lon, lat = np.asarray(lon), np.asarray(lat)

    if len(lon.shape) > 1:
        raise ValueError('lon can only be 1d')
    if len(lat.shape) > 1:
        raise ValueError('lat can only be 1d')

    assert lon.shape == lat.shape

    locstream = LocStream.from_xarray(lon, lat)

    return locstream, (1,) + lon.shape, dim_names


def polys_to_ESMFmesh(polys):
    """
    Convert a sequence of shapely Polygons to a ESMF.Mesh object.

    MultiPolygons are split in their polygon parts and holes are ignored.

    Parameters
    ----------
    polys : sequence of shapely Polygon or MultiPolygon

    Returns
    -------
    exterior : ESMF.Mesh
        A mesh where elements are the exterior rings of the polygons
    tuple
        The shape of the mesh : (1, N_elements)

    """
    ext, holes, _, _ = split_polygons_and_holes(polys)
    if len(holes) > 0:
        warnings.warn(
            'Some passed polygons have holes, those are not represented in the returned Mesh.'
        )
    return Mesh.from_polygons(ext), (1, len(ext))


class BaseRegridder(object):
    def __init__(
        self,
        grid_in,
        grid_out,
        method,
        filename=None,
        reuse_weights=False,
        extrap_method=None,
        extrap_dist_exponent=None,
        extrap_num_src_pnts=None,
        weights=None,
        ignore_degenerate=None,
        input_dims=None,
        output_dims=None,
        unmapped_to_nan=False,
        parallel=False,
    ):
        """
        Base xESMF regridding class supporting ESMF objects: `Grid`, `Mesh` and `LocStream`.

        Create or use existing subclasses to support other types of input objects. See for example `Regridder`
        to regrid `xarray.DataArray` objects, or `SpatialAverager`
        to average grids over regions defined by polygons.

        Parameters
        ----------
        grid_in, grid_out : ESMF Grid or Locstream or Mesh
            Input and output grid structures as ESMFpy objects.

        method : str
            Regridding method. Options are

            - 'bilinear'
            - 'conservative', **need grid corner information**
            - 'conservative_normed', **need grid corner information**
            - 'patch'
            - 'nearest_s2d'
            - 'nearest_d2s'

        filename : str, optional
            Name for the weight file. The default naming scheme is::

                {method}_{Ny_in}x{Nx_in}_{Ny_out}x{Nx_out}.nc

            e.g. bilinear_400x600_300x400.nc

        reuse_weights : bool, optional
            Whether to read existing weight file to save computing time.
            False by default (i.e. re-compute, not reuse).

        extrap_method : str, optional
            Extrapolation method. Options are

            - 'inverse_dist'
            - 'nearest_s2d'

        extrap_dist_exponent : float, optional
            The exponent to raise the distance to when calculating weights for the
            extrapolation method. If none are specified, defaults to 2.0

        extrap_num_src_pnts : int, optional
            The number of source points to use for the extrapolation methods
            that use more than one source point. If none are specified, defaults to 8

        weights : None, coo_matrix, dict, str, Dataset, Path,
            Regridding weights, stored as
              - a scipy.sparse COO matrix,
              - a dictionary with keys `row_dst`, `col_src` and `weights`,
              - an xarray Dataset with data variables `col`, `row` and `S`,
              - or a path to a netCDF file created by ESMF.
            If None, compute the weights.

        ignore_degenerate : bool, optional
            If False (default), raise error if grids contain degenerated cells
            (i.e. triangles or lines, instead of quadrilaterals)

        input_dims : tuple of str, optional
            A tuple of dimension names to look for when regridding DataArrays or Datasets.
            If not given or if those are not found on the regridded object, regridding
            uses the two last dimensions of the object (or the last one for input LocStreams and Meshes).

        output_dims : tuple of str, optional
            A tuple of dimension names to look for when regridding DataArrays or Datasets.
            If not given or if those are not found on the regridded object, regridding
            uses the two last dimensions of the object (or the last one for output LocStreams and Meshes)

        unmapped_to_nan: boolean, optional
            Set values of unmapped points to `np.nan` instead of zero (ESMF default). This is useful for
            target cells lying outside of the source domain when no output mask is defined.
            If an output mask is defined, or regridding method is `nearest_s2d` or `nearest_d2s`,
            this option has no effect.

        parallel: bool, optional
            Are the weights generated in parallel with Dask. Default is False. When True, the weight
            generation in the BaseRegridder is skipped and weights are generated in paralell in the
            subsest_regridder instead.

        Returns
        -------
        baseregridder : xESMF BaseRegridder object

        """
        self.method = method
        self.reuse_weights = reuse_weights
        self.extrap_method = extrap_method
        self.extrap_dist_exponent = extrap_dist_exponent
        self.extrap_num_src_pnts = extrap_num_src_pnts
        self.ignore_degenerate = ignore_degenerate
        self.periodic = getattr(grid_in, 'periodic_dim', None) is not None
        self.sequence_in = isinstance(grid_in, (LocStream, Mesh))
        self.sequence_out = isinstance(grid_out, (LocStream, Mesh))

        if input_dims is not None and len(input_dims) != int(not self.sequence_in) + 1:
            raise ValueError(f'Wrong number of dimension names in `input_dims` ({len(input_dims)}.')
        self.in_horiz_dims = input_dims

        if output_dims is not None and len(output_dims) != int(not self.sequence_out) + 1:
            raise ValueError(
                f'Wrong number of dimension names in `output dims` ({len(output_dims)}.'
            )
        self.out_horiz_dims = output_dims

        # record grid shape information
        # We need to invert Grid shapes to respect xESMF's convention (y, x).
        self.shape_in = grid_in.get_shape()[::-1]
        self.shape_out = grid_out.get_shape()[::-1]
        self.n_in = self.shape_in[0] * self.shape_in[1]
        self.n_out = self.shape_out[0] * self.shape_out[1]

        # some logic about reusing weights with either filename or weights args
        if reuse_weights and (filename is None) and (weights is None):
            raise ValueError('To reuse weights, you need to provide either filename or weights.')

        if not parallel:
            if not reuse_weights and weights is None:
                weights = self._compute_weights(grid_in, grid_out)  # Dictionary of weights
            else:
                weights = filename if filename is not None else weights

            assert weights is not None

            # Convert weights, whatever their format, to a sparse coo matrix
            self.weights = read_weights(weights, self.n_in, self.n_out)

            # replace zeros by NaN for weight matrix entries of unmapped target cells if specified or a mask is present
            if (
                (grid_out.mask is not None) and (grid_out.mask[0] is not None)
            ) or unmapped_to_nan is True:
                self.weights = add_nans_to_weights(self.weights)

            # follows legacy logic of writing weights if filename is provided
            if filename is not None and not reuse_weights:
                self.to_netcdf(filename=filename)

            # set default weights filename if none given
            self.filename = self._get_default_filename() if filename is None else filename

    @property
    def A(self):
        message = (
            'regridder.A is deprecated and will be removed in future versions. '
            'Use regridder.weights instead.'
        )

        warnings.warn(message, DeprecationWarning)
        # DeprecationWarning seems to be ignored by certain Python environments
        # Also print to make sure users notice this.
        print(message)
        return self.weights

    @property
    def w(self) -> xr.DataArray:
        """Return weights as a 4D DataArray with dimensions (y_out, x_out, y_in, x_in).

        ESMF stores the weights in a 2D array with dimensions (out_dim, in_dim), the size of the output and input
        grids respectively (ny x nx). This property returns the weights reshaped as a 4D array to simplify
        comparisons with the original grids.
        """
        # TODO: Add coords ?
        s = self.shape_out + self.shape_in
        data = self.weights.data.reshape(s)
        dims = 'y_out', 'x_out', 'y_in', 'x_in'
        return xr.DataArray(data, dims=dims)

    def _get_default_filename(self):
        # e.g. bilinear_400x600_300x400.nc
        filename = '{0}_{1}x{2}_{3}x{4}'.format(
            self.method,
            self.shape_in[0],
            self.shape_in[1],
            self.shape_out[0],
            self.shape_out[1],
        )

        if self.periodic:
            filename += '_peri.nc'
        else:
            filename += '.nc'

        return filename

    def _compute_weights(self, grid_in, grid_out):
        regrid = esmf_regrid_build(
            grid_in,
            grid_out,
            self.method,
            extrap_method=self.extrap_method,
            extrap_dist_exponent=self.extrap_dist_exponent,
            extrap_num_src_pnts=self.extrap_num_src_pnts,
            ignore_degenerate=self.ignore_degenerate,
        )

        w = regrid.get_weights_dict(deep_copy=True)
        esmf_regrid_finalize(regrid)  # only need weights, not regrid object
        return w

    def __call__(self, indata, keep_attrs=False, skipna=False, na_thres=1.0, output_chunks=None):
        """
        Apply regridding to input data.

        Parameters
        ----------
        indata : numpy array, dask array, xarray DataArray or Dataset.
            If not an xarray object or if `input_dìms` was not given in the init,
            the rightmost two dimensions must be the same as ``ds_in``.
            Can have arbitrary additional dimensions.

            Examples of valid shapes

            - (n_lat, n_lon), if ``ds_in`` has shape (n_lat, n_lon)
            - (n_time, n_lev, n_y, n_x), if ``ds_in`` has shape (Ny, n_x)

            Either give `input_dims` or transpose your input data
            if the horizontal dimensions are not the rightmost two dimensions

            Variables without the regridded dimensions are silently skipped when passing a Dataset.

        keep_attrs : bool, optional
            Keep attributes for xarray DataArrays or Datasets.
            Defaults to False.

        skipna: bool, optional
            Whether to skip missing values when regridding.
            When set to False, an output value is masked when a single
            input value is missing and no grid mask is provided.
            When set to True, missing values do not contaminate the regridding
            since only valid values are taken into account.
            In this case, a given output point is set to NaN only if the ratio
            of missing values exceeds the level set by `na_thres`:
            for instance, when the center of a cell is computed linearly
            from its four corners, one of which is missing, the output value
            is set to NaN if `na_thres` is smaller than 0.25.

        na_thres: float, optional
            A value within the [0., 1.] interval that defines the maximum
            ratio of missing grid points involved in the regrdding over which
            the output value is set to NaN. For instance, if `na_thres` is set
            to 0, the output value is NaN if a single NaN is found in the input
            values that are used to compute the output value; similarly,
            if `na_thres` is set to 1, all input values must be missing to
            mask the output value.

        output_chunks: dict or tuple, optional
            The desired chunks to have on the output along the spatial axes, if indata is a dask array.
            Other non-spatial axes inherit the same chunks as indata.
            Default behavior depends on the chunking of indata. If it is not chunked along
            the spatial dimension, the output will also not be chunked,
            equivalent to passing ``output_chunks=(-1, -1)``.
            If it is chunked, the output will preserve the chunk sizes,
            equivalent to passing ``output_chunks=ìndata.chunks``.
            Chunks have to be specified for all spatial dimensions
            of the output data otherwise regridding will fail. output_chunks can
            either be a tuple the same size as the spatial axes of outdata or it
            can be a dict with defined dims. If output_chunks is a dict, the
            keys must match the dims of the output grid passed when initializing this Regridder.

        Returns
        -------
        outdata : Data type is the same as input data type, except for datasets.
            On the same horizontal grid as ``ds_out``,
            with extra dims in ``dr_in``.

            Assuming ``ds_out`` has the shape of (n_y_out, n_x_out),
            examples of returning shapes are

            - (n_y_out, n_x_out), if ``dr_in`` is 2D
            - (n_time, n_lev, n_y_out, n_x_out), if ``dr_in`` has shape
              (n_time, n_lev, n_y, n_x)

            Datasets with dask-backed variables will have modified dtypes.
            If all input variables are 'float32', all output will be 'float32',
            for any other case, all outputs will be 'float64'.

        """
        if isinstance(indata, dask_array_type + (np.ndarray,)):
            return self.regrid_array(
                indata,
                self.weights.data,
                skipna=skipna,
                na_thres=na_thres,
                output_chunks=output_chunks,
            )
        elif isinstance(indata, xr.DataArray):
            return self.regrid_dataarray(
                indata,
                keep_attrs=keep_attrs,
                skipna=skipna,
                na_thres=na_thres,
                output_chunks=output_chunks,
            )
        elif isinstance(indata, xr.Dataset):
            return self.regrid_dataset(
                indata,
                keep_attrs=keep_attrs,
                skipna=skipna,
                na_thres=na_thres,
                output_chunks=output_chunks,
            )
        else:
            raise TypeError('input must be numpy array, dask array, xarray DataArray or Dataset!')

    @staticmethod
    def _regrid(indata, weights, *, shape_in, shape_out, skipna, na_thres):
        # skipna: set missing values to zero
        if skipna:
            missing = np.isnan(indata)
            indata = np.where(missing, 0.0, indata)

        # apply weights
        outdata = apply_weights(weights, indata, shape_in, shape_out)

        # skipna: Compute the influence of missing data at each interpolation point and filter those not meeting acceptable threshold.
        if skipna:
            fraction_valid = apply_weights(weights, (~missing).astype('d'), shape_in, shape_out)
            tol = 1e-6
            bad = fraction_valid < np.clip(1 - na_thres, tol, 1 - tol)
            fraction_valid[bad] = 1
            outdata = np.where(bad, np.nan, outdata / fraction_valid)

        return outdata

    def regrid_array(self, indata, weights, skipna=False, na_thres=1.0, output_chunks=None):
        """See __call__()."""
        if self.sequence_in:
            indata = np.reshape(indata, (*indata.shape[:-1], 1, indata.shape[-1]))

        # If output_chunk is dict, order output chunks to match order of out_horiz_dims and convert to tuple
        if isinstance(output_chunks, dict):
            output_chunks = tuple([output_chunks.get(key) for key in self.out_horiz_dims])

        kwargs = {
            'shape_in': self.shape_in,
            'shape_out': self.shape_out,
        }

        check_shapes(indata, weights, **kwargs)

        kwargs.update(skipna=skipna, na_thres=na_thres)

        weights = self.weights.data.reshape(self.shape_out + self.shape_in)
        if isinstance(indata, dask_array_type):  # dask
            if output_chunks is None:
                # Default : same chunk size as the input to preserve chunksize
                # Unless the input is not chunked along the dimension (shape_in == in_chunk_size), in which case we do not chunk along the dimension
                # This preserves the pre-0.8 behaviour.
                output_chunks = tuple(
                    min(chnkin, shpout) if shpin != chnkin else shpout
                    for shpout, shpin, chnkin in zip(
                        self.shape_out, self.shape_in, indata.chunksize[-2:]
                    )
                )
                fac = np.prod(
                    [np.ceil(shp / chnk) for shp, chnk in zip(self.shape_out, output_chunks)]
                )
                if fac > 4:  # Dask's built-in threshold is 10
                    warnings.warn(
                        (
                            f'Regridding is increasing the number of chunks by a factor of {fac}, '
                            'you might want to specify sizes in `output_chunks` in the regridder call. '
                            f'Default behaviour is to preserve the chunk sizes from the input {indata.chunksize[-2:]}.'
                        ),
                        da.core.PerformanceWarning,
                        stacklevel=3,
                    )
            if len(output_chunks) != len(self.shape_out):
                if len(output_chunks) == 1 and self.sequence_out:
                    output_chunks = (1, output_chunks[0])
                else:
                    raise ValueError(
                        f'output_chunks must have same dimension as ds_out,'
                        f' output_chunks dimension ({len(output_chunks)}) does not '
                        f'match ds_out dimension ({len(self.shape_out)})'
                    )
            weights = da.from_array(weights, chunks=(output_chunks + indata.chunksize[-2:]))
            outdata = self._regrid(indata, weights, **kwargs)
        else:  # numpy
            outdata = self._regrid(indata, weights, **kwargs)
        return outdata

    def regrid_numpy(self, indata, **kwargs):
        warnings.warn(
            '`regrid_numpy()` will be removed in xESMF 0.7, please use `regrid_array` instead.',
            category=FutureWarning,
        )
        return self.regrid_array(indata, self.weights.data, **kwargs)

    def regrid_dask(self, indata, **kwargs):
        warnings.warn(
            '`regrid_dask()` will be removed in xESMF 0.7, please use `regrid_array` instead.',
            category=FutureWarning,
        )
        return self.regrid_array(indata, self.weights.data, **kwargs)

    def regrid_dataarray(
        self, dr_in, keep_attrs=False, skipna=False, na_thres=1.0, output_chunks=None
    ):
        """See __call__()."""

        input_horiz_dims, temp_horiz_dims = self._parse_xrinput(dr_in)
        kwargs = dict(skipna=skipna, na_thres=na_thres, output_chunks=output_chunks)
        dr_out = xr.apply_ufunc(
            self.regrid_array,
            dr_in,
            self.weights,
            kwargs=kwargs,
            input_core_dims=[input_horiz_dims, ('out_dim', 'in_dim')],
            output_core_dims=[temp_horiz_dims],
            dask='allowed',
            keep_attrs=keep_attrs,
        )

        return self._format_xroutput(dr_out, temp_horiz_dims)

    def regrid_dataset(
        self, ds_in, keep_attrs=False, skipna=False, na_thres=1.0, output_chunks=None
    ):
        """See __call__()."""

        # get the first data variable to infer input_core_dims
        input_horiz_dims, temp_horiz_dims = self._parse_xrinput(ds_in)

        kwargs = dict(skipna=skipna, na_thres=na_thres, output_chunks=output_chunks)

        non_regriddable = [
            name
            for name, data in ds_in.data_vars.items()
            if not set(input_horiz_dims).issubset(data.dims)
        ]
        ds_in = ds_in.drop_vars(non_regriddable)

        ds_out = xr.apply_ufunc(
            self.regrid_array,
            ds_in,
            self.weights,
            kwargs=kwargs,
            input_core_dims=[input_horiz_dims, ('out_dim', 'in_dim')],
            output_core_dims=[temp_horiz_dims],
            dask='allowed',
            keep_attrs=keep_attrs,
        )

        return self._format_xroutput(ds_out, temp_horiz_dims)

    def _parse_xrinput(self, dr_in):
        # dr could be a DataArray or a Dataset
        # Get input horiz dim names and set output horiz dim names
        if self.in_horiz_dims is not None and all(dim in dr_in.dims for dim in self.in_horiz_dims):
            input_horiz_dims = self.in_horiz_dims
        else:
            if isinstance(dr_in, Dataset):
                name, dr_in = next(iter(dr_in.items()))
            else:
                # For warning purposes
                name = dr_in.name

            if self.sequence_in:
                input_horiz_dims = dr_in.dims[-1:]
            else:
                input_horiz_dims = dr_in.dims[-2:]

            # help user debugging invalid horizontal dimensions
            warnings.warn(
                (
                    f'Using dimensions {input_horiz_dims} from data variable {name} '
                    'as the horizontal dimensions for the regridding.'
                ),
                UserWarning,
            )

        if self.sequence_out:
            temp_horiz_dims = ['dummy', 'locations']
        else:
            temp_horiz_dims = [s + '_new' for s in input_horiz_dims]

        if self.sequence_in and not self.sequence_out:
            temp_horiz_dims = ['dummy_new'] + temp_horiz_dims
        return input_horiz_dims, temp_horiz_dims

    def _format_xroutput(self, out, new_dims=None):
        out.attrs['regrid_method'] = self.method
        return out

    def __repr__(self):
        info = (
            'xESMF Regridder \n'
            'Regridding algorithm:       {} \n'
            'Weight filename:            {} \n'
            'Reuse pre-computed weights? {} \n'
            'Input grid shape:           {} \n'
            'Output grid shape:          {} \n'
            'Periodic in longitude?      {}'.format(
                self.method,
                self.filename,
                self.reuse_weights,
                self.shape_in,
                self.shape_out,
                self.periodic,
            )
        )

        return info

    def to_netcdf(self, filename=None):
        """Save weights to disk as a netCDF file."""
        if filename is None:
            filename = self.filename
        w = self.weights.data
        dim = 'n_s'
        ds = xr.Dataset(
            {'S': (dim, w.data), 'col': (dim, w.coords[1, :] + 1), 'row': (dim, w.coords[0, :] + 1)}
        )
        ds.to_netcdf(filename)
        return filename


class Regridder(BaseRegridder):
    def __init__(
        self,
        ds_in,
        ds_out,
        method,
        locstream_in=False,
        locstream_out=False,
        periodic=False,
        parallel=False,
        **kwargs,
    ):
        """
        Make xESMF regridder

        Parameters
        ----------
        ds_in, ds_out : xarray Dataset, DataArray, or dictionary
            Contain input and output grid coordinates.
            All variables that the cf-xarray accessor understand are accepted.
            Otherwise, look for ``lon``, ``lat``,
            optionally ``lon_b``, ``lat_b`` for conservative methods,
            and ``mask``. Note that for `mask`, the ESMF convention is used,
            where masked values are identified by 0, and non-masked values by 1.

            For conservative methods, if bounds are not present, they will be
            computed using `cf-xarray` (only 1D coordinates are currently supported).

            Shape can be 1D (n_lon,) and (n_lat,) for rectilinear grids,
            or 2D (n_y, n_x) for general curvilinear grids.
            Shape of bounds should be (n+1,) or (n_y+1, n_x+1).
            CF-bounds (shape (n, 2) or (n, m, 4)) are also accepted if they are
            accessible through the cf-xarray accessor.

            If either dataset includes a 2d mask variable, that will also be
            used to inform the regridding.

            If DataArrays are passed, the are simply converted to Datasets.

        method : str
            Regridding method. Options are

            - 'bilinear'
            - 'conservative', **need grid corner information**
            - 'conservative_normed', **need grid corner information**
            - 'patch'
            - 'nearest_s2d'
            - 'nearest_d2s'

        periodic : bool, optional
            Periodic in longitude? Default to False.
            Only useful for global grids with non-conservative regridding.
            Will be forced to False for conservative regridding.

        parallel : bool, optional
            Compute the weights in parallel with Dask. Default to False.
            If True, weights are computed in parallel with Dask on subsets of the output grid using
            chunks of the output grid.

        filename : str, optional
            Name for the weight file. The default naming scheme is::

                {method}_{Ny_in}x{Nx_in}_{Ny_out}x{Nx_out}.nc

            e.g. bilinear_400x600_300x400.nc

        reuse_weights : bool, optional
            Whether to read existing weight file to save computing time.
            False by default (i.e. re-compute, not reuse).

        extrap_method : str, optional
            Extrapolation method. Options are

            - 'inverse_dist'
            - 'nearest_s2d'

        extrap_dist_exponent : float, optional
            The exponent to raise the distance to when calculating weights for the
            extrapolation method. If none are specified, defaults to 2.0

        extrap_num_src_pnts : int, optional
            The number of source points to use for the extrapolation methods
            that use more than one source point. If none are specified, defaults to 8

        weights : None, coo_matrix, dict, str, Dataset, Path,
            Regridding weights, stored as
              - a scipy.sparse COO matrix,
              - a dictionary with keys `row_dst`, `col_src` and `weights`,
              - an xarray Dataset with data variables `col`, `row` and `S`,
              - or a path to a netCDF file created by ESMF.

            If None, compute the weights.

        ignore_degenerate : bool, optional
            If False (default), raise error if grids contain degenerated cells
            (i.e. triangles or lines, instead of quadrilaterals)

        unmapped_to_nan: boolean, optional
            Set values of unmapped points to `np.nan` instead of zero (ESMF default). This is useful for
            target cells lying outside of the source domain when no output mask is defined.
            If an output mask is defined, or regridding method is `nearest_s2d` or `nearest_d2s`,
            this option has no effect.

        Returns
        -------
        regridder : xESMF regridder object
        """
        methods_avail_ls_in = ['nearest_s2d', 'nearest_d2s']
        methods_avail_ls_out = ['bilinear', 'patch'] + methods_avail_ls_in

        if locstream_in and method not in methods_avail_ls_in:
            raise ValueError(
                f'locstream input is only available for method in {methods_avail_ls_in}'
            )
        if locstream_out and method not in methods_avail_ls_out:
            raise ValueError(
                f'locstream output is only available for method in {methods_avail_ls_out}'
            )

        reuse_weights = kwargs.get('reuse_weights', False)

        weights = kwargs.get('weights', None)

        if parallel and (reuse_weights or weights is not None):
            parallel = False
            warnings.warn(
                'Cannot use parallel=True when reuse_weights=True or when weights is not None. Building Regridder normally.'
            )

        # Record basic switches
        if method in ['conservative', 'conservative_normed']:
            need_bounds = True
            periodic = False  # bound shape will not be N+1 for periodic grid
        else:
            need_bounds = False

        # Ensure we have Datasets and not DataArrays.
        if isinstance(ds_in, xr.DataArray):
            ds_in = ds_in._to_temp_dataset()

        if isinstance(ds_out, xr.DataArray):
            ds_out = ds_out._to_temp_dataset()

        # Construct ESMF grid, with some shape checking
        if locstream_in:
            grid_in, shape_in, input_dims = ds_to_ESMFlocstream(ds_in)
        else:
            grid_in, shape_in, input_dims = ds_to_ESMFgrid(
                ds_in, need_bounds=need_bounds, periodic=periodic
            )
        if locstream_out:
            grid_out, shape_out, output_dims = ds_to_ESMFlocstream(ds_out)
        else:
            grid_out, shape_out, output_dims = ds_to_ESMFgrid(ds_out, need_bounds=need_bounds)

        # Create the BaseRegridder
        super().__init__(
            grid_in,
            grid_out,
            method,
            input_dims=input_dims,
            output_dims=output_dims,
            parallel=parallel,
            **kwargs,
        )
        # Weights are computed, we do not need the grids anymore
        grid_in.destroy()
        grid_out.destroy()

        # Record output grid and metadata
        lon_out, lat_out = _get_lon_lat(ds_out)
        if not isinstance(lon_out, DataArray):
            if lon_out.ndim == 2:
                dims = [('y', 'x'), ('y', 'x')]
            elif self.sequence_out:
                dims = [('locations',), ('locations',)]
            else:
                dims = [('lon',), ('lat',)]
            lon_out = xr.DataArray(lon_out, dims=dims[0], name='lon', attrs=LON_CF_ATTRS)
            lat_out = xr.DataArray(lat_out, dims=dims[1], name='lat', attrs=LAT_CF_ATTRS)

        if lat_out.ndim == 2:
            self.out_horiz_dims = lat_out.dims
        elif self.sequence_out:
            if lat_out.dims != lon_out.dims:
                raise ValueError(
                    'Regridder expects a locstream output, but the passed longitude '
                    'and latitude are not specified along the same dimension. '
                    f'(lon: {lon_out.dims}, lat: {lat_out.dims})'
                )
            self.out_horiz_dims = ('dummy',) + lat_out.dims
        else:
            self.out_horiz_dims = (lat_out.dims[0], lon_out.dims[0])

        if isinstance(ds_out, Dataset):
            out_coords = ds_out.coords.to_dataset()
            grid_mapping = {
                var.attrs['grid_mapping']
                for var in ds_out.data_vars.values()
                if 'grid_mapping' in var.attrs
            }
            #  to keep : grid_mappings    and    non-scalar coords that have the spatial dims
            self.out_coords = out_coords.drop_vars(
                [
                    name
                    for name, crd in out_coords.coords.items()
                    if not (
                        (name in grid_mapping)
                        or (len(crd.dims) > 0 and set(self.out_horiz_dims).issuperset(crd.dims))
                    )
                ]
            )
        else:
            self.out_coords = xr.Dataset(coords={lat_out.name: lat_out, lon_out.name: lon_out})

        if parallel:
            self._init_para_regrid(ds_in, ds_out, kwargs)

    def _init_para_regrid(self, ds_in, ds_out, kwargs):
        # Check if we have bounds as variable and not coords, and add them to coords in both datasets
        if 'lon_b' in ds_out.data_vars:
            ds_out = ds_out.set_coords(['lon_b', 'lat_b'])
        if 'lon_b' in ds_in.data_vars:
            ds_in = ds_in.set_coords(['lon_b', 'lat_b'])
        if not (set(self.out_horiz_dims) - {'dummy'}).issubset(ds_out.chunksizes.keys()):
            raise ValueError(
                'Using `parallel=True` requires the output grid to have chunks along all spatial dimensions. '
                'If the dataset has no variables, consider adding an all-True spatial mask with appropriate chunks.'
            )
        # Drop everything in ds_out except mask or create mask if None. This is to prevent map_blocks loading unnecessary large data
        if self.sequence_out:
            ds_out_dims_drop = set(ds_out.variables).difference(ds_out.data_vars)
            ds_out = ds_out.drop_dims(ds_out_dims_drop)
        else:
            if 'mask' in ds_out:
                mask = ds_out.mask
                ds_out = ds_out.coords.to_dataset()
                ds_out['mask'] = mask
            else:
                ds_out_chunks = tuple([ds_out.chunksizes[i] for i in self.out_horiz_dims])
                ds_out = ds_out.coords.to_dataset()
                mask = da.ones(self.shape_out, dtype=bool, chunks=ds_out_chunks)
                ds_out['mask'] = (self.out_horiz_dims, mask)

            ds_out_dims_drop = set(ds_out.cf.coordinates.keys()).difference(
                ['longitude', 'latitude']
            )
            ds_out = ds_out.cf.drop_dims(ds_out_dims_drop)

        # Drop unnecessary variables in ds_in to save memory
        if not self.sequence_in:
            # Drop unnecessary dims
            ds_in_dims_drop = set(ds_in.cf.coordinates.keys()).difference(['longitude', 'latitude'])
            ds_in = ds_in.cf.drop_dims(ds_in_dims_drop)

            # Drop unnecessary vars
            ds_in = ds_in.coords.to_dataset()

        # Ensure ds_in is not dask-backed
        ds_in = ds_in.load()

        # if bounds in ds_out, we switch to cf bounds for map_blocks
        if 'lon_b' in ds_out and (ds_out.lon_b.ndim == ds_out.cf['longitude'].ndim):
            ds_out = ds_out.assign_coords(
                lon_bounds=cfxr.vertices_to_bounds(
                    ds_out.lon_b, ('bounds', *ds_out.cf['longitude'].dims)
                ),
                lat_bounds=cfxr.vertices_to_bounds(
                    ds_out.lat_b, ('bounds', *ds_out.cf['latitude'].dims)
                ),
            )
            # Make cf-xarray aware of the new bounds
            ds_out[ds_out.cf['longitude'].name].attrs['bounds'] = 'lon_bounds'
            ds_out[ds_out.cf['latitude'].name].attrs['bounds'] = 'lat_bounds'
            ds_out = ds_out.drop_dims(ds_out.lon_b.dims + ds_out.lat_b.dims)
        # rename dims to avoid map_blocks confusing ds_in and ds_out dims.
        if self.sequence_in:
            ds_in = ds_in.rename({self.in_horiz_dims[0]: 'x_in'})
        else:
            ds_in = ds_in.rename({self.in_horiz_dims[0]: 'y_in', self.in_horiz_dims[1]: 'x_in'})

        if self.sequence_out:
            ds_out = ds_out.rename({self.out_horiz_dims[1]: 'x_out'})
        else:
            ds_out = ds_out.rename(
                {self.out_horiz_dims[0]: 'y_out', self.out_horiz_dims[1]: 'x_out'}
            )

        out_chunks = {k: ds_out.chunks.get(k) for k in ['y_out', 'x_out']}
        in_chunks = {k: ds_in.chunks.get(k) for k in ['y_in', 'x_in']}
        chunks = out_chunks | in_chunks

        # Rename coords to avoid issues in xr.map_blocks
        # If coords and dims are the same, renaming has already been done.
        ds_out = ds_out.rename(
            {
                coord: coord + '_out'
                for coord in self.out_coords.coords.keys()
                if coord not in self.out_horiz_dims
            }
        )

        weights_dims = ('y_out', 'x_out', 'y_in', 'x_in')
        templ = sps.zeros((self.shape_out + self.shape_in))
        w_templ = xr.DataArray(templ, dims=weights_dims).chunk(
            chunks
        )  # template has same chunks as ds_out

        w = xr.map_blocks(
            subset_regridder,
            ds_out,
            args=[
                ds_in,
                self.method,
                self.in_horiz_dims,
                self.out_horiz_dims,
                self.sequence_in,
                self.sequence_out,
                self.periodic,
            ],
            kwargs=kwargs,
            template=w_templ,
        )
        w = w.compute(scheduler='processes')
        weights = w.stack(out_dim=weights_dims[:2], in_dim=weights_dims[2:])
        weights.name = 'weights'
        self.weights = weights

        # follows legacy logic of writing weights if filename is provided
        if 'filename' in kwargs:
            filename = kwargs['filename']
        else:
            filename = None
        if filename is not None and not self.reuse_weights:
            self.to_netcdf(filename=filename)

        # set default weights filename if none given
        self.filename = self._get_default_filename() if filename is None else filename

    def _format_xroutput(self, out, new_dims=None):
        if new_dims is not None:
            # rename dimension name to match output grid
            out = out.rename({nd: od for nd, od in zip(new_dims, self.out_horiz_dims)})

        out = out.assign_coords(self.out_coords.coords)
        out.attrs['regrid_method'] = self.method

        if self.sequence_out:
            out = out.squeeze(dim='dummy')

        return out


class SpatialAverager(BaseRegridder):
    def __init__(
        self,
        ds_in,
        polys,
        ignore_holes=False,
        periodic=False,
        filename=None,
        reuse_weights=False,
        weights=None,
        ignore_degenerate=False,
        geom_dim_name='geom',
    ):
        """Compute the exact average of a gridded array over a geometry.

        This uses the ESMF `conservative` regridding method to compute and apply weights
        mapping a 2D field unto geometries defined by polygons. The `conservative` method
        preserves the areal average of the input field. That is, *the value at each output
        grid cell is the average input value over the output grid area*. Here, the output
        grid cells are not rectangles defined by four corners, but polygons defined by
        multiple vertices (`ESMF.Mesh` objects). The regridding weights thus compute the
        areal-average of the input grid over each polygon.

        For multi-parts geometries (shapely.MultiPolygon), weights are computed for each
        geometry, then added, to compute the average over all geometries.

        When polygons include holes, the weights over the holes can either be substracted,
        or ignored.

        Parameters
        ----------
        ds_in : xr.DataArray or xr.Dataset or dictionary
            Contain input and output grid coordinates. Look for variables
            ``lon``, ``lat``, ``lon_b`` and ``lat_b``.

            Optionally looks for ``mask``, in which case  the ESMF convention is used,
            where masked values are identified by 0, and non-masked values by 1.

            Shape can be 1D (n_lon,) and (n_lat,) for rectilinear grids,
            or 2D (n_y, n_x) for general curvilinear grids.
            Shape of bounds should be (n+1,) or (n_y+1, n_x+1).
            DataArrays are converted to Datasets.

        polys : sequence of shapely Polygons and MultiPolygons
            Sequence of polygons (lon, lat) over which to average `ds_in`.

        ignore_holes : bool
            Whether to ignore holes in polygons.
            Default (True) is to subtract the weight of holes from the weight of the polygon.

        filename : str, optional
            Name for the weight file. The default naming scheme is::

                spatialavg_{Ny_in}x{Nx_in}_{Npoly_out}.nc

            e.g. spatialavg_400x600_30.nc

        reuse_weights : bool, optional
            Whether to read existing weight file to save computing time.
            False by default (i.e. re-compute, not reuse).

        weights : None, coo_matrix, dict, str, Dataset, Path,
            Regridding weights, stored as
              - a scipy.sparse COO matrix,
              - a dictionary with keys `row_dst`, `col_src` and `weights`,
              - an xarray Dataset with data variables `col`, `row` and `S`,
              - or a path to a netCDF file created by ESMF.

            If None, compute the weights.

        ignore_degenerate : bool, optional
            If False (default), raise error if grids contain degenerated cells
            (i.e. triangles or lines, instead of quadrilaterals)

        self.geom_dim_name : str
            Name of dimension along which averages for each polygon are stored.

        Returns
        -------
        xarray.DataArray
          Average over polygons along `geom_dim_name` dimension. The `lon` and
          `lat` coordinates are the polygon centroid coordinates.

        References
        ----------
        This approach is inspired by `OCGIS <https://github.com/NCPP/ocgis>`_.
        """
        # Note, I suggest we refactor polys -> geoms
        self.ignore_holes = ignore_holes
        self.polys = polys
        self.geom_dim_name = geom_dim_name

        # Ensure we have a Dataset
        if isinstance(ds_in, xr.DataArray):
            ds_in = ds_in._to_temp_dataset()

        grid_in, shape_in, input_dims = ds_to_ESMFgrid(ds_in, need_bounds=True, periodic=periodic)

        # Create an output locstream so that the regridder knows the output shape and coords.
        # Latitude and longitude coordinates are the polygon centroid.
        lon_out, lat_out = _get_lon_lat(ds_in)
        if hasattr(lon_out, 'name'):
            self._lon_out_name = lon_out.name
            self._lat_out_name = lat_out.name
        else:
            self._lon_out_name = 'lon'
            self._lat_out_name = 'lat'

        # Check length of polys segments
        self._check_polys_length(polys)

        poly_centers = [poly.centroid.xy for poly in polys]
        self._lon_out = np.asarray([c[0][0] for c in poly_centers])
        self._lat_out = np.asarray([c[1][0] for c in poly_centers])

        # We put names 'lon' and 'lat' so ds_to_ESMFlocstream finds them easily.
        # _lon_out_name and _lat_out_name are used on the output anyway.
        ds_out = {'lon': self._lon_out, 'lat': self._lat_out}
        locstream_out, shape_out, _ = ds_to_ESMFlocstream(ds_out)

        # BaseRegridder with custom-computed weights and dummy out grid
        super().__init__(
            grid_in,
            locstream_out,
            'conservative',
            input_dims=input_dims,
            weights=weights,
            filename=filename,
            reuse_weights=reuse_weights,
            ignore_degenerate=ignore_degenerate,
            unmapped_to_nan=False,
        )
        # Weights are computed, we do not need the grids anymore
        grid_in.destroy()
        locstream_out.destroy()

    @staticmethod
    def _check_polys_length(polys, threshold=1):
        # Check length of polys segments, issue warning if too long
        check_polys, check_holes, _, _ = split_polygons_and_holes(polys)
        check_polys.extend(check_holes)
        poly_segments = []
        for check_poly in check_polys:
            b = check_poly.boundary.coords
            # Length of each segment
            poly_segments.extend([LineString(b[k : k + 2]).length for k in range(len(b) - 1)])
        if np.any(np.array(poly_segments) > threshold):
            warnings.warn(
                f'`polys` contains large (> {threshold}°) segments. This could lead to errors over large regions. For a more accurate average, segmentize (densify) your shapes with  `shapely.segmentize(polys, {threshold})`',
                UserWarning,
                stacklevel=2,
            )

    def _compute_weights_and_area(self, grid_in, mesh_out):
        """Return the weights and the area of the destination mesh cells."""

        # Build the regrid object
        regrid = esmf_regrid_build(
            grid_in,
            mesh_out,
            method='conservative',
            ignore_degenerate=self.ignore_degenerate,
        )

        # Get the weights and convert to a DataArray
        weights = regrid.get_weights_dict(deep_copy=True)
        w = _parse_coords_and_values(weights, self.n_in, mesh_out.element_count)

        # Get destination area - important for renormalizing the subgeometries.
        regrid.dstfield.get_area()
        dstarea = regrid.dstfield.data.copy()

        return w, dstarea

    def _compute_weights(self, grid_in, grid_out):
        """Return weight sparse matrix.

        This function first explodes the geometries into a flat list of Polygon exterior objects:
          - Polygon -> polygon.exterior
          - MultiPolygon -> list of polygon.exterior

        and a list of Polygon.interiors (holes).

        Individual meshes are created for the exteriors and the interiors, and their regridding weights computed.
        We cannot compute the exterior and interior weights at the same time, because the meshes overlap.

        Weights for the subgeometries are then aggregated back to the original geometries. Because exteriors and
        interiors are computed independently, we need to normalize the weights according to their area.
        """

        # Explode geometries into a flat list of polygon exteriors and interiors.
        # Keep track of original geometry index.
        # The convention used here is to list the exteriors first and then the interiors.
        exteriors, interiors, i_ext, i_int = split_polygons_and_holes(self.polys)
        geom_indices = np.array(i_ext + i_int)

        # Create mesh from external polygons (positive contribution)
        mesh_ext = Mesh.from_polygons(exteriors)

        # Get weights for external polygons
        w, area = self._compute_weights_and_area(grid_in, mesh_ext)
        mesh_ext.destroy()  # release mesh memory

        # Get weights for interiors and append them to weights from exteriors as a negative contribution.
        if len(interiors) > 0 and not self.ignore_holes:
            mesh_int = Mesh.from_polygons(interiors)

            # Get weights for interiors
            w_int, area_int = self._compute_weights_and_area(grid_in, mesh_int)
            mesh_int.destroy()  # release mesh memory

            # Append weights from holes as negative weights
            # In sparse >= 0.16, a fill_value of -0.0 is different from 0.0 and the concat would fail
            inv_w_int = -w_int
            inv_w_int.data.fill_value = 0.0
            w = xr.concat((w, inv_w_int), 'out_dim')

            # Append areas
            area = np.concatenate([area, area_int])

        # Combine weights for all the subgeometries belonging to the same geometry
        return _combine_weight_multipoly(w, area, geom_indices).T

    @property
    def w(self) -> xr.DataArray:
        """Return weights as a 3D DataArray with dimensions (geom, y_in, x_in).

        ESMF stores the weights in a 2D array with dimensions (out_dim, in_dim), the size of the output and input
        grids respectively (ny x nx). This property returns the weights reshaped as a 3D array to simplify
        comparisons with the original grids.
        """
        s = self.shape_out[1:2] + self.shape_in
        data = self.weights.data.reshape(s)
        dims = self.geom_dim_name, 'y_in', 'x_in'
        return xr.DataArray(data, dims=dims)

    def _get_default_filename(self):
        # e.g. bilinear_400x600_300x400.nc
        filename = 'spatialavg_{0}x{1}_{2}.nc'.format(
            self.shape_in[0], self.shape_in[1], self.n_out
        )

        return filename

    def __repr__(self):
        info = (
            'xESMF SpatialAverager \n'
            'Weight filename:            {} \n'
            'Reuse pre-computed weights? {} \n'
            'Input grid shape:           {} \n'
            'Output list length:         {} \n'.format(
                self.filename, self.reuse_weights, self.shape_in, self.n_out
            )
        )

        return info

    def _format_xroutput(self, out, new_dims=None):
        out = out.squeeze(dim='dummy')

        # rename dimension name to match output grid
        out = out.rename(locations=self.geom_dim_name)

        # append output horizontal coordinate values
        # extra coordinates are automatically tracked by apply_ufunc
        out.coords[self._lon_out_name] = xr.DataArray(self._lon_out, dims=(self.geom_dim_name,))
        out.coords[self._lat_out_name] = xr.DataArray(self._lat_out, dims=(self.geom_dim_name,))
        out.attrs['regrid_method'] = self.method
        return out
