"""
Backend for xESMF. This module wraps ESMPy's complicated API and can create
ESMF Grid and Regrid objects only using basic numpy arrays.

General idea:

1) Only use pure numpy array in this low-level backend. xarray should only be
used in higher-level APIs which interface with this low-level backend.

2) Use simple, procedural programming here. Because ESMPy Classes are
complicated enough, building new Classes will make debugging very difficult.

3) Add some basic error checking in this wrapper level.
ESMPy is hard to debug because the program often dies in the Fortran level.
So it would be helpful to catch some common mistakes in Python level.
"""

import os
import warnings
from collections.abc import Sequence

try:
    import esmpy as ESMF
except ImportError:
    import ESMF
import numpy as np
import numpy.lib.recfunctions as nprec


def warn_f_contiguous(a):
    """
    Give a warning if input array if not Fortran-ordered.

    ESMPy expects Fortran-ordered array. Passing C-ordered array will slow down
    performance due to memory rearrangement.

    Parameters
    ----------
    a : numpy array
    """
    if not a.flags['F_CONTIGUOUS']:
        warnings.warn('Input array is not F_CONTIGUOUS. ' 'Will affect performance.')


def warn_lat_range(lat):
    """
    Give a warning if latitude is outside of [-90, 90]

    Longitute, on the other hand, can be in any range,
    since the it the transform is done in (x, y, z) space.

    Parameters
    ----------
    lat : numpy array
    """
    if (lat.max() > 90.0) or (lat.min() < -90.0):
        warnings.warn('Latitude is outside of [-90, 90]')


class Grid(ESMF.Grid):
    @classmethod
    def from_xarray(cls, lon, lat, periodic=False, mask=None):
        """
        Create an ESMF.Grid object, for constructing ESMF.Field and ESMF.Regrid.

        Parameters
        ----------
        lon, lat : 2D numpy array
             Longitute/Latitude of cell centers.

             Recommend Fortran-ordering to match ESMPy internal.

             Shape should be ``(Nlon, Nlat)`` for rectilinear grid,
             or ``(Nx, Ny)`` for general quadrilateral grid.

        periodic : bool, optional
            Periodic in longitude? Default to False.
            Only useful for source grid.

        mask : 2D numpy array, optional
            Grid mask. According to the ESMF convention, masked cells
            are set to 0 and unmasked cells to 1.

            Shape should be ``(Nlon, Nlat)`` for rectilinear grid,
            or ``(Nx, Ny)`` for general quadrilateral grid.

        Returns
        -------
        grid : ESMF.Grid object
        """

        # ESMPy expects Fortran-ordered array.
        # Passing C-ordered array will slow down performance.
        for a in [lon, lat]:
            warn_f_contiguous(a)

        warn_lat_range(lat)

        # ESMF.Grid can actually take 3D array (lon, lat, radius),
        # but regridding only works for 2D array
        assert lon.ndim == 2, 'Input grid must be 2D array'
        assert lon.shape == lat.shape, 'lon and lat must have same shape'

        staggerloc = ESMF.StaggerLoc.CENTER  # actually just integer 0

        if periodic:
            num_peri_dims = 1
        else:
            num_peri_dims = None

        # ESMPy documentation claims that if staggerloc and coord_sys are None,
        # they will be set to default values (CENTER and SPH_DEG).
        # However, they actually need to be set explicitly,
        # otherwise grid._coord_sys and grid._staggerloc will still be None.
        grid = cls(
            np.array(lon.shape),
            staggerloc=staggerloc,
            coord_sys=ESMF.CoordSys.SPH_DEG,
            num_peri_dims=num_peri_dims,
        )

        # The grid object points to the underlying Fortran arrays in ESMF.
        # To modify lat/lon coordinates, need to get pointers to them
        lon_pointer = grid.get_coords(coord_dim=0, staggerloc=staggerloc)
        lat_pointer = grid.get_coords(coord_dim=1, staggerloc=staggerloc)

        # Use [...] to avoid overwritting the object. Only change array values.
        lon_pointer[...] = lon
        lat_pointer[...] = lat

        # Follows SCRIP convention where 1 is unmasked and 0 is masked.
        # See https://github.com/NCPP/ocgis/blob/61d88c60e9070215f28c1317221c2e074f8fb145/src/ocgis/regrid/base.py#L391-L404
        if mask is not None:
            # remove fractional values
            mask = np.where(mask == 0, 0, 1)
            # convert array type to integer (ESMF compat)
            grid_mask = mask.astype(np.int32)
            if not (grid_mask.shape == lon.shape):
                raise ValueError(
                    'mask must have the same shape as the latitude/longitude '
                    'coordinates, got: mask.shape = %s, lon.shape = %s' % (mask.shape, lon.shape)
                )
            grid.add_item(ESMF.GridItem.MASK, staggerloc=ESMF.StaggerLoc.CENTER, from_file=False)
            grid.mask[0][:] = grid_mask

        return grid

    def get_shape(self, loc=ESMF.StaggerLoc.CENTER):
        """Return shape of grid for specified StaggerLoc"""
        # We cast explicitly to python's int (numpy >=2)
        return tuple(map(int, self.size[loc]))


class LocStream(ESMF.LocStream):
    @classmethod
    def from_xarray(cls, lon, lat):
        """
        Create an ESMF.LocStream object, for contrusting ESMF.Field and ESMF.Regrid

        Parameters
        ----------
        lon, lat : 1D numpy array
             Longitute/Latitude of cell centers.

        Returns
        -------
        locstream : ESMF.LocStream object
        """

        if len(lon.shape) > 1:
            raise ValueError('lon can only be 1d')
        if len(lat.shape) > 1:
            raise ValueError('lat can only be 1d')

        assert lon.shape == lat.shape

        location_count = len(lon)

        locstream = cls(location_count, coord_sys=ESMF.CoordSys.SPH_DEG)

        locstream['ESMF:Lon'] = lon.astype(np.dtype('f8'))
        locstream['ESMF:Lat'] = lat.astype(np.dtype('f8'))

        return locstream

    def get_shape(self):
        """Return LocStream shape."""
        return (self.size, 1)


def add_corner(grid, lon_b, lat_b):
    """
    Add corner information to ESMF.Grid for conservative regridding.

    Not needed for other methods like bilinear or nearest neighbour.

    Parameters
    ----------
    grid : ESMF.Grid object
        Generated by ``Grid.from_xarray()``. Will be modified in-place.

    lon_b, lat_b : 2D numpy array
        Longitute/Latitude of cell corner
        Recommend Fortran-ordering to match ESMPy internal.
        Shape should be ``(Nlon+1, Nlat+1)``, or ``(Nx+1, Ny+1)``
    """

    # codes here are almost the same as Grid.from_xarray(),
    # except for the "staggerloc" keyword
    staggerloc = ESMF.StaggerLoc.CORNER  # actually just integer 3

    for a in [lon_b, lat_b]:
        warn_f_contiguous(a)

    warn_lat_range(lat_b)

    assert lon_b.ndim == 2, 'Input grid must be 2D array'
    assert lon_b.shape == lat_b.shape, 'lon_b and lat_b must have same shape'
    assert np.array_equal(lon_b.shape, grid.max_index + 1), 'lon_b should be size (Nx+1, Ny+1)'
    assert (grid.num_peri_dims == 0) and (
        grid.periodic_dim is None
    ), 'Cannot add corner for periodic grid'

    grid.add_coords(staggerloc=staggerloc)

    lon_b_pointer = grid.get_coords(coord_dim=0, staggerloc=staggerloc)
    lat_b_pointer = grid.get_coords(coord_dim=1, staggerloc=staggerloc)

    lon_b_pointer[...] = lon_b
    lat_b_pointer[...] = lat_b


class Mesh(ESMF.Mesh):
    @staticmethod
    def _lonlat_to_xyz(lon_deg, lat_deg):
        lon = np.radians(lon_deg)
        lat = np.radians(lat_deg)
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)
        return x, y, z

    @classmethod
    def from_ugrid(
        cls,
        node_lon,
        node_lat,
        face_node_connectivity,
        face_lon,
        face_lat,
        fill_value=None,
        start_index=None,
    ):
        """
        Create an ESMF.Mesh from UGRID-style lon/lat mesh coordinates.

        Parameters
        ----------
        node_lon, node_lat : array-like
            One-dimensional longitude and latitude coordinates of mesh nodes.
        face_node_connectivity : array-like
            Two-dimensional integer array with shape ``(n_face, n_max_face_nodes)``
            mapping each face to its corner nodes.
        face_lon, face_lat : array-like
            One-dimensional longitude and latitude coordinates of face centers.
        fill_value : int, optional
            Fill value used for padded connectivity entries. Defaults to ``-1``.
        start_index : {0, 1}, optional
            Index offset used by ``face_node_connectivity``. If omitted, the offset
            is inferred from the connectivity values.

        Returns
        -------
        mesh
            ESMF.Mesh object.

        Notes
        -----
        Longitude and latitude coordinates are converted to Cartesian coordinates on
        the unit sphere before constructing the ESMF mesh. The input face-node
        ordering is preserved.
        """

        node_lon = np.asarray(node_lon, dtype=np.float64)
        node_lat = np.asarray(node_lat, dtype=np.float64)
        face_node_connectivity = np.asarray(face_node_connectivity, dtype=np.int64)
        face_lon = np.asarray(face_lon, dtype=np.float64)
        face_lat = np.asarray(face_lat, dtype=np.float64)

        if node_lon.ndim != 1 or node_lat.ndim != 1:
            raise ValueError('node_lon and node_lat must be 1D')
        if node_lon.shape != node_lat.shape:
            raise ValueError('node_lon and node_lat must have the same shape')
        if face_node_connectivity.ndim != 2:
            raise ValueError('face_node_connectivity must be 2D')
        if face_lon.ndim != 1 or face_lat.ndim != 1:
            raise ValueError('face_lon and face_lat must be 1D')
        if face_lon.shape != face_lat.shape:
            raise ValueError('face_lon and face_lat must have the same shape')
        if face_lon.size != face_node_connectivity.shape[0]:
            raise ValueError('face coordinates must match number of faces')

        if fill_value is None:
            fill_value = -1

        valid = face_node_connectivity != fill_value
        n_nodes_per_face = valid.sum(axis=1).astype(np.int32)

        if np.any(n_nodes_per_face < 3):
            raise ValueError('each face must contain at least 3 valid nodes')

        conn = face_node_connectivity.copy()

        if start_index is None:
            valid_conn = conn[valid]
            if valid_conn.size == 0:
                raise ValueError('face_node_connectivity contains no valid entries')
            start_index = 0 if valid_conn.min() == 0 else 1

        conn[valid] = conn[valid] - start_index

        if conn[valid].min() < 0 or conn[valid].max() >= node_lon.size:
            raise ValueError('face_node_connectivity contains out-of-range node indices')

        node_x, node_y, node_z = cls._lonlat_to_xyz(node_lon, node_lat)
        face_x, face_y, face_z = cls._lonlat_to_xyz(face_lon, face_lat)

        mesh = cls(parametric_dim=2, spatial_dim=3)

        num_node = node_lon.size
        node_ids = np.arange(1, num_node + 1, dtype=np.int32)
        node_coords = np.column_stack((node_x, node_y, node_z)).ravel()
        node_owners = np.zeros(num_node, dtype=np.int32)

        mesh.add_nodes(num_node, node_ids, node_coords, node_owners)

        elem_count = face_node_connectivity.shape[0]
        elem_ids = np.arange(1, elem_count + 1, dtype=np.int32)
        elem_types = n_nodes_per_face.astype(np.int32)

        flat_conn = []
        for row, n in zip(conn, n_nodes_per_face):
            flat_conn.extend(np.asarray(row[:n], dtype=np.int32).tolist())

        elem_conn = np.asarray(flat_conn, dtype=np.int32)
        elem_coords = np.column_stack((face_x, face_y, face_z)).ravel()

        mesh.add_elements(
            elem_count,
            elem_ids,
            elem_types,
            elem_conn,
            element_coords=elem_coords,
        )

        return mesh

    @classmethod
    def from_ugrid_xyz(
        cls,
        node_x,
        node_y,
        node_z,
        face_node_connectivity,
        face_x,
        face_y,
        face_z,
        fill_value=None,
        start_index=None,
    ):
        """
        Create an ESMF.Mesh from UGRID-style Cartesian mesh coordinates.

        Parameters
        ----------
        node_x, node_y, node_z : array-like
            One-dimensional Cartesian coordinates of mesh nodes.
        face_node_connectivity : array-like
            Two-dimensional integer array with shape ``(n_face, n_max_face_nodes)``
            mapping each face to its corner nodes.
        face_x, face_y, face_z : array-like
            One-dimensional Cartesian coordinates of face centers.
        fill_value : int, optional
            Fill value used for padded connectivity entries. Defaults to ``-1``.
        start_index : {0, 1}, optional
            Index offset used by ``face_node_connectivity``. If omitted, the offset
            is inferred from the connectivity values.

        Returns
        -------
        mesh
            ESMF.Mesh object.

        Notes
        -----
        The input face-node ordering is preserved.
        """

        node_x = np.asarray(node_x, dtype=np.float64)
        node_y = np.asarray(node_y, dtype=np.float64)
        node_z = np.asarray(node_z, dtype=np.float64)
        face_node_connectivity = np.asarray(face_node_connectivity, dtype=np.int64)
        face_x = np.asarray(face_x, dtype=np.float64)
        face_y = np.asarray(face_y, dtype=np.float64)
        face_z = np.asarray(face_z, dtype=np.float64)

        if node_x.ndim != 1 or node_y.ndim != 1 or node_z.ndim != 1:
            raise ValueError('node_x, node_y, and node_z must be 1D')
        if not (node_x.shape == node_y.shape == node_z.shape):
            raise ValueError('node_x, node_y, and node_z must have the same shape')
        if face_node_connectivity.ndim != 2:
            raise ValueError('face_node_connectivity must be 2D')
        if face_x.ndim != 1 or face_y.ndim != 1 or face_z.ndim != 1:
            raise ValueError('face_x, face_y, and face_z must be 1D')
        if not (face_x.shape == face_y.shape == face_z.shape):
            raise ValueError('face_x, face_y, and face_z must have the same shape')
        if face_x.size != face_node_connectivity.shape[0]:
            raise ValueError('face coordinates must match number of faces')

        if fill_value is None:
            fill_value = -1

        valid = face_node_connectivity != fill_value
        n_nodes_per_face = valid.sum(axis=1).astype(np.int32)

        if np.any(n_nodes_per_face < 3):
            raise ValueError('each face must contain at least 3 valid nodes')

        conn = face_node_connectivity.copy()

        if start_index is None:
            valid_conn = conn[valid]
            if valid_conn.size == 0:
                raise ValueError('face_node_connectivity contains no valid entries')
            start_index = 0 if valid_conn.min() == 0 else 1

        conn[valid] = conn[valid] - start_index

        if conn[valid].min() < 0 or conn[valid].max() >= node_x.size:
            raise ValueError('face_node_connectivity contains out-of-range node indices')

        mesh = cls(parametric_dim=2, spatial_dim=3)

        num_node = node_x.size
        node_ids = np.arange(1, num_node + 1, dtype=np.int32)
        node_coords = np.column_stack((node_x, node_y, node_z)).ravel()
        node_owners = np.zeros(num_node, dtype=np.int32)

        mesh.add_nodes(num_node, node_ids, node_coords, node_owners)

        elem_count = face_node_connectivity.shape[0]
        elem_ids = np.arange(1, elem_count + 1, dtype=np.int32)
        elem_types = n_nodes_per_face.astype(np.int32)

        flat_conn = []
        for row, n in zip(conn, n_nodes_per_face):
            flat_conn.extend(np.asarray(row[:n], dtype=np.int32).tolist())

        elem_conn = np.asarray(flat_conn, dtype=np.int32)
        elem_coords = np.column_stack((face_x, face_y, face_z)).ravel()

        mesh.add_elements(
            elem_count,
            elem_ids,
            elem_types,
            elem_conn,
            element_coords=elem_coords,
        )

        return mesh

    @classmethod
    def from_polygons(cls, polys, element_coords='centroid'):
        """
        Create an ESMF.Mesh object from a list of polygons.

        All exterior ring points are added to the mesh as nodes and each polygon
        is added as an element, with the polygon centroid as the element's coordinates.

        Parameters
        ----------
        polys : sequence of shapely Polygon
           Holes are not represented by the Mesh.
        element_coords : array or "centroid", optional
            If "centroid", the polygon centroids will be used (default)
            If an array of shape (len(polys), 2) : the element coordinates of the mesh.
            If None, the Mesh's elements will not have coordinates.

        Returns
        -------
        mesh : ESMF.Mesh
            A mesh where each polygon is represented as an Element.
        """
        node_num = sum(len(e.exterior.coords) - 1 for e in polys)
        elem_num = len(polys)

        # Pre alloc arrays. Special structure for coords makes the code faster.
        crd_dt = np.dtype([('x', np.float32), ('y', np.float32)])
        node_coords = np.empty(node_num, dtype=crd_dt)
        node_coords[:] = (np.nan, np.nan)  # Fill with impossible values

        element_types = np.empty(elem_num, dtype=np.uint32)
        element_conn = np.empty(node_num, dtype=np.uint32)

        # Flag for centroid calculation
        calc_centroid = isinstance(element_coords, str) and element_coords == 'centroid'
        if calc_centroid:
            element_coords = np.empty(elem_num, dtype=crd_dt)

        inode = 0
        iconn = 0
        for ipoly, poly in enumerate(polys):
            ring = poly.exterior
            if calc_centroid:
                element_coords[ipoly] = poly.centroid.coords[0]
            element_types[ipoly] = len(ring.coords) - 1
            for coord in ring.coords[:-1] if ring.is_ccw else ring.coords[:0:-1]:
                crd = np.asarray(coord, dtype=crd_dt)  # Cast so we can compare
                node_index = np.where(node_coords == crd)[0]
                if node_index.size == 0:  # New node
                    node_coords[inode] = crd
                    element_conn[iconn] = inode
                    inode += 1
                else:  # Node already exists
                    element_conn[iconn] = node_index[0]
                iconn += 1

        node_num = inode  # With duplicate nodes, inode < node_num

        mesh = cls(2, 2, coord_sys=ESMF.CoordSys.SPH_DEG)
        mesh.add_nodes(
            node_num,
            np.arange(node_num) + 1,
            nprec.structured_to_unstructured(node_coords[:node_num]).ravel(),
            np.zeros(node_num),
        )

        if calc_centroid:
            element_coords = nprec.structured_to_unstructured(element_coords)
        if element_coords is not None:
            element_coords = element_coords.ravel()

        try:
            mesh.add_elements(
                elem_num,
                np.arange(elem_num) + 1,
                element_types,
                element_conn,
                element_coords=element_coords,
            )
        except ValueError as err:
            raise ValueError(
                'ESMF failed to create the Mesh, this usually happen when some polygons are invalid (test with `poly.is_valid`)'
            ) from err

        return mesh

    def get_shape(self, loc=ESMF.MeshLoc.ELEMENT):
        """Return the shape of the Mesh at specified MeshLoc location."""
        return (self.size[loc], 1)


def esmf_regrid_build(  # noqa: C901
    sourcegrid,
    destgrid,
    method,
    filename=None,
    extra_dims=None,
    extrap_method=None,
    extrap_dist_exponent=None,
    extrap_num_src_pnts=None,
    extrap_num_levels=None,
    ignore_degenerate=None,
    vector_regrid=None,
):
    """
    Create an ESMF.Regrid object, containing regridding weights.

    Parameters
    ----------
    sourcegrid, destgrid : ESMF.Grid or ESMF.Mesh object
        Source and destination grids.

        Should create them by ``Grid.from_xarray()``
        (with optionally ``add_corner()``),
        instead of ESMPy's original API.

    method : str
        Regridding method. Options are

        - 'bilinear'
        - 'conservative', **need grid corner information**
        - 'conservative_normed', **need grid corner information**
        - 'patch'
        - 'nearest_s2d'
        - 'nearest_d2s'

    filename : str, optional
        Offline weight file. **Require ESMPy 7.1.0.dev38 or newer.**
        With the weights available, we can use Scipy's sparse matrix
        multiplication to apply weights, which is faster and more Pythonic
        than ESMPy's online regridding. If None, weights are stored in
        memory only.

    extra_dims : a list of integers, optional
        Extra dimensions (e.g. time or levels) in the data field

        This does NOT affect offline weight file, only affects online regrid.

        Extra dimensions will be stacked to the fastest-changing dimensions,
        i.e. following Fortran-like instead of C-like conventions.
        For example, if extra_dims=[Nlev, Ntime], then the data field dimension
        will be [Nlon, Nlat, Nlev, Ntime]

    extrap_method : str, optional
        Extrapolation method. Options are

        - 'inverse_dist'
        - 'nearest_s2d'
        - 'creep_fill'

    extrap_dist_exponent : float, optional
        The exponent to raise the distance to when calculating weights for the
        extrapolation method. If none are specified, defaults to 2.0

    extrap_num_src_pnts : int, optional
        The number of source points to use for the extrapolation methods
        that use more than one source point. If none are specified, defaults to 8

    extrap_num_levels : int, optional
        Number of extrapolation levels to apply for the 'creep_fill' method.

        The creep fill algorithm iteratively fills unmapped target points by
        propagating values from neighboring mapped cells. Each level corresponds
        to one iteration of this filling process. Larger values allow extrapolation
        to reach farther into unmapped regions, but may increase computational cost
        and smoothness of the result.

        Required when ``extrap_method='creep_fill'``.

    ignore_degenerate : bool, optional
        If False (default), raise error if grids contain degenerated cells
        (i.e. triangles or lines, instead of quadrilaterals)

    vector_regrid : bool, optional
        If True, treat a single extra (non-spatial) dimension in the source and
        destination data fields as the components of a vector. (If True and
        there is more than one extra dimension in either the source or
        destination data fields, an error will be raised.) If not specified,
        defaults to False.

        Only vector dimensions of size 2 are supported. The first entry is
        interpreted as the east component and the second as the north component.
        i.e., ``extra_dims`` must be ``[2]``.

        Requires ESMPy 8.9.0 or newer.

    Returns
    -------
    regrid : ESMF.Regrid object

    """

    # use shorter, clearer names for options in ESMF.RegridMethod
    method_dict = {
        'bilinear': ESMF.RegridMethod.BILINEAR,
        'conservative': ESMF.RegridMethod.CONSERVE,
        'conservative_normed': ESMF.RegridMethod.CONSERVE,
        'patch': ESMF.RegridMethod.PATCH,
        'nearest_s2d': ESMF.RegridMethod.NEAREST_STOD,
        'nearest_d2s': ESMF.RegridMethod.NEAREST_DTOS,
    }
    try:
        esmf_regrid_method = method_dict[method]
    except Exception:
        raise ValueError('method should be chosen from ' '{}'.format(list(method_dict.keys())))

    # use shorter, clearer names for options in ESMF.ExtrapMethod
    extrap_dict = {
        'inverse_dist': ESMF.ExtrapMethod.NEAREST_IDAVG,
        'nearest_s2d': ESMF.ExtrapMethod.NEAREST_STOD,
        'creep_fill': ESMF.ExtrapMethod.CREEP_FILL,
        None: None,
    }
    try:
        esmf_extrap_method = extrap_dict[extrap_method]
    except KeyError:
        raise KeyError(
            '`extrap_method` should be chosen from ' '{}'.format(list(extrap_dict.keys()))
        )
    # CREEP_FILL requires a finite number of fill levels and is unsupported
    # for conservative regridding methods.
    if extrap_method == 'creep_fill':
        if extrap_num_levels is None:
            raise ValueError(
                '`extrap_num_levels` must be provided when `extrap_method="creep_fill"`.'
            )
        if method in ['conservative', 'conservative_normed']:
            raise ValueError(
                '`extrap_method="creep_fill"` is not supported with conservative regridding methods.'
            )
    # until ESMPy updates ESMP_FieldRegridStoreFile, extrapolation is not possible
    # if files are written on disk
    if (extrap_method is not None) & (filename is not None):
        raise ValueError('`extrap_method` cannot be used along with `filename`.')

    # conservative regridding needs cell corner information
    if method in ['conservative', 'conservative_normed']:
        if not isinstance(sourcegrid, ESMF.Mesh) and not sourcegrid.has_corners:
            raise ValueError(
                'source grid has no corner information. ' 'cannot use conservative regridding.'
            )
        if not isinstance(destgrid, ESMF.Mesh) and not destgrid.has_corners:
            raise ValueError(
                'destination grid has no corner information. ' 'cannot use conservative regridding.'
            )

    if vector_regrid:
        # Check this ESMPy requirement in order to give a more helpful error message if it
        # isn't met
        if not (isinstance(extra_dims, Sequence) and len(extra_dims) == 1 and extra_dims[0] == 2):
            raise ValueError('`vector_regrid` currently requires `extra_dims` to be `[2]`')

    # ESMF.Regrid requires Field (Grid+data) as input, not just Grid.
    # Extra dimensions are specified when constructing the Field objects,
    # not when constructing the Regrid object later on.
    if isinstance(sourcegrid, ESMF.Mesh):
        sourcefield = ESMF.Field(sourcegrid, meshloc=ESMF.MeshLoc.ELEMENT, ndbounds=extra_dims)
    else:
        sourcefield = ESMF.Field(sourcegrid, ndbounds=extra_dims)
    if isinstance(destgrid, ESMF.Mesh):
        destfield = ESMF.Field(destgrid, meshloc=ESMF.MeshLoc.ELEMENT, ndbounds=extra_dims)
    else:
        destfield = ESMF.Field(destgrid, ndbounds=extra_dims)

    # ESMF bug? when using locstream objects, options src_mask_values
    # and dst_mask_values produce runtime errors
    allow_masked_values = True
    if isinstance(sourcefield.grid, ESMF.LocStream):
        allow_masked_values = False
    if isinstance(destfield.grid, ESMF.LocStream):
        allow_masked_values = False

    # ESMPy will throw an incomprehensive error if the weight file
    # already exists. Better to catch it here!
    if filename is not None:
        assert not os.path.exists(
            filename
        ), 'Weight file already exists! Please remove it or use a new name.'

    # re-normalize conservative regridding results
    # https://github.com/JiaweiZhuang/xESMF/issues/17
    if method == 'conservative_normed':
        norm_type = ESMF.NormType.FRACAREA
    else:
        norm_type = ESMF.NormType.DSTAREA

    # Calculate regridding weights.
    # Must set unmapped_action to IGNORE, otherwise the function will fail,
    # if the destination grid is larger than the source grid.
    kwargs = dict(
        filename=filename,
        regrid_method=esmf_regrid_method,
        unmapped_action=ESMF.UnmappedAction.IGNORE,
        ignore_degenerate=ignore_degenerate,
        norm_type=norm_type,
        extrap_method=esmf_extrap_method,
        extrap_dist_exponent=extrap_dist_exponent,
        extrap_num_src_pnts=extrap_num_src_pnts,
        extrap_num_levels=extrap_num_levels,
        factors=filename is None,
    )
    if allow_masked_values:
        kwargs.update(dict(src_mask_values=[0], dst_mask_values=[0]))
    # Only add the vector_regrid argument if it is given and true; this supports backwards
    # compatibility with versions of ESMPy prior to 8.9.0 that do not have this option.
    # (If the user explicitly sets the vector_regrid argument, however, that will still be
    # passed through; this will lead to an exception with older ESMPy versions.)
    if vector_regrid:
        kwargs['vector_regrid'] = vector_regrid

    regrid = ESMF.Regrid(sourcefield, destfield, **kwargs)

    return regrid


def esmf_regrid_apply(regrid, indata):
    """
    Apply existing regridding weights to the data field,
    using ESMPy's built-in functionality.

    xESMF use Scipy to apply weights instead of this.
    This is only for benchmarking Scipy's result and performance.

    Parameters
    ----------
    regrid : ESMF.Regrid object
        Contains the mapping from the source grid to the destination grid.

        Users should create them by esmf_regrid_build(),
        instead of ESMPy's original API.

    indata : numpy array of shape ``(Nlon, Nlat, N1, N2, ...)``
        Extra dimensions ``(N1, N2, ...)`` are specified in
        ``esmf_regrid_build()``.

        Recommend Fortran-ordering to match ESMPy internal.

    Returns
    -------
    outdata : numpy array of shape ``(Nlon_out, Nlat_out, N1, N2, ...)``

    """

    # Passing C-ordered input data will be terribly slow,
    # since indata is often quite large and re-ordering memory is expensive.
    warn_f_contiguous(indata)

    # Get the pointers to source and destination fields.
    # Because the regrid object points to its underlying field&grid,
    # we can just pass regrid from ESMF_regrid_build() to ESMF_regrid_apply(),
    # without having to pass all the field&grid objects.
    sourcefield = regrid.srcfield
    destfield = regrid.dstfield

    # pass numpy array to the underlying Fortran array
    sourcefield.data[...] = indata

    # apply regridding weights
    destfield = regrid(sourcefield, destfield)

    return destfield.data


def esmf_regrid_finalize(regrid):
    """
    Free the underlying Fortran array to avoid memory leak.

    After calling ``destroy()`` on regrid or its fields, we cannot use the
    regrid method anymore, but the input and output data still exist.

    Parameters
    ----------
    regrid : ESMF.Regrid object

    """
    # We do not destroy the Grids here, as they might be reused between multiple regrids
    regrid.destroy()
    regrid.srcfield.destroy()
    regrid.dstfield.destroy()

    # double check
    assert regrid.finalized
    assert regrid.srcfield.finalized
    assert regrid.dstfield.finalized


# Deprecated as of version 0.5.0


def esmf_locstream(lon, lat):
    warnings.warn(
        '`esmf_locstream` is being deprecated in favor of `LocStream.from_xarray`',
        DeprecationWarning,
    )
    return LocStream.from_xarray(lon, lat)


def esmf_grid(lon, lat, periodic=False, mask=None):
    warnings.warn(
        '`esmf_grid` is being deprecated in favor of `Grid.from_xarray`', DeprecationWarning
    )
    return Grid.from_xarray(lon, lat)
