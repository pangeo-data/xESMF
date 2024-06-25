import warnings

import numpy as np
import xarray as xr
from shapely.geometry import MultiPolygon, Polygon

try:
    import esmpy as ESMF
except ImportError:
    import ESMF

LON_CF_ATTRS = {'standard_name': 'longitude', 'units': 'degrees_east'}
LAT_CF_ATTRS = {'standard_name': 'latitude', 'units': 'degrees_north'}


def _grid_1d(start_b, end_b, step):
    """
    1D grid centers and bounds

    Parameters
    ----------
    start_b, end_b : float
        start/end position. Bounds, not centers.

    step: float
        step size, i.e. grid resolution

    Returns
    -------
    centers : 1D numpy array

    bounds : 1D numpy array, with one more element than centers
    """

    bounds = np.arange(start_b, end_b + step / 2, step)
    centers = (bounds[:-1] + bounds[1:]) / 2

    return centers, bounds


def grid_2d(lon0_b, lon1_b, d_lon, lat0_b, lat1_b, d_lat):
    """
    2D rectilinear grid centers and bounds

    Parameters
    ----------
    lon0_b, lon1_b : float
        Longitude bounds

    d_lon : float
        Longitude step size, i.e. grid resolution

    lat0_b, lat1_b : float
        Latitude bounds

    d_lat : float
        Latitude step size, i.e. grid resolution

    Returns
    -------
    ds : xarray DataSet with coordinate values

    """

    lon_1d, lon_b_1d = _grid_1d(lon0_b, lon1_b, d_lon)
    lat_1d, lat_b_1d = _grid_1d(lat0_b, lat1_b, d_lat)

    lon, lat = np.meshgrid(lon_1d, lat_1d)
    lon_b, lat_b = np.meshgrid(lon_b_1d, lat_b_1d)

    ds = xr.Dataset(
        coords={
            'lon': (['y', 'x'], lon, {'standard_name': 'longitude'}),
            'lat': (['y', 'x'], lat, {'standard_name': 'latitude'}),
            'lon_b': (['y_b', 'x_b'], lon_b),
            'lat_b': (['y_b', 'x_b'], lat_b),
        }
    )

    return ds


def cf_grid_2d(lon0_b, lon1_b, d_lon, lat0_b, lat1_b, d_lat):
    """
    CF compliant 2D rectilinear grid centers and bounds.

    Parameters
    ----------
    lon0_b, lon1_b : float
        Longitude bounds

    d_lon : float
        Longitude step size, i.e. grid resolution

    lat0_b, lat1_b : float
        Latitude bounds

    d_lat : float
        Latitude step size, i.e. grid resolution

    Returns
    -------
    ds : xarray.DataSet with coordinate values

    """
    from cf_xarray import vertices_to_bounds

    lon_1d, lon_b_1d = _grid_1d(lon0_b, lon1_b, d_lon)
    lat_1d, lat_b_1d = _grid_1d(lat0_b, lat1_b, d_lat)

    ds = xr.Dataset(
        coords={
            'lon': (
                'lon',
                lon_1d,
                {'bounds': 'lon_bounds', **LON_CF_ATTRS},
            ),
            'lat': (
                'lat',
                lat_1d,
                {'bounds': 'lat_bounds', **LAT_CF_ATTRS},
            ),
            'latitude_longitude': xr.DataArray(),
        },
        data_vars={
            'lon_bounds': vertices_to_bounds(lon_b_1d, ('bound', 'lon')),
            'lat_bounds': vertices_to_bounds(lat_b_1d, ('bound', 'lat')),
        },
    )

    return ds


def grid_global(d_lon, d_lat, cf=False, lon1=180):
    """
    Global 2D rectilinear grid centers and bounds

    Parameters
    ----------
    d_lon : float
      Longitude step size, i.e. grid resolution
    d_lat : float
      Latitude step size, i.e. grid resolution
    cf : bool
      Return a CF compliant grid.
    lon1 : {180, 360}
      Right longitude bound. According to which convention is used longitudes will
      vary from -180 to 180 or from 0 to 360.

    Returns
    -------
    ds : xarray DataSet with coordinate values

    """

    if not np.isclose(360 / d_lon, 360 // d_lon):
        warnings.warn(
            '360 cannot be divided by d_lon = {}, '
            'might not cover the globe uniformly'.format(d_lon)
        )

    if not np.isclose(180 / d_lat, 180 // d_lat):
        warnings.warn(
            '180 cannot be divided by d_lat = {}, '
            'might not cover the globe uniformly'.format(d_lat)
        )
    lon0 = lon1 - 360

    if cf:
        return cf_grid_2d(lon0, lon1, d_lon, -90, 90, d_lat)

    return grid_2d(lon0, lon1, d_lon, -90, 90, d_lat)


def _flatten_poly_list(polys):
    """Iterator flattening MultiPolygons."""
    for i, poly in enumerate(polys):
        if isinstance(poly, MultiPolygon):
            for sub_poly in poly.geoms:
                yield (i, sub_poly)
        else:
            yield (i, poly)


def split_polygons_and_holes(polys):
    """Split the exterior boundaries and the holes for a list of polygons.

    If MultiPolygons are encountered in the list, they are flattened out
    in their constituents.

    Parameters
    ----------
    polys : Sequence of shapely Polygons or MultiPolygons

    Returns
    -------
    exteriors : list of Polygons
        The polygons without any holes
    holes : list of Polygons
        Holes of the polygons as polygons
    i_ext : list of integers
       The index in `polys` of each polygon in `exteriors`.
    i_hol : list of integers
       The index in `polys` of the owner of each hole in `holes`.
    """
    exteriors = []
    holes = []
    i_ext = []
    i_hol = []
    for i, poly in _flatten_poly_list(polys):
        exteriors.append(Polygon(poly.exterior))
        i_ext.append(i)
        holes.extend(map(Polygon, poly.interiors))
        i_hol.extend([i] * len(poly.interiors))

    return exteriors, holes, i_ext, i_hol


# Constants
PI_180 = np.pi / 180.0
_default_Re = 6371.0e3  # MIDAS
HUGE = 1.0e30


def simple_tripolar_grid(nlons, nlats, lat_cap=60, lon_cut=-300):
    """Generate a simple tripolar grid, regular under `lat_cap`.

    Parameters
    ----------
    nlons: int
      Number of longitude points.
    nlats: int
      Number of latitude points.
    lat_cap: float
      Latitude of the northern cap.
    lon_cut: float
      Longitude of the periodic boundary.

    """

    # first generate the bipolar cap for north poles
    nj_cap = np.rint(nlats * lat_cap / 180.0).astype('int')

    lams, phis, _, _ = _generate_bipolar_cap_mesh(
        nlons, nj_cap, lat_cap, lon_cut, ensure_nj_even=True
    )

    # then extend south
    lams_south_1d = lams[0, :]
    phis_south_1d = np.linspace(-90, lat_cap, nlats - nj_cap + 1)[:-1]

    lams_south, phis_south = np.meshgrid(lams_south_1d, phis_south_1d)

    # concatenate the 2 parts
    lon = np.concatenate([lams_south, lams], axis=0)
    lat = np.concatenate([phis_south, phis], axis=0)

    return lon, lat


# these functions are copied from https://github.com/NOAA-GFDL/ocean_model_grid_generator
# rather than using the package as a dependency


def _bipolar_projection(lamg, phig, lon_bp, rp, metrics_only=False):
    """Makes a stereographic bipolar projection of the input coordinate mesh (lamg,phig)
    Returns the projected coordinate mesh and their metric coefficients (h^-1).
    The input mesh must be a regular spherical grid capping the pole with:
        latitudes between 2*arctan(rp) and 90  degrees
        longitude between lon_bp       and lonp+360
    """
    # symmetry meridian resolution fix
    phig = 90 - 2 * np.arctan(np.tan(0.5 * (90 - phig) * PI_180) / rp) / PI_180
    tmp = _mdist(lamg, lon_bp) * PI_180
    sinla = np.sin(tmp)  # This makes phis symmetric
    sphig = np.sin(phig * PI_180)
    alpha2 = (np.cos(tmp)) ** 2  # This makes dy symmetric
    beta2_inv = (np.tan(phig * PI_180)) ** 2
    rden = 1.0 / (1.0 + alpha2 * beta2_inv)

    if not metrics_only:
        B = sinla * np.sqrt(rden)  # Actually two equations  +- |B|
        # Deal with beta=0
        B = np.where(np.abs(beta2_inv) > HUGE, 0.0, B)
        lamc = np.arcsin(B) / PI_180
        # But this equation accepts 4 solutions for a given B, {l, 180-l, l+180, 360-l }
        # We have to pickup the "correct" root.
        # One way is simply to demand lamc to be continuous with lam on the equator phi=0
        # I am sure there is a more mathematically concrete way to do this.
        lamc = np.where((lamg - lon_bp > 90) & (lamg - lon_bp <= 180), 180 - lamc, lamc)
        lamc = np.where((lamg - lon_bp > 180) & (lamg - lon_bp <= 270), 180 + lamc, lamc)
        lamc = np.where((lamg - lon_bp > 270), 360 - lamc, lamc)
        # Along symmetry meridian choose lamc
        lamc = np.where(
            (lamg - lon_bp == 90), 90, lamc
        )  # Along symmetry meridian choose lamc=90-lon_bp
        lamc = np.where(
            (lamg - lon_bp == 270), 270, lamc
        )  # Along symmetry meridian choose lamc=270-lon_bp
        lams = lamc + lon_bp

    # Project back onto the larger (true) sphere so that the projected equator shrinks to latitude \phi_P=lat0_tp
    # then we have tan(\phi_s'/2)=tan(\phi_p'/2)tan(\phi_c'/2)
    A = sinla * sphig
    chic = np.arccos(A)
    phis = 90 - 2 * np.arctan(rp * np.tan(chic / 2)) / PI_180
    # Calculate the Metrics
    rden2 = 1.0 / (1 + (rp * np.tan(chic / 2)) ** 2)
    M_inv = rp * (1 + (np.tan(chic / 2)) ** 2) * rden2
    chig = (90 - phig) * PI_180
    rden2 = 1.0 / (1 + (rp * np.tan(chig / 2)) ** 2)
    N = rp * (1 + (np.tan(chig / 2)) ** 2) * rden2
    N_inv = 1 / N
    cos2phis = (np.cos(phis * PI_180)) ** 2

    h_j_inv_t1 = cos2phis * alpha2 * (1 - alpha2) * beta2_inv * (1 + beta2_inv) * (rden**2)
    h_j_inv_t2 = M_inv * M_inv * (1 - alpha2) * rden
    h_j_inv = h_j_inv_t1 + h_j_inv_t2

    # Deal with beta=0. Prove that cos2phis/alpha2 ---> 0 when alpha, beta  ---> 0
    h_j_inv = np.where(np.abs(beta2_inv) > HUGE, M_inv * M_inv, h_j_inv)
    h_j_inv = np.sqrt(h_j_inv) * N_inv

    h_i_inv = cos2phis * (1 + beta2_inv) * (rden**2) + M_inv * M_inv * alpha2 * beta2_inv * rden
    # Deal with beta=0
    h_i_inv = np.where(np.abs(beta2_inv) > HUGE, M_inv * M_inv, h_i_inv)
    h_i_inv = np.sqrt(h_i_inv)

    if not metrics_only:
        return lams, phis, h_i_inv, h_j_inv
    else:
        return h_i_inv, h_j_inv


def _generate_bipolar_cap_mesh(Ni, Nj_ncap, lat0_bp, lon_bp, ensure_nj_even=True):
    # Define a (lon,lat) coordinate mesh on the Northern hemisphere of the globe sphere
    # such that the resolution of latg matches the desired resolution of the final grid along the symmetry meridian
    print('Generating bipolar grid bounded at latitude ', lat0_bp)
    if Nj_ncap % 2 != 0 and ensure_nj_even:
        print('   Supergrid has an odd number of area cells!')
        if ensure_nj_even:
            print("   The number of j's is not even. Fixing this by cutting one row.")
            Nj_ncap = Nj_ncap - 1

    lon_g = lon_bp + np.arange(Ni + 1) * 360.0 / float(Ni)
    lamg = np.tile(lon_g, (Nj_ncap + 1, 1))
    latg0_cap = lat0_bp + np.arange(Nj_ncap + 1) * (90 - lat0_bp) / float(Nj_ncap)
    phig = np.tile(latg0_cap.reshape((Nj_ncap + 1, 1)), (1, Ni + 1))
    rp = np.tan(0.5 * (90 - lat0_bp) * PI_180)
    lams, phis, h_i_inv, h_j_inv = _bipolar_projection(lamg, phig, lon_bp, rp)
    h_i_inv = h_i_inv[:, :-1] * 2 * np.pi / float(Ni)
    h_j_inv = h_j_inv[:-1, :] * PI_180 * (90 - lat0_bp) / float(Nj_ncap)
    print('   number of js=', phis.shape[0])
    return lams, phis, h_i_inv, h_j_inv


def _mdist(x1, x2):
    """Returns positive distance modulo 360."""
    return np.minimum(np.mod(x1 - x2, 360.0), np.mod(x2 - x1, 360.0))


# end code from https://github.com/NOAA-GFDL/ocean_model_grid_generator


def cell_area(ds, earth_radius=None):
    """
    Get cell area of a grid, assuming a sphere.

    Parameters
    ----------
    ds : xarray Dataset
        Input grid, longitude and latitude required.
        Curvilinear coordinate system also require cell bounds to be present.
    earth_radius : float, optional
        Earth radius, assuming a sphere, in km.

    Returns
    -------
    area : xarray DataArray
        Cell area. If the earth radius is given, units are km^2, otherwise they are steradian (sr).
    """
    from .frontend import _get_lon_lat, ds_to_ESMFgrid  # noqa

    grid, _, names = ds_to_ESMFgrid(ds, need_bounds=True)
    field = ESMF.Field(grid)
    field.get_area()  # compute area

    # F-ordering to C-ordering
    # copy the array to make sure it persists after ESMF object is freed
    area = field.data.T.copy()
    field.destroy()

    # Wrap in xarray
    area = xr.DataArray(
        area,
        dims=names,
        attrs={
            'units': 'sr',
            'standard_name': 'cell_area',
            'long_name': 'Cell area, assuming a sphere.',
        },
    )
    # Fancy trick to get all related coordinates without needing to list them explicitly.
    # We add all and let xarray choose which one to keep when selecting the variable
    area = ds.coords.to_dataset().assign(area=area).area

    if earth_radius is not None:
        area = (area * earth_radius**2).assign_attrs(units='km2')
    return area
