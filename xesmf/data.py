"""
Standard test data for regridding benchmark.
"""

import numpy as np
import xarray


def wave_smooth(  # type: ignore
    lon: np.ndarray[float] | xarray.DataArray,  # type: ignore
    lat: np.ndarray[float] | xarray.DataArray,  # type: ignore
) -> np.ndarray[float] | xarray.DataArray:  # type: ignore
    """
    Spherical harmonic with low frequency.

    Parameters
    ----------
    lon, lat : 2D numpy array or xarray DataArray
            Longitute/Latitude of cell centers

    Returns
    -------
    f : 2D numpy array or xarray DataArray depending on input2D wave field

    Notes
    -------
    Equation from [1]_ [2]_:

    .. math:: Y_2^2 = 2 + cos^2(lat) * cos(2 * lon)

    References
    ----------
    .. [1] Jones, P. W. (1999). First-and second-order conservative remapping
        schemes for grids in spherical coordinates. Monthly Weather Review,
        127(9), 2204-2210.

    .. [2] Ullrich, P. A., Lauritzen, P. H., & Jablonowski, C. (2009).
        Geometrically exact conservative remapping (GECoRe): regular
        latitude-longitude and cubed-sphere grids. Monthly Weather Review,
        137(6), 1721-1741.
    """
    # degree to radius, make a copy
    lat *= np.pi / 180.0  # type: ignore
    lon *= np.pi / 180.0  # type: ignore

    f = 2 + pow(np.cos(lat), 2) * np.cos(2 * lon)  # type: ignore
    return f
