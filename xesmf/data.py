"""
Standard test data for regridding benchmark.
"""

from typing import Any
import numpy as np
import numpy.typing as npt
import xarray


def wave_smooth(  # type: ignore
    lon: npt.NDArray[np.floating[Any]] | xarray.DataArray,
    lat: npt.NDArray[np.floating[Any]] | xarray.DataArray,
) -> npt.NDArray[np.floating[Any]] | xarray.DataArray:
    """
    Spherical harmonic with low frequency.

    Parameters
    ----------
    lon, lat : 2D numpy array or xarray DataArray
            Longitude/Latitude of cell centers

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
    lat *= np.pi / 180.0
    lon *= np.pi / 180.0

    f = 2 + pow(np.cos(lat), 2) * np.cos(2 * lon)
    return f
