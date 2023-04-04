What's new
==========

0.7.1 (2023-04-03)
------------------

Bug fixes
~~~~~~~~~
* Fix ``Mesh.from_polygons`` and unpin Shapely to add support for Shapely 2.0 (:pull:`219`). By `Pascal Bourgault <https://github.com/aulemahal>`_
* Implement workaround for setup conda problem (:pull:`229`). By `Raphael Dussin <https://github.com/raphaeldussin>`_
* Update CI and doc - fix for DataArrays (:pull:`230`). By `Pascal Bourgault <https://github.com/aulemahal>`_
* Fix ci/cd badge for build status (:pull:`231`). By `Pierre Manchon <https://github.com/pierre-manchon>`_
* Update CI for Micromamba environments (:pull:`233`). By `Trevor James Smith <https://github.com/Zeitsperre>`_
* Fix error in test with Shapely 2.0 (:pull:`251`). By `David Huard <https://github.com/huard>`_

New features
~~~~~~~~~~~~
* Add util to build tripolar grid (:pull:`228`). By `Raphael Dussin <https://github.com/raphaeldussin>`_

Documentation
~~~~~~~~~~~~~
* Document installation options for ESMpy (:pull:`241`). By `Matthew Plough <https://github.com/mplough-kobold>`_

Internal changes
~~~~~~~~~~~~~~~~
* Modernize the package configuration / publish to PyPI (:pull:`248`). By `Filipe Fernandes <https://github.com/ocefpaf>`_


0.7.0 (2022-12-16)
------------------

Bug fixes
~~~~~~~~~
- Fix bug in `util.grid_global` where grid centers could go beyond 180 degrees (:issue:`181`). By `David Huard <https://github.com/huard>`_

New features
~~~~~~~~~~~~
- Support both [-180, 180] and [0, 360] conventions in `grid_global` (:issue:`149`). By `David Huard <https://github.com/huard>`_


Documentation
~~~~~~~~~~~~~
- Fix API doc build (:pull:`194`). By `David Huard <https://github.com/huard>`_
- Include `conservative_normed` into the notebook comparing regridding algorithms. By `David Huard <https://github.com/huard>`_
- Fix typos (:pull:`191`). By `Jemma Stachelek <https://github.com/jsta>`_
- Copy-editing (:pull:`178`, :pull:`179`). By `RichardScottOZ <https://github.com/RichardScottOZ>`_

Internal changes
~~~~~~~~~~~~~~~~
- Constrain `numba>=0.55.2`. See (:issue:`185`).
- Constrain `shapely<2.0`. See (:issue:`216`).
- Add support for esmpy name change in import. See (:pull:`214`,:issue:`212`)


0.6.3 (29-06-2022)
------------------

Bug fixes
~~~~~~~~~
- Spatial coordinates of `ds_out` are kept within the regridder and transferred to the regridded DataArray or Dataset (:pull:`175`). By `Pascal Bourgault <https://github.com/aulemahal>`_
- Added `numba` as an explicit dependency to fix installation with conda (:pull:`168`). By `Pascal Bourgault <https://github.com/aulemahal>`_

Internal changes
~~~~~~~~~~~~~~~~
- Use `cf-xarray` to guess missing CF coordinates before extracting bounds (:pull:`147`). By `Pascal Bourgault <https://github.com/aulemahal>`_


0.6.2 (23-11-2021)
------------------

Bug fixes
~~~~~~~~~
- The introduction of `sparse`, with `numba` under the hood, restricted input data to little-endian dtypes. For big-endian dtypes, xESMF will convert to little-endian, regrid and convert back (:pull:`135`). By `Pascal Bourgault <https://github.com/aulemahal>`_
- ``SpatialAverager`` did not compute the same weights as ``Regridder`` when source cell areas were not uniform (:pull:`128`). By `David Huard <https://github.com/huard>`_
- Refactor of how the regridding is called internally, to fix a bug with dask and sparse (:pull:`135`). By `Pascal Bourgault <https://github.com/aulemahal>`_

Internal changes
~~~~~~~~~~~~~~~~
- Deprecation of ``regrid_numpy`` and ``regrid_dask`` is scheduled for 0.7.0. All checks on shape, array layout and numba support are now done at call time, rather then at computation time (:pull:`135`).

0.6.1 (23-09-2021)
------------------
Note that this version creates very large dask task graphs that can affect performance for large grids.

Internal changes
~~~~~~~~~~~~~~~~
- Weights are now stored in a ``xr.DataArray`` backed by ``sparse.COO``, which allows to pass them as an argument to the ``xr.apply_ufunc`` and decrease memory usage when using dask. By `Pascal Bourgault <https://github.com/aulemahal>`_
- New dependency `sparse <https://sparse.pydata.org>`_ replacing ``scipy``.


0.6.0 (07-08-2021)
------------------

New features
~~~~~~~~~~~~
- Add the ``skipna`` and ``na_threshold`` options to deal with masks over non-spatial dimensions (:pull:`29`). This is useful when, for example, masks vary over time. By `Stéphane Raynaud <https://github.com/stefraynaud>`_
- Add ``unmapped_to_nan`` argument to regridder frontend. When True, this sets target cells outside the source domain to NaN instead of zero for all regridding methods except nearest neighbour (:pull:`94`). By `Martin Schupfner <https://github.com/sol1105>`_

Bug fixes
~~~~~~~~~
- Drop the PyPi badge and replace by a Conda badge (:pull:`97`). By `Ray Bell <https://github.com/raybellwaves>`_


0.5.3 (04-12-2021)
------------------

Bug fixes
~~~~~~~~~
- Fix regression regarding support for non-CF-compliant coordinate names (:pull:`73`). By `Sam Levang <https://github.com/slevang>`_
- Infer `bounds` dimension name using cf-xarray (:pull:`78`). By `Pascal Bourgault <https://github.com/aulemahal>`_
- Do not regrid variables that are not defined over horizontal dimensions (:pull:`79`). By `Pascal Bourgault <https://github.com/aulemahal>`_
- Ensure locstream dimension name is consistent with `ds_out` (:pull:`81`). By `Mattia Almansi  <https://github.com/malmans2>`_

Documentation
~~~~~~~~~~~~~
- Add release instructions (:pull:`75`). By `David Huard <https://github.com/huard>`_
- Update Zenodo DOI badge


0.5.2 (01-20-2021)
------------------

Bug fixes
~~~~~~~~~

* Restore original behavior for lon/lat discovery, uses cf-xarray if lon/lat not found in dataset (:pull:`64`)
* Solve issue of dimension order in dataset (#53) with (:pull:`66`)

0.5.1 (01-11-2021)
------------------

Documentation
~~~~~~~~~~~~~
* Update installation instructions to mention that PyPi only holds xesmf up to version 0.3.0.

New features
~~~~~~~~~~~~
* Regridded xarray.Dataset now preserves the name and attributes of target coordinates (:pull:`60`)

Bug fixes
~~~~~~~~~
* Fix doc build for API/Regridder (:pull:`61`)


0.5.0 (11-11-2020)
------------------

Breaking changes
~~~~~~~~~~~~~~~~
* Deprecate `esmf_grid` in favor of `Grid.from_xarray`
* Deprecate `esmf_locstream` in favor of `LocStream.from_xarray`
* Installation requires numpy>=1.16 and cf-xarray>=0.3.1

New features
~~~~~~~~~~~~
* Create `ESMF.Mesh` objects from `shapely.polygons` (:pull:`24`). By `Pascal Bourgault <https://github.com/aulemahal>`_
* New class `SpatialAverager` offers user-friendly mechanism to average a 2-D field over a polygon. Includes support to handle interior holes and multi-part geometries. (:pull:`24`) By `Pascal Bourgault <https://github.com/aulemahal>`_
* Automatic detection of coordinates and computation of vertices based on cf-xarray. (:pull:`49`) By `Pascal Bourgault <https://github.com/aulemahal>`_

Bug fixes
~~~~~~~~~
* Fix serialization bug when using dask's distributed scheduler (:pull:`39`).
  By `Pascal Bourgault <https://github.com/aulemahal>`_.

Internal changes
~~~~~~~~~~~~~~~~
* Subclass `ESMF.Mesh` and create `from_polygon` method
* Subclass `ESMF.Grid` and `ESMF.LocStream` and create `from_xarray` methods.
* New `BaseRegridder` class, with support for `Grid`, `LocStream` and `Mesh` objects. Not all regridding methods are supported for `Mesh` objects.
* Refactor `Regridder` to subclass `BaseRegridder`.


0.4.0 (01-10-2020)
------------------
The git repo is now hosted by pangeo-data (https://github.com/pangeo-data/xESMF)

Breaking changes
~~~~~~~~~~~~~~~~
* By default, weights are not written to disk, but instead kept in memory.
* Installation requires ESMPy 8.0.0 and up.

New features
~~~~~~~~~~~~
* The `Regridder` object now takes a `weights` argument accepting a scipy.sparse COO matrix,
  a dictionary, an xarray.Dataset, or a path to a netCDF file created by ESMF. If None, weights
  are computed and can be written to disk using the `to_netcdf` method. This `weights` parameter
  replaces the `filename` and `reuse_weights` arguments, which are preserved for backward compatibility (:pull:`3`).
  By `David Huard <https://github.com/huard>`_ and `Raphael Dussin <https://github.com/raphaeldussin>`_
* Added documentation discussion how to compute weights from a shell using MPI, and reuse from xESMF (:pull:`12`).
  By `Raphael Dussin <https://github.com/raphaeldussin>`_
* Add support for masks in :py:func`esmf_grid`. This avoid NaNs to bleed into the interpolated values.
  When using a mask and the `conservative` regridding method, use a new method called
  `conservative_normed` to properly handle normalization (:pull:`1`).
  By `Raphael Dussin <https://github.com/raphaeldussin>`_


0.3.0 (06-03-2020)
------------------

New features
~~~~~~~~~~~~
* Add support for `ESMF.LocStream` `(#81) <https://github.com/JiaweiZhuang/xESMF/pull/81>`_
  By `Raphael Dussin <https://github.com/raphaeldussin>`_


0.2.2 (07-10-2019)
------------------

New features
~~~~~~~~~~~~
* Add option to allow degenerated grid cells `(#61) <https://github.com/JiaweiZhuang/xESMF/pull/61>`_
  By `Jiawei Zhuang <https://github.com/JiaweiZhuang>`_


0.2.0 (04-08-2019)
------------------

Breaking changes
~~~~~~~~~~~~~~~~
All user-facing APIs in v0.1.x should still work exactly the same. That said, because some internal codes have changed a lot, there might be unexpected edge cases that break current user code. If that happens, you can revert to the previous version by `pip install xesmf==0.1.2` and follow `old docs <https://xesmf.readthedocs.io/en/v0.1.2/>`_.

New features
~~~~~~~~~~~~
* Lazy evaluation on dask arrays (uses :py:func:`xarray.apply_ufunc` and :py:func:`dask.array.map_blocks`)
* Automatic looping over variables in an xarray Dataset
* Add tutorial notebooks on those new features

By `Jiawei Zhuang <https://github.com/JiaweiZhuang>`_


0.1.2 (03-08-2019)
------------------
This release mostly contains internal clean-ups to facilitate future development.

New features
~~~~~~~~~~~~
* Deprecates `regridder.A` in favor of `regridder.weights`
* Speed-up test suites by using coarser grids
* Use parameterized tests when appropriate
* Fix small memory leaks from `ESMF.Grid`
* Properly assert ESMF enums

By `Jiawei Zhuang <https://github.com/JiaweiZhuang>`_


0.1.1 (31-12-2017)
------------------
Initial release.
By `Jiawei Zhuang <https://github.com/JiaweiZhuang>`_
