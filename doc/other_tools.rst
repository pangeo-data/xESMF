.. _other_tools-label:

Other geospatial regridding tools
=================================

Here is a brief overview of other regridding tools that the authors are aware of
(for geospatial data on the sphere, excluding traditional image resizing functions).
They are all great tools and have helped the author a lot in both scientific research
and xESMF development. Check them out if xESMF cannot suit your needs.

- `ESMF <https://earthsystemmodeling.org/docs/release/latest/ESMF_refdoc/>`_ (*Fortran package*)

Although its name "Earth System Modeling Framework" doesn't indicate a regridding
functionality, it actually contains a very powerful regridding engine.
It is widely used in Earth System Models (ESMs), serving as both the software infrastructure
and the regridder for transforming data between the atmosphere, ocean, and land components.
It can deal with general irregular meshes, in either 2D or 3D.

ESMF is a huge beast, containing one million lines of source code.
Even just compiling it requires some effort.
It is more for building ESMs than for data analysis.

- `ESMPy <http://earthsystemmodeling.org/esmpy/>`_ (*Python interface to ESMF*)

ESMPy provides a much simpler way to use ESMF's regridding functionality.
The greatest thing is, it is pre-compiled as a
`conda package <https://anaconda.org/NESII/esmpy>`_,
so you can install it with one-click and don't have to go through
the daunting compiling process on your own.

However, ESMPy is a complicated Python API that controls a huge Fortran beast
hidden underneath. It is not as intuitive as native Python packages, and even
a simple regridding task requires more than 10 lines of arcane code. The
purpose of xESMF is to provide a friendlier interface to the xarray community.
Check out this nice `tutorial <https://github.com/nawendt/esmpy-tutorial>`_
before going to the
`official doc <http://www.earthsystemmodeling.org/esmf_releases/last_built/esmpy_doc/html/index.html>`_.

- `TempestRemap <https://github.com/ClimateGlobalChange/tempestremap>`_
  (*C++ package*)

A pretty modern and powerful package,
supporting arbitrary-order conservative remapping.
It can also generate cubed-sphere grids on the fly
and can be modified to support many cubed-sphere grid variations
(`example <https://github.com/JiaweiZhuang/Tempest_for_GCHP>`_, only if you can read C++).

- `SCRIP <http://oceans11.lanl.gov/trac/SCRIP>`_ (*Fortran package*)

An old package, once popular but **no longer maintained** (long live SCRIP).
You should not use it now, but should know that it exists.
Newer regridding packages often follow its standards --
you will see "SCRIP format" here and there, for example in ESMF or TempestRemap.

- `Regridder in NCL <https://www.ncl.ucar.edu/Applications/regrid.shtml>`_
  (*NCAR Command Language*)

Has bilinear and conservative algorithms for rectilinear grids,
and also supports some specialized curvilinear grids.
There is also an `ESMF wrapper <https://www.ncl.ucar.edu/Applications/ESMF.shtml>`_
that works for more grid types.

- `Regridder in NCO <http://nco.sourceforge.net/nco.html#Regridding>`_
  (*command line tool*)

- `Regridder in Iris <https://scitools-iris.readthedocs.io/en/v3.4.1/userguide/interpolation_and_regridding.html>`_
  (*Python package*)

- `Regridder in xCDAT <https://xcdat.readthedocs.io/en/latest/generated/xcdat.regridder.accessor.RegridderAccessor.html>`_
  (*Python package*)

 Offers regridding algorithms from xESMF and `regrid2`, originally from the CDAT `cdutil` package, it also includes code for vertical regridding.
