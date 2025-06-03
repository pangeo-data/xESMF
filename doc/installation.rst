.. _installation-label:

Installation
============

Try on Binder without local installation
----------------------------------------

The `Binder project <https://mybinder.readthedocs.io>`_ provides pre-configured environment in the cloud. You just need a web browser to access it. Please follow the Binder link on `xESMF's GitHub page <https://github.com/pangeo-data/xESMF>`_.

Install on local machine with Conda
-----------------------------------

xESMF requires Python>=3.8. The major dependencies are xarray and ESMPy, and the best way to install them is using Conda_.
Note that the latest xarray releases require Python 3.9 or later.

First, `install miniconda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_. Then, we recommend creating a new, clean environment:

.. code-block:: bash

    $ conda create -n xesmf_env
    $ conda activate xesmf_env

Getting xESMF is as simple as:

.. code-block:: bash

    $ conda install -c conda-forge xesmf

We also highly recommend those extra packages for full functionality:

.. code-block:: bash

    # to support all features in xESMF
    $ conda install -c conda-forge dask netCDF4

    # optional dependencies for executing all notebook examples
    $ conda install -c conda-forge matplotlib cartopy jupyterlab


Alternatively, you can first install dependencies, and then use ``pip`` to install xESMF:

.. code-block:: bash

    $ conda install -c conda-forge esmpy xarray numpy shapely cf_xarray sparse numba
    $ pip install git+https://github.com/pangeo-data/xesmf.git

This will install the latest version from the github repo. To install a specific release, append the version tag to the url (e.g. `@v0.5.0`).

Notes about ESMpy
-----------------

* ESMpy 8.4 is only compatible with xESMF >= 0.7.
* ESMpy must be installed through Conda or compiled manually; it is not available through PyPI.  When installing xESMF with pip, the ESMpy package must be manually installed first.

Testing your installation
-------------------------

xESMF itself is a lightweight package, but its dependency ESMPy is a quite heavy and sometimes might be installed incorrectly. To validate & debug your installation, you can use pytest to run the test suites:

.. code-block:: bash

    $ conda install pytest
    $ pytest -v --pyargs xesmf  # should all pass

A common cause of error (especially for HPC cluster users) is that pre-installed modules like NetCDF, MPI, and ESMF are incompatible with the conda-installed equivalents. Make sure you have a clean environment when running ``conda install`` (do not ``module load`` other libraries). See `this issue <https://github.com/JiaweiZhuang/xESMF/issues/55#issuecomment-514298498>`_ for more discussions.

Notes for Windows users
-----------------------

The ESMPy conda package is usually only available for Linux and Mac OSX.
Builds for windows have been made for some versions (8.4.2).
Windows users can try the
`Linux subsystem <https://docs.microsoft.com/en-us/windows/wsl/about>`_
or `docker-miniconda <https://hub.docker.com/r/continuumio/miniconda3/>`_ .

Installing scientific software on Windows can often be a pain, and
`Docker <https://www.docker.com>`_ is a pretty good workaround.
It takes some time to learn but worths the effort.
Check out this `tutorial on using Docker with Anaconda <https://towardsdatascience.com/how-docker-can-help-you-become-a-more-effective-data-scientist-7fc048ef91d5>`_.

This problem is being investigated. See `this other issue <https://github.com/conda-forge/esmf-feedstock/pull/1198>`_.

Install development version from GitHub repo
--------------------------------------------

To get the latest version that is not uploaded to PyPI_ yet::

    $ pip install --upgrade git+https://github.com/pangeo-data/xESMF.git

Developers can track source code change::

    $ git clone https://github.com/pangeo-data/xESMF.git
    $ cd xESMF
    $ pip install -e .

.. _xarray: http://xarray.pydata.org
.. _ESMPy: http://earthsystemmodeling.org/esmpy/
.. _Conda: https://docs.conda.io/
.. _PyPI: https://pypi.python.org/pypi
.. _NESII: https://www.esrl.noaa.gov/gsd/nesii/
