import os

import numpy as np
import xarray as xr

import xesmf as xe


def main():

    # Open tripole SST dataset
    ds_in = xr.open_dataset('xesmf/tripole_SST.nc', engine='netcdf4')

    # Open input grid specification
    ds_ingrid = xr.open_dataset('xesmf/mom6_grid_spec.nc')
    ds_sst_grid = ds_ingrid.rename({'geolat': 'lat', 'geolon': 'lon'})
    ds_sst_grid['mask'] = ds_ingrid['wet']

    # Get MOM6 mask
    ds_ingrid['mask'] = ds_ingrid['wet']

    # Open output grid specification
    ds_outgrid = xr.open_dataset('xesmf/C384_gaussian_grid.nc')

    # Get C384 land-sea mask
    ds_outgrid['mask'] = 1 - ds_outgrid['land'].where(ds_outgrid['land'] < 2.0).squeeze()

    # Create regridder
    baseregrid = xe.Regridder(ds_sst_grid, ds_outgrid, 'bilinear', periodic=True)
    base = baseregrid(ds_in['SST']).rename('SST')
    base.to_netcdf('output_from_base_regrid_method.nc')

    # Add pole_kind to grid. 1 denoted monopole, 2 bipole
    ds_sst_grid['pole_kind'] = np.array([1, 2], np.int32)
    ds_outgrid['pole_kind'] = np.array([1, 1], np.int32)

    newregrid = xe.Regridder(ds_sst_grid, ds_outgrid, 'bilinear', periodic=True)
    new = newregrid(ds_in['SST']).rename('SST')
    new.to_netcdf('output_from_new_regrid_method.nc')

    # Add incorrect grid information
    ds_sst_grid['pole_kind'] = np.array([1, 1], np.int32)
    ds_outgrid['pole_kind'] = np.array([1, 1], np.int32)

    errregrid = xe.Regridder(ds_sst_grid, ds_outgrid, 'bilinear', periodic=True)
    err = errregrid(ds_in['SST']).rename('SST')
    err.to_netcdf('output_from_err_regrid_method.nc')


if __name__ == '__main__':
    main()
