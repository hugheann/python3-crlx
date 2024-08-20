import numpy as np
import xarray as xr

from crlx.flagging.qartod import FLAG


def suspect_surface_mhw_test(data: xr.DataArray,
                             latitude: xr.DataArray,
                             longitude: xr.DataArray,
                             lookup: xr.Dataset) -> xr.DataArray:

    cells = lookup.sel(dayofyear=data.time.dt.dayofyear, latitude = latitude, longitude = longitude, method = 'nearest')
    percentile_var = 'sst_90'
    results = xr.full_like(data, fill_value=FLAG.NOT_EVALUATED).astype('int8')  # Assign NOT_EVALUATED by default.
    results = results.where(cells[percentile_var] >= data, FLAG.HIGH_INTEREST)
    results = results.where(cells[percentile_var] < data, FLAG.PASS)
    results = results.where((~np.isnan(data)), FLAG.MISSING_DATA)
    results = results.where(~np.isnan(cells[percentile_var]), FLAG.NOT_EVALUATED)
    results = results.drop_vars(['latitude', 'longitude', 'dayofyear', 'month'], errors='ignore')


    results.attrs['test_name'] = f'Suspect Surface Marine Heatwave Test - {data.name}'
    results.attrs['is_qartod'] = "False"
    results.attrs['is_experimental'] = "True"
    results.attrs[
        'description'] = """The Suspect Surface Marine Heatwave Test flags values that are above the 
        climatological 90th percentile. Depending on the frequency of the data collected, singular values can be safely
        ignored. Subsequent and repeated values over a 24hr period may be of interest in marine heatwave analysis."""
    results.attrs['valid_flags'] = "1 = PASS, 2 = NOT_EVALUATED, 3 = HIGH_INTEREST or SUSPECT, 9 = MISSING_DATA"
    results.attrs['reference_dataset'] = lookup.attrs['dataset']
    results.attrs['temporal_reference_resolution'] = lookup.attrs['temporal_resolution']
    results.attrs['spatial_reference_resolution'] = lookup.attrs['spatial_resolution']
    results.attrs['spatial_reference_units'] = lookup.attrs['spatial_resolution_units']
    results.attrs['reference_dataset'] = lookup.attrs['dataset']
    return results




def suspect_surface_mcs_test(data: xr.DataArray,
                             latitude: xr.DataArray,
                             longitude: xr.DataArray,
                             lookup: xr.Dataset) -> xr.DataArray:

    cells = lookup.sel(dayofyear=data.time.dt.dayofyear, latitude = latitude, longitude = longitude, method = 'nearest')
    percentile_var = 'sst_10'
    results = xr.full_like(data, fill_value=FLAG.NOT_EVALUATED).astype('int8')  # Assign NOT_EVALUATED by default.
    results = results.where(cells[percentile_var] <= data, FLAG.HIGH_INTEREST)
    results = results.where(cells[percentile_var] > data, FLAG.PASS)
    results = results.where((~np.isnan(data)), FLAG.MISSING_DATA)
    results = results.where(~np.isnan(cells[percentile_var]), FLAG.NOT_EVALUATED)
    results = results.drop_vars(['latitude', 'longitude', 'dayofyear', 'month'], errors='ignore')


    results.attrs['test_name'] = f'Suspect Surface Marine Cold Spell Test - {data.name}'
    results.attrs['is_qartod'] = "False"
    results.attrs['is_experimental'] = "True"
    results.attrs[
        'description'] = """The Suspect Surface Marine Cold Spell Test flags values that are below the 
        climatological 10th percentile. Depending on the frequency of the data collected, singular values can be safely
        ignored. Subsequent and repeated values over a 24hr period may be of interest in marine cold spell analysis."""
    results.attrs['valid_flags'] = "1 = PASS, 2 = NOT_EVALUATED, 3 = HIGH_INTEREST or SUSPECT, 9 = MISSING_DATA"
    results.attrs['reference_dataset'] = lookup.attrs['dataset']
    results.attrs['temporal_reference_resolution'] = lookup.attrs['temporal_resolution']
    results.attrs['spatial_reference_resolution'] = lookup.attrs['spatial_resolution']
    results.attrs['spatial_reference_units'] = lookup.attrs['spatial_resolution_units']
    results.attrs['reference_dataset'] = lookup.attrs['dataset']
    return results




