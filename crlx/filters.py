import numpy as np
import xarray as xr



def med_filt(da: xr.DataArray, window_size: int = 11) -> xr.DataArray:
  mfda = da.rolling({'time': window_size}, center = True, min_periods = 1).median(skipna = True)
  return mfda

def stdev_filt(da: xr.DataArray, multiplier: float = 3) -> xr.DataArray:
  sfda = da.where((da > da.mean() - multiplier * da.std()) & (da < da.mean() + multiplier * da.std()), np.nan)
  return sfda