from datetime import datetime, timezone, timedelta
import os
import pandas as pd
import xarray as xr
from crlx import SIKULIAQ


SAVE_DIR_BASE = 'C:/Users/Ian/Box/pco2_skq/data/daily/'
ENCODING = {'time': {'units': 'nanoseconds since 1900-01-01'}}

VESSEL = 'SIKULIAQ'
SENSOR = 'APOLLO-ASP3'

BDT = datetime(2024,7,26).replace(tzinfo = None)
EDT = datetime.now(timezone.utc).replace(tzinfo = None)
SKQ = SIKULIAQ(verbose = True)


def main():
    dtr = pd.date_range(BDT,EDT, freq = '1d')

    for dt in dtr:
        bdt = dt.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
        edt = dt + timedelta(hours  = 23, minutes = 59, seconds =59, microseconds = 999999)

        ds = SKQ.get_wet_lab_apollo(bdt, edt)

        if ds is not None:
            save_dir = os.path.join(SAVE_DIR_BASE, f'{VESSEL}_{str(ds.system)}_{str(ds.location)}_{SENSOR}/')
            os.makedirs(save_dir,exist_ok=True)
            fdt = bdt.strftime('%Y%m%d')
            fname = f'{VESSEL}_{str(ds.system)}_{str(ds.location)}_{SENSOR}_{fdt}.nc'
            fpath = os.path.join(save_dir, fname)
            ds.attrs['vessel'] = 'R/V SIKULIAQ'
            ds.to_netcdf(fpath, encoding = ENCODING)


if __name__ == "__main__":
    main()