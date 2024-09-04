from datetime import datetime, timezone, timedelta
import os
import pandas as pd

from crlx.core import ENCODING
from crlx import SIKULIAQ

#SAVE_DIR_BASE = 'C:/Users/Ian/Box/pco2_skq/data/daily/'
SAVE_DIR_BASE = '/media/support/data/sikuliaq/daily/'

VESSEL = 'SIKULIAQ'
SENSOR = 'MET4A-1'
LOC = 'FWD-MAST'
SYS = 'ATMOSPHERIC'

BDT = datetime(2024,7,26).replace(tzinfo = None)
EDT = datetime.now(timezone.utc).replace(tzinfo = None)
SKQ = SIKULIAQ()

def main():
    dtr = pd.date_range(BDT,EDT, freq = '1d')

    for dt in dtr:
        bdt = dt.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
        edt = dt + timedelta(hours  = 23, minutes = 59, seconds =59, microseconds = 999999)

        save_dir = os.path.join(SAVE_DIR_BASE, f'{VESSEL}_{SYS}_{LOC}_{SENSOR}/')
        fdt = bdt.strftime('%Y%m%d')
        fname = f'{VESSEL}_{SYS}_{LOC}_{SENSOR}_{fdt}.nc'
        fpath = os.path.join(save_dir, fname)

        delta_time = (datetime.now(timezone.utc).replace(tzinfo=None) - bdt).total_seconds()
        if not os.path.isfile(fpath) or delta_time <= 86400 * 3 :

            ds = SKQ.get_met_station(bdt, edt)
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