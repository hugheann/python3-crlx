from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import requests
from requests.compat import urljoin
import warnings
import xarray as xr

from crlx.maps import VARMAP, SYSMAP, LOCMAP


REQ_DT_FMT = '%Y-%m-%d %H:%M:%S'
ENCODING = {'time': {'units': 'nanoseconds since 1900-01-01'}}  # xr.Dataset to netcdf encoding for time


def match_variable(ds, da_to_merge, method='nearest', max_gap=60 * 2):
    ogt = ds[['time']]
    combo = xr.combine_by_coords([ogt, da_to_merge])
    interp = combo.interpolate_na(dim='time', method=method, max_gap=timedelta(seconds=max_gap))
    matching = interp.sel(time=ds.time)
    new_ds = xr.combine_by_coords([ds, matching])
    new_ds = new_ds.sortby('time')
    return new_ds


class CRLX():
    def __init__(self, base_url: str, verify: bool = True, verbose: bool = False):
        self.base_url = base_url
        self._verify = verify
        self._verbose = verbose

    def get_sensor_info(self, sensor_id, enabled: bool = True):
        params = {'sensor_id': sensor_id,
                  'enabled': enabled,
                  'format': 'json'}
        url = urljoin(self.base_url, 'sensor')
        response = requests.get(url, params = params, verify = self._verify)
        if response.status_code == requests.codes.ok:
            request_data = response.json()
            df = pd.DataFrame(request_data)
            ds = df.to_xarray()

            ds = ds[['sensor_id','sensor_name', 'sensor_class','sensor_prefix',
                     'serial_number','description', 'location_text','vendor','model']]
            return ds


    def get_associated_parameters(self, sensor_id):
        params = {'sensor_id': sensor_id,
                  'format': 'json'}
        url = urljoin(self.base_url, 'parameter')
        response = requests.get(url, params = params, verify = self._verify)
        if response.status_code == requests.codes.ok:
            request_data = response.json()
            df = pd.DataFrame(request_data)
            ds = df.to_xarray()
            ds = ds.rename({'gross_min': 'sensor_min', 'gross_max': 'sensor_max',
                            'recommended_min': 'operator_min','recommended_max':'operator_max'})
            ds = ds[['parameter_id','short_name','long_name','description','sensor_min','sensor_max',
                     'operator_min','operator_max','data_model','data_table','archive_data_table','data_fieldname',
                     'flag_fieldname' ,'units_abbrev','sensor_id', 'standard_name']]
            return ds


    def get_data(self,table_or_model: str,
                 bdt: datetime = datetime(2024,7,26), edt: datetime = datetime(2024,12,31,23,59,59),
                 decimator: int = 1) -> xr.Dataset:
        bdt = bdt.replace(tzinfo = None)
        edt = edt.replace(tzinfo = None)
        params = {'date_0': bdt.strftime(REQ_DT_FMT),
                  'date_1': edt.strftime(REQ_DT_FMT),
                  'date_after': bdt.strftime(REQ_DT_FMT),
                  'date_before': edt.strftime(REQ_DT_FMT),
                  'decfactr': decimator,
                  'format': 'json'}

        url = urljoin(self.base_url, table_or_model)
        response = requests.get(url, params = params, verify = self._verify)
        if response.status_code != requests.codes.ok:
            url = urljoin(self.base_url,'decimateData')
            params['model'] = table_or_model
            response = requests.get(url, params = params, verify = self._verify)
            if response.status_code == requests.codes.ok:
                request_data = response.json()
            else:
                raise ConnectionError(response.reason, response.url)
        else:
            request_data = response.json()

        if len(request_data) == 0:
            if self._verbose is True:
                warnings.warn('No data available for request.')
            return None

        df = pd.DataFrame(request_data)
        df['time'] = pd.to_datetime(df['datetime_corrected'], format='mixed').dt.tz_localize(None)
        df.index = df.time
        ds = df.to_xarray()
        ds['sensor_id'] = ds['sensor_id'].astype(str)

        datasets = []
        sensor_ids = np.unique(ds.sensor_id)
        for sensor_id in sensor_ids:
            _ds = ds.where(ds.sensor_id == sensor_id)
            if sensor_id == 'pcotwo002202':  # Brief hack for the SKQ.
                sensor_id = 'pcotwo002101'

            parameter_metadata = self.get_associated_parameters(sensor_id)

            # Parse out relevant parameters from the table. NOTE: Some sensors have data split between tables, which this does not account for.
            if '_' in table_or_model:
                parameter_metadata = parameter_metadata.where(parameter_metadata.data_table == table_or_model,drop = True)
            else:
                parameter_metadata = parameter_metadata.where(parameter_metadata.data_model == table_or_model,drop = True)
            _ds = _ds[parameter_metadata.data_fieldname.values.tolist() + ['sensor_id']] #Only keep relevant fields from table.


            # Map variables to the short name in CORIOLIX.
            short_name_map = dict(zip(parameter_metadata.data_fieldname.values.tolist(),
                                      parameter_metadata.short_name.values.tolist()))
            _ds = _ds.rename(short_name_map)


            # Add parameter attributes to each variable in dataset.
            for var in _ds.data_vars:
                if var == 'sensor_id':
                    continue
                pmd = parameter_metadata.where(parameter_metadata.short_name == var, drop = True)
                for pmdv in pmd.data_vars:
                    _ds[var].attrs[pmdv] = str(pmd[pmdv].values[0])

            # Map variables again to custom names found in maps.py.
            #vmap = {key: VARMAP[key] for key in list(_ds.data_vars)}

            try:
                vmap = {key: VARMAP[key] for key in list(_ds.data_vars)}
                _ds = _ds.rename(vmap)
            except:
                pass

            # Add sensor attributes to root dataset.
            try:
                sensor_metadata = self.get_sensor_info(sensor_id)
            except:
                sensor_metadata = self.get_sensor_info(sensor_id, enabled = False)

            for smv in sensor_metadata.data_vars:
                _ds.attrs[smv] = str(sensor_metadata[smv].values[0])

            system = SYSMAP[str(sensor_metadata.sensor_class.values[0])]
            location = LOCMAP[str(sensor_metadata.location_text.values[0])]
            _ds.attrs['system'] = system
            _ds.attrs['location'] = location
            datasets.append(_ds)
        try:
            ds = xr.combine_by_coords(datasets, combine_attrs='drop_conflicts')
        except:
            ds = xr.concat(datasets, dim = 'time', combine_attrs = 'drop_conflicts')

        ds = ds.drop_duplicates(dim = 'time')
        ds = ds.sortby('time')

        ds['time'].attrs['description'] = 'Datetime in UTC.'
        ds = ds[sorted(ds.data_vars)]
        return ds


