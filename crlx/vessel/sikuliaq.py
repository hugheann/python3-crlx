from datetime import datetime,timedelta
import numpy as np
import warnings
import xarray as xr

from crlx.core import CRLX
from crlx.converters import mbar2atm
from crlx.dpa.co2 import CO2

class SIKULIAQ(CRLX):

    API_URL: str = "https://coriolix.sikuliaq.alaska.edu/api/"

    def __init__(self, verify: bool = True, verbose: bool = False):
        super().__init__(self.API_URL, verify, verbose)

    def get_main_lab_sbe45a(self, bdt: datetime, edt: datetime):
        ds = self.get_data('sensor_float_2', bdt, edt)
        return ds

    def get_main_lab_sbe45b(self, bdt: datetime, edt: datetime):
        ds = self.get_data('sensor_float_3', bdt, edt)
        return ds


    def get_wet_lab_ldeo(self,bdt, edt, max_interp_gap: int = 60):
        ds = self.get_data('SensorFloat17', bdt=bdt, edt=edt)
        if ds is not None:
            ds = ds.drop_vars(['t_eq_rtd_bad', 'pco2'], errors='ignore')

            ovs = [dv for dv in ds.data_vars if ds[dv].dtype == 'O']
            for ov in ovs:
                if ov in ['sensor_id']:
                    continue
                ds[ov] = ds[ov].where((ds[ov] is None), np.nan).astype(float)

            air = ds.where(ds.source_id == 1, drop=True)
            sw = ds.where(ds.source_id == 0, drop=True)
            nit = ds.where(ds.source_id == 2, drop=True)
            sd1 = ds.where(ds.source_id == 4, drop=True)
            sd2 = ds.where(ds.source_id == 6, drop=True)
            sd3 = ds.where(ds.source_id == 8, drop=True)
            sd4 = ds.where(ds.source_id == 10, drop=True)
            sd5 = ds.where(ds.source_id == 12, drop=True)

            try:
                met = self.get_met_station(bdt - timedelta(seconds=3600 * 2), edt + timedelta(seconds=3600 * 2))[
                    ['barometric_pressure', 'air_temperature']]
                airt = met.air_temperature
                barop = met.barometric_pressure * 1000
            except:
                airt = xr.full_like(ds.xco2_corr, np.nan)
                airt = airt.rename('air_temperature')
                barop = xr.full_like(ds.xco2_corr, np.nan)
                barop = barop.rename('barometric_pressure')

            try:
                sst = self.get_centerboard_sbe38(bdt - timedelta(seconds=3600 * 2), edt + timedelta(seconds=3600 * 2)).rename(
                    {'sea_water_temperature': 'sea_surface_temperature'})['sea_surface_temperature']
            except:
                sst = xr.full_like(ds.xco2_corr,np.nan)
                sst = sst.rename('sea_surface_temperature')

            try:
                pracsal = self.get_main_lab_sbe45a(bdt - timedelta(seconds=3600 * 2), edt + timedelta(seconds=3600 * 2))[
                    'sea_water_practical_salinity']
            except:
                pracsal = xr.full_like(ds.xco2_corr,np.nan)
                pracsal = pracsal.rename('sea_water_practical_salinity')

            try:
                gps = self.get_gps_gga_bow(bdt - timedelta(seconds=3600 * 2), edt + timedelta(seconds=3600 * 2))[['latitude','longitude']]
                lat = gps.latitude
                lon = gps.longitude
            except:
                lat = xr.full_like(ds.xco2_corr,np.nan)
                lat = lat.rename('latitude')
                lon = xr.full_like(ds.xco2_corr,np.nan)
                lon = lon.rename('longitude')


            dpaCO2 = CO2()

            # Recompute Air
            air_diag = air[['cell_temperature', 'gas_flow_rate']]
            air_xco2 = air.xco2_corr
            air_combo = xr.combine_by_coords([air_xco2, sst, barop, pracsal, airt,lat, lon])
            air_combo = air_combo.interpolate_na(dim='time', method='nearest', max_gap=timedelta(seconds=max_interp_gap))
            air_combo = air_combo.sel(time=air_xco2.time)
            air_combo['xco2_corr'] = air_xco2
            air_combo['water_vapor_pressure'] = dpaCO2.calc_water_vapor_pressure(air_combo.sea_surface_temperature,
                                                                                 air_combo.sea_water_practical_salinity)
            air_combo['pco2'] = dpaCO2.calc_pco2(air_combo.xco2_corr, mbar2atm(air_combo.barometric_pressure),
                                                 air_combo.water_vapor_pressure)
            air_combo['fco2'] = dpaCO2.calc_fugacity(air_combo.pco2, air_combo.xco2_corr, air_combo.barometric_pressure,
                                                     air_combo.air_temperature)
            air_combo = xr.combine_by_coords([air_combo, air_diag], combine_attrs='drop_conflicts')
            air_combo = air_combo.expand_dims(sample_source=['air'], axis=1)

            # Recompute Seawater
            sw_diag = sw[['cell_temperature', 'gas_flow_rate']]
            sw_xco2 = sw.xco2_corr
            teq = sw.equilibrator_temperature_sbe38
            peq = sw.equilibrator_pressure
            sw_combo = xr.combine_by_coords([sw_xco2, sst, barop, pracsal, teq, peq,lat,lon])
            sw_combo = sw_combo.interpolate_na(dim='time', method='nearest', max_gap=timedelta(seconds=max_interp_gap))
            sw_combo = sw_combo.sel(time=sw_xco2.time)
            sw_combo['xco2_corr'] = sw_xco2
            sw_combo['water_vapor_pressure'] = dpaCO2.calc_water_vapor_pressure(sw_combo.equilibrator_temperature_sbe38,
                                                                                sw_combo.sea_water_practical_salinity)
            sw_combo['equilibrator_pco2'] = dpaCO2.calc_pco2(sw_combo.xco2_corr, mbar2atm(sw_combo.equilibrator_pressure),
                                                             sw_combo.water_vapor_pressure)
            sw_combo['pco2'] = dpaCO2.calc_sea_water_pco2_t_corr(sw_combo.equilibrator_pco2,
                                                                 sw_combo.sea_surface_temperature,
                                                                 mbar2atm(sw_combo.barometric_pressure))
            sw_combo['fco2'] = dpaCO2.calc_fugacity(sw_combo.pco2, sw_combo.xco2_corr, sw_combo.barometric_pressure,
                                                    sw_combo.sea_surface_temperature)
            sw_combo = xr.combine_by_coords([sw_combo, sw_diag], combine_attrs='drop_conflicts')
            sw_combo = sw_combo.expand_dims(sample_source=['seawater'], axis=1)

            sdvars = ['cell_temperature', 'gas_flow_rate', 'xco2_corr']

            nit = nit[sdvars]
            nit = nit.expand_dims(sample_source=['nitrogen'], axis=1)

            sd1 = sd1[sdvars]
            sd1 = sd1.expand_dims(sample_source=['sd1'], axis=1)

            sd2 = sd2[sdvars]
            sd2 = sd2.expand_dims(sample_source=['sd2'], axis=1)

            sd3 = sd3[sdvars]
            sd3 = sd3.expand_dims(sample_source=['sd3'], axis=1)

            sd4 = sd4[sdvars]
            sd4 = sd4.expand_dims(sample_source=['sd4'], axis=1)

            sd5 = sd5[sdvars]
            sd5 = sd5.expand_dims(sample_source=['sd5'], axis=1)

            try:
                combo = xr.combine_by_coords([air_combo, sw_combo, nit, sd1, sd2, sd3, sd4, sd5],
                                             combine_attrs='drop_conflicts')
            except:
                combo = xr.concat([air_combo, sw_combo, nit, sd1, sd2, sd3, sd4, sd5], dim='time',
                                  combine_attrs='drop_conflicts')
                combo = combo.drop_duplicates(dim='time')
            combo = combo[sorted(combo.data_vars)]
            combo = combo.sortby('time')
            return combo
        else:
            if self._verbose is True:
                warnings.warn(f'No Data {bdt}- {edt}')
            return None


    def get_wet_lab_apollo(self,bdt, edt, max_interp_gap: int = 60):
        ds = self.get_data('sensor_mixlg_11', bdt=bdt, edt=edt)
        if ds is not None:
            ds['equilibrator_temperature_sbe45'] = ds.sea_water_temperature
            ds = ds.rename({'sea_water_conductivity': 'apollo_sbe45_sea_water_electrical_conductivity',
                            'sea_water_practical_salinity': 'apollo_sbe45_sea_water_practical_salinity',
                            'sea_water_temperature': 'apollo_sbe45_sea_water_temperature'})

            ds = ds.drop_vars(['gps_latitude',
                               'gps_longitude',
                               'operator',
                               'gps_datetime'], errors='ignore')

            ovs = [dv for dv in ds.data_vars if ds[dv].dtype == 'O']
            for ov in ovs:
                if ov in ['sample_source', 'instrument_mode', 'instrument_timestamp', 'overflow_alarm',
                          'shutoff_valve_alarm', 'test_type', 'sensor_id']:
                    continue
                ds[ov] = ds[ov].where((ds[ov] is None), np.nan).astype(float)

            air = ds.where(ds.sample_source.str.contains('Air'), drop=True)
            sw = ds.where(ds.sample_source.str.contains('Seawater'), drop=True)
            sd1 = ds.where(ds.sample_source.str.contains('SD-1'), drop=True)
            sd2 = ds.where(ds.sample_source.str.contains('SD-2'), drop=True)
            sd3 = ds.where(ds.sample_source.str.contains('SD-3'), drop=True)
            sd4 = ds.where(ds.sample_source.str.contains('SD-4'), drop=True)
            sd5 = ds.where(ds.sample_source.str.contains('SD-5'), drop=True)

            try:
                met = self.get_met_station(bdt - timedelta(seconds=3600 * 2), edt + timedelta(seconds=3600 * 2))[
                    ['barometric_pressure', 'air_temperature']]
                airt = met.air_temperature
                barop = met.barometric_pressure * 1000
            except:
                airt = xr.full_like(ds.xco2_corr, np.nan)
                airt = airt.rename('air_temperature')
                barop = xr.full_like(ds.xco2_corr, np.nan)
                barop = barop.rename('barometric_pressure')

            try:
                sst = self.get_centerboard_sbe38(bdt - timedelta(seconds=3600 * 2), edt + timedelta(seconds=3600 * 2)).rename(
                    {'sea_water_temperature': 'sea_surface_temperature'})['sea_surface_temperature']
            except:
                sst = xr.full_like(ds.xco2_corr,np.nan)
                sst = sst.rename('sea_surface_temperature')

            try:
                pracsal = self.get_main_lab_sbe45a(bdt - timedelta(seconds=3600 * 2), edt + timedelta(seconds=3600 * 2))[
                    'sea_water_practical_salinity']
            except:
                pracsal = xr.full_like(ds.xco2_corr,np.nan)
                pracsal = pracsal.rename('sea_water_practical_salinity')

            try:
                gps = self.get_gps_gga_bow(bdt - timedelta(seconds=3600 * 2), edt + timedelta(seconds=3600 * 2))[['latitude','longitude']]
                lat = gps.latitude
                lon = gps.longitude
            except:
                lat = xr.full_like(ds.xco2_corr,np.nan)
                lat = lat.rename('latitude')
                lon = xr.full_like(ds.xco2_corr,np.nan)
                lon = lon.rename('longitude')



            dpaCO2 = CO2()

            # Recompute Air
            air_diag = air[['cooling_pad_temperature', 'gas_flow_rate', 'instrument_mode', 'instrument_timestamp',
                            'li7815_cavity_pressure', 'li7815_cavity_temperature', 'li7815_diagnostic', 'li7815_h2o',
                            'li7815_xco2', 'li7815_xco2_stdev', 'measurement_flow_rate', 'measurement_interval',
                            'measurement_stdev', 'number_of_measurements', 'overflow_alarm', 'purge_flow_rate',
                            'purge_interval', 'shutoff_valve_alarm', 'test_type']]
            air_xco2 = air.xco2_corr
            air_combo = xr.combine_by_coords([air_xco2, sst, barop, pracsal, airt, lat, lon])
            air_combo = air_combo.interpolate_na(dim='time', method='nearest', max_gap=timedelta(seconds=max_interp_gap))
            air_combo = air_combo.sel(time=air_xco2.time)
            air_combo['xco2_corr'] = air_xco2

            air_combo['water_vapor_pressure'] = dpaCO2.calc_water_vapor_pressure(air_combo.sea_surface_temperature,
                                                                                 air_combo.sea_water_practical_salinity)
            air_combo['pco2'] = dpaCO2.calc_pco2(air_combo.xco2_corr, mbar2atm(air_combo.barometric_pressure),
                                                 air_combo.water_vapor_pressure)
            air_combo['fco2'] = dpaCO2.calc_fugacity(air_combo.pco2, air_combo.xco2_corr, air_combo.barometric_pressure,
                                                     air_combo.air_temperature)
            air_combo = xr.combine_by_coords([air_combo, air_diag], combine_attrs='drop_conflicts')
            air_combo = air_combo.expand_dims(sample_source=['air'], axis=1)

            # Recompute Seawater
            sw_diag = sw[['cooling_pad_temperature', 'equilibrator_temperature', 'gas_flow_rate', 'instrument_mode',
                          'instrument_timestamp', 'li7815_cavity_pressure', 'li7815_cavity_temperature',
                          'li7815_diagnostic', 'li7815_h2o', 'li7815_xco2', 'li7815_xco2_stdev', 'measurement_flow_rate',
                          'measurement_interval', 'measurement_stdev', 'number_of_measurements', 'overflow_alarm',
                          'purge_flow_rate', 'purge_interval', 'apollo_sbe45_sea_water_electrical_conductivity',
                          'apollo_sbe45_sea_water_practical_salinity', 'apollo_sbe45_sea_water_temperature',
                          'shutoff_valve_alarm', 'test_type', 'water_flow_rate']]
            sw_xco2 = sw.xco2_corr
            teq = sw.equilibrator_temperature_sbe45
            peq = sw.equilibrator_pressure
            sw_combo = xr.combine_by_coords([sw_xco2, sst, barop, pracsal, teq, peq,lat,lon])
            sw_combo = sw_combo.interpolate_na(dim='time', method='nearest', max_gap=timedelta(seconds=max_interp_gap))
            sw_combo = sw_combo.sel(time=sw_xco2.time)
            sw_combo['xco2_corr'] = sw_xco2
            sw_combo['water_vapor_pressure'] = dpaCO2.calc_water_vapor_pressure(sw_combo.equilibrator_temperature_sbe45,
                                                                                sw_combo.sea_water_practical_salinity)
            sw_combo['equilibrator_pco2'] = dpaCO2.calc_pco2(sw_combo.xco2_corr, mbar2atm(sw_combo.equilibrator_pressure),
                                                             sw_combo.water_vapor_pressure)
            sw_combo['pco2'] = dpaCO2.calc_sea_water_pco2_t_corr(sw_combo.equilibrator_pco2,
                                                                 sw_combo.sea_surface_temperature,
                                                                 mbar2atm(sw_combo.barometric_pressure))
            sw_combo['fco2'] = dpaCO2.calc_fugacity(sw_combo.pco2, sw_combo.xco2_corr, sw_combo.barometric_pressure,
                                                    sw_combo.sea_surface_temperature)
            sw_combo = xr.combine_by_coords([sw_combo, sw_diag], combine_attrs='drop_conflicts')
            sw_combo = sw_combo.expand_dims(sample_source=['seawater'], axis=1)

            sdvars = ['measurement_interval', 'gas_flow_rate', 'cooling_pad_temperature', 'li7815_cavity_pressure',
                      'li7815_xco2_stdev', 'li7815_cavity_temperature', 'li7815_h2o', 'number_of_measurements',
                      'li7815_diagnostic', 'li7815_xco2', 'instrument_mode', 'instrument_timestamp',
                      'measurement_flow_rate', 'measurement_stdev', 'purge_interval', 'purge_flow_rate', 'test_type']

            sd1 = sd1[sdvars]
            sd1 = sd1.expand_dims(sample_source=['sd1'], axis=1)

            sd2 = sd2[sdvars]
            sd2 = sd2.expand_dims(sample_source=['sd2'], axis=1)

            sd3 = sd3[sdvars]
            sd3 = sd3.expand_dims(sample_source=['sd3'], axis=1)

            sd4 = sd4[sdvars]
            sd4 = sd4.expand_dims(sample_source=['sd4'], axis=1)

            sd5 = sd5[sdvars]
            sd5 = sd5.expand_dims(sample_source=['sd5'], axis=1)

            try:
                combo = xr.combine_by_coords([air_combo, sw_combo, sd1, sd2, sd3, sd4, sd5], combine_attrs = 'drop_conflicts')
            except:
                combo = xr.concat([air_combo, sw_combo, sd1, sd2, sd3, sd4, sd5], combine_attrs = 'drop_conflicts', dim = 'time')
                combo = combo.drop_duplicates(dim = 'time')
            combo = combo[sorted(combo.data_vars)]
            combo = combo.sortby('time')
            return combo
        else:
            if self._verbose is True:
                warnings.warn(f'No Data {bdt}- {edt}')
            return None


    def get_centerboard_sbe38(self, bdt: datetime, edt: datetime):
        ds = self.get_data('sensor_float_6', bdt, edt)
        return ds

    def get_bowthruster_sbe38(self, bdt: datetime, edt: datetime):
        ds = self.get_data('sensor_float_9', bdt, edt)
        return ds

    def get_rain_gauge(self, bdt: datetime, edt: datetime):
        ds = self.get_data('sensor_mixed_18', bdt, edt)
        return ds

    def get_gps_gga_bow(self, bdt: datetime, edt: datetime):
        ds = self.get_data('gnss_gga_bow', bdt, edt)
        ds = ds.drop_vars(['geo_point'], errors = 'ignore')
        return ds


    def get_gps_vtg_bow(self, bdt: datetime, edt: datetime):
        ds = self.get_data('gnss_vtg_bow', bdt, edt)
        return ds

    def get_gps_cs_nav(self, bdt: datetime, edt: datetime):
        ds = self.get_data('sensor_mixed_9', bdt, edt)
        return ds



    def get_heading(self, bdt: datetime, edt: datetime):
        ds = self.get_data('sensor_float_1', bdt, edt)
        return ds


    def get_main_lab_tripleta(self, bdt: datetime, edt: datetime):
        ds = self.get_data('sensor_mixlg_2', bdt, edt)
        return ds


    def get_fwd_mast_wind_a(self, bdt: datetime, edt: datetime):
        ds = self.get_data('SensorFloat16', bdt, edt)
        return ds

    def get_fwd_mast_wind_b(self, bdt: datetime, edt: datetime):
        ds = self.get_data('sensor_mixed_1', bdt, edt)
        return ds


    def get_met_station(self, bdt: datetime, edt: datetime):
        ds = self.get_data('sensor_mixed_5', bdt, edt)
        return ds

    def get_main_lab_sunav2(self, bdt: datetime, edt: datetime):
        ds = self.get_data('sensor_mixlg_1', bdt, edt)
        return ds

    def get_main_lab_oxygen(self, bdt: datetime, edt: datetime):
        ds = self.get_data('SensorFloat13', bdt, edt)
        return ds


    def get_centerboard_depth(self, bdt: datetime, edt: datetime):
        ds = self.get_data('sensor_integer_3', bdt, edt)
        return ds


    def get_flowmeter_1(self, bdt: datetime, edt: datetime):
        ds = self.get_data('sensor_float_10', bdt, edt)
        return ds