from datetime import datetime,timedelta
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

    def get_wet_lab_ldeo(self, bdt: datetime, edt: datetime):

        def recompute_air(ds, barometric_pressure, sst,sal, at):
            dpaCO2 = CO2()
            diag = ds[['cell_temperature','gas_flow_rate','source_id']]

            xco2_corr = ds.xco2_corr
            bp = barometric_pressure * 1000
            combo = xr.combine_by_coords([xco2_corr, sst, bp, sal,at])
            combo = combo.interpolate_na(dim='time', method='nearest', max_gap=timedelta(seconds=10))
            combo = combo.sel(time=xco2_corr.time)
            combo['air_water_vapor_pressure'] = dpaCO2.calc_water_vapor_pressure(combo.sea_surface_temperature,
                                                                                 combo.sea_water_practical_salinity)
            combo['air_pco2'] = dpaCO2.calc_pco2(combo.xco2_corr, mbar2atm(combo.barometric_pressure),
                                                 combo.air_water_vapor_pressure)
            combo['air_fco2'] = dpaCO2.calc_fugacity(combo.air_pco2, combo.xco2_corr, combo.barometric_pressure,
                                                     combo.air_temperature)
            combo = xr.combine_by_coords([combo, diag], combine_attrs='drop_conflicts')
            return combo


        def recompute_sw(ds, barometric_pressure, sst,sal):
            dpaCO2 = CO2()


            diag = ds[['cell_temperature','gas_flow_rate','source_id']]


            xco2_corr = ds.xco2_corr
            bp = barometric_pressure * 1000
            teq = ds.equilibrator_temperature_sbe38
            peq = ds.equilibrator_pressure

            combo = xr.combine_by_coords([xco2_corr, sst, bp, sal, teq, peq])
            combo = combo.interpolate_na(dim='time', method='nearest', max_gap=timedelta(seconds=10))
            combo = combo.sel(time=xco2_corr.time)
            combo['equilibrator_water_vapor_pressure'] = dpaCO2.calc_water_vapor_pressure(
                combo.equilibrator_temperature_sbe38, combo.sea_water_practical_salinity)
            combo['equilibrator_pco2'] = dpaCO2.calc_pco2(combo.xco2_corr, mbar2atm(combo.equilibrator_pressure),
                                                          combo.equilibrator_water_vapor_pressure)
            combo['sea_water_pco2'] = dpaCO2.calc_sea_water_pco2_t_corr(combo.equilibrator_pco2,combo.sea_surface_temperature,mbar2atm(combo.barometric_pressure))
            combo['sea_water_fco2'] = dpaCO2.calc_fugacity(combo.equilibrator_pco2, combo.xco2_corr,
                                                           combo.barometric_pressure, combo.sea_surface_temperature)

            combo = xr.combine_by_coords([combo, diag], combine_attrs='drop_conflicts')
            return combo


        def reformat_standards(ds):
            ds = ds[['cell_temperature','gas_flow_rate','source_id','xco2_corr']]

            return ds


        met = self.get_met_station(bdt-timedelta(hours = 2), edt +timedelta(hours = 2))[['barometric_pressure','air_temperature']]
        at = met.air_temperature
        bp = met.barometric_pressure
        sst = self.get_centerboard_sbe38(bdt - timedelta(hours=2), edt + timedelta(hours=2)).rename({'sea_water_temperature': 'sea_surface_temperature'})['sea_surface_temperature']
        sal = self.get_main_lab_sbe45a(bdt - timedelta(hours=2), edt + timedelta(hours=2))['sea_water_practical_salinity']




        ds = self.get_data('SensorFloat17', bdt, edt)
        air = ds.where(ds.source_id == 1, drop=True)
        sw = ds.where(ds.source_id == 0, drop=True)
        nit = ds.where(ds.source_id == 2, drop=True)
        sd1 = ds.where(ds.source_id == 4, drop=True)
        sd2 = ds.where(ds.source_id == 6, drop=True)
        sd3 = ds.where(ds.source_id == 8, drop=True)
        sd4 = ds.where(ds.source_id == 10, drop=True)
        sd5 = ds.where(ds.source_id == 12, drop=True)

        air = recompute_air(air,bp,sst,sal,at)
        sw = recompute_sw(sw,bp,sst,sal)
        nit = reformat_standards(nit)
        sd1 = reformat_standards(sd1)
        sd2 = reformat_standards(sd2)
        sd3 = reformat_standards(sd3)
        sd4 = reformat_standards(sd4)
        sd5 = reformat_standards(sd5)

        try:
            cds = xr.combine_by_coords([air, sw, nit, sd1, sd2, sd3, sd4, sd5], combine_attrs='drop_conflicts')
        except:
            cds = xr.concat([air, sw, nit, sd1, sd2, sd3, sd4, sd5], combine_attrs='drop_conflicts', dim='time')
            cds = cds.drop_duplicates(dim='time')
        return cds

    def get_wet_lab_apollo(self, bdt: datetime, edt: datetime):

        def recompute_air(ds, barometric_pressure, sst,at):
            dpaCO2 = CO2()

            diag = ds[
                ['measurement_interval', 'gas_flow_rate', 'cooling_pad_temperature', 'li7815_cavity_pressure',
                 'li7815_co2_stdev', 'li7815_cavity_temperature',
                 'li7815_h2o', 'number_of_measurements', 'li7815_diagnostic', 'sample_source','sea_water_temperature','sea_water_practical_salinity','purge_interval']]
            diag = diag.rename({'sea_water_temperature':'apollo_sbe45_temp',
                                'sea_water_practical_salinity': 'apollo_sbe45_pracsal'})

            xco2_corr = ds.xco2_corr
            sal = ds.sea_water_practical_salinity
            bp = barometric_pressure * 1000
            combo = xr.combine_by_coords([xco2_corr, sst, bp, sal,at])
            combo = combo.interpolate_na(dim='time', method='nearest', max_gap=timedelta(seconds=10))
            combo = combo.sel(time=xco2_corr.time)
            combo['air_water_vapor_pressure'] = dpaCO2.calc_water_vapor_pressure(combo.sea_surface_temperature,
                                                                                 combo.sea_water_practical_salinity)
            combo['air_pco2'] = dpaCO2.calc_pco2(combo.xco2_corr, mbar2atm(combo.barometric_pressure),
                                                 combo.air_water_vapor_pressure)
            combo['air_fco2'] = dpaCO2.calc_fugacity(combo.air_pco2, combo.xco2_corr, combo.barometric_pressure,
                                                     combo.air_temperature)

            combo = xr.combine_by_coords([combo, diag], combine_attrs='drop_conflicts')
            return combo


        def recompute_sw(ds, barometric_pressure, sst):
            dpaCO2 = CO2()


            diag = ds[
                ['measurement_interval', 'gas_flow_rate', 'cooling_pad_temperature', 'li7815_cavity_pressure',
                 'li7815_co2_stdev', 'li7815_cavity_temperature',
                 'li7815_h2o', 'number_of_measurements', 'li7815_diagnostic', 'sample_source','sea_water_temperature','sea_water_practical_salinity','purge_interval']]
            diag = diag.rename({'sea_water_temperature':'apollo_sbe45_temp',
                                'sea_water_practical_salinity': 'apollo_sbe45_pracsal'})

            xco2_corr = ds.xco2_corr
            sal = ds.sea_water_practical_salinity
            bp = barometric_pressure * 1000
            teq = ds.equilibrator_temperature
            peq = ds.equilibrator_pressure

            combo = xr.combine_by_coords([xco2_corr, sst, bp, sal, teq, peq])
            combo = combo.interpolate_na(dim='time', method='nearest', max_gap=timedelta(seconds=10))
            combo = combo.sel(time=xco2_corr.time)
            combo['equilibrator_water_vapor_pressure'] = dpaCO2.calc_water_vapor_pressure(
                combo.equilibrator_temperature, combo.sea_water_practical_salinity)
            combo['equilibrator_pco2'] = dpaCO2.calc_pco2(combo.xco2_corr, mbar2atm(combo.equilibrator_pressure),
                                                          combo.equilibrator_water_vapor_pressure)
            combo['sea_water_pco2'] = dpaCO2.calc_sea_water_pco2_t_corr(combo.equilibrator_pco2,combo.sea_surface_temperature,mbar2atm(combo.barometric_pressure))
            combo['sea_water_fco2'] = dpaCO2.calc_fugacity(combo.equilibrator_pco2, combo.xco2_corr,
                                                           combo.barometric_pressure, combo.sea_surface_temperature)

            combo = xr.combine_by_coords([combo, diag], combine_attrs='drop_conflicts')
            return combo

        def reformat_standards(ds):
            ds = ds[
                ['measurement_interval', 'gas_flow_rate', 'cooling_pad_temperature', 'li7815_cavity_pressure',
                 'li7815_co2_stdev', 'li7815_cavity_temperature',
                 'li7815_h2o', 'number_of_measurements', 'li7815_diagnostic', 'sample_source','li7815_xco2']]
            return ds

        ds = self.get_data('sensor_mixlg_11', bdt, edt)
        if ds is not None:
            ds = ds.rename({'sea_water_conductivity': 'sea_water_electrical_conductivity'})
            ds = ds.drop_vars(['gps_latitude',
                               'gps_longitude',
                               'operator',
                               'gps_datetime'],errors = 'ignore')


            air = ds.where(ds.sample_source.str.contains('Air'), drop=True)
            sw = ds.where(ds.sample_source.str.contains('Seawater'), drop=True)
            sd1 = ds.where(ds.sample_source.str.contains('SD-1'), drop=True)
            sd2 = ds.where(ds.sample_source.str.contains('SD-2'), drop=True)
            sd3 = ds.where(ds.sample_source.str.contains('SD-3'), drop=True)
            sd4 = ds.where(ds.sample_source.str.contains('SD-4'), drop=True)
            sd5 = ds.where(ds.sample_source.str.contains('SD-5'), drop=True)

            met = self.get_met_station(bdt-timedelta(hours = 2), edt +timedelta(hours = 2))[['barometric_pressure','air_temperature']]
            at = met.air_temperature
            bp = met.barometric_pressure
            sst = self.get_centerboard_sbe38(bdt - timedelta(hours=2), edt + timedelta(hours=2)).rename({'sea_water_temperature': 'sea_surface_temperature'})['sea_surface_temperature']

            air = recompute_air(air, bp, sst, at)
            sw = recompute_sw(sw, bp, sst)
            sd1 = reformat_standards(sd1)
            sd2 = reformat_standards(sd2)
            sd3 = reformat_standards(sd3)
            sd4 = reformat_standards(sd4)
            sd5 = reformat_standards(sd5)

            try:
                cds = xr.combine_by_coords([air, sw, sd1, sd2, sd3, sd4, sd5], combine_attrs = 'drop_conflicts')
            except:
                cds = xr.concat([air, sw, sd1, sd2, sd3, sd4, sd5], combine_attrs = 'drop_conflicts',dim = 'time')
                cds = cds.drop_duplicates(dim = 'time')
            cds = cds[sorted(cds.data_vars)]

        return cds

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