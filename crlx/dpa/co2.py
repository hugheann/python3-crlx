import numpy as np
from typing import Union

from crlx.converters import mbar2atm, C2K, MPa2atm, uatm2atm

class CO2():
    """Generic functions for CO2 data products."""

    def __init__(self):
        self.R_SI = 8.31447  # Gas Constant in J * mol^-1 * K^-1
        self.R_OTHER = 8.20573660809596 * 10 # Gas Constant in cm^3 atm K^-1 mol^-1

    def calc_xco2_corr(self, xco2_raw: Union[float,np.array],
                       xco2_r1: Union[float,np.array], xco2_r2: Union[float,np.array],
                       xco2_s1: Union[float,np.array], xco2_s2: Union[float,np.array]) -> Union[float,np.array]:
        """
        Compute a corrected xCO2 value from known standard values measured by the same analyzer.
            Standard #1 and Standard #2 represent the two standards that bracket the measured value.
            They do not represent the order that the standards were sampled in.
            Most underway pCO2 systems will inherently perform this calculation, so it is provided here for reference
                and/or replication purposes.

        :param xco2_raw: The raw xCO2 value output by the analyzer.
        :param xco2_r1: The raw value of standard #1 measured by the analyzer.
        :param xco2_r2: The raw value of standard #2 measured by the analyzer.
        :param xco2_s1: The canister value of standard #1 as stated by the standard manufacturer.
        :param xco2_s2: The canister value of standard #1 as stated by the standard manufacturer.
        :return: The standard corrected xCO2 value.
        """

        xco2_corr = (xco2_raw - xco2_r1) * (xco2_s2 - xco2_s1)/(xco2_r2 - xco2_r1) + xco2_s1
        return xco2_corr


    def calc_pco2(self, xco2_corr: Union[float,np.array],
                  barometric_pressure: Union[float,np.array],
                  water_vapor_pressure: Union[float,np.array]) -> Union[float,np.array]:
        """
        Convert xCO2 to 100% humidity pCO2. This function can be used to calculate pCO2 in the equilibrator
            and atmospheric pCO2.
        References: Apollo ASP3 Manual v2024.6, pg 31

        Uses:
            equilibrator_pco2 = calc_pco2(xco2_corr, equilibrator_pressure, equilibrator_water_vapor_pressure)
            atmospheric_pco2 = calc_pco2(xco2_corr, barometric_pressure_sea_surface, sea_surface_water_vapor_pressure)

        :param xco2_corr: The standard corrected xCO2 value in ppm.
        :param barometric_pressure: Requires units of atm.
            If computing pCO2 in the equilibrator, this value should represent the  pressure in the equilibrator.
            If computing atmospheric pCO2, this value should represent the barometric pressure near the sea surface.
        :param water_vapor_pressure: Requires units of atm.
            If computing pCO2 in the equilibrator, this value should represent the water vapor pressure
                calculated from the temperature and salinity within the equilibrator.
            If computing atmospheric pCO2, this values should represent the water vapor pressure
                calculated from the temperature and salinity at the intake.
        :return: The 100% humidity pCO2 in micro-atm (uatm).
        """

        pco2 = xco2_corr * (barometric_pressure - water_vapor_pressure)
        return pco2


    def calc_sea_water_pco2_t_corr(self, equilibrator_pco2: Union[float, np.array],
                                   intake_temperature: Union[float,np.array],
                                   equilibrator_temperature: Union[float, np.array]) -> Union[float, np.array]:
        """
        Apply a sea-surface temperature correction to the pCO2 measured in the sensor equilibrator.
        References: Takahashi et al, 2002.
                    Apollo ASP3 Manual v2024.6, pg 31

        :param equilibrator_pco2: The seawater pCO2, measured through the equilibrator.
        :param intake_temperature: The temperature at the vessel or platform flowthrough intake.
        :param equilibrator_temperature: The temperature of the seawater in the equilibrator.
        :return: The seawater pCO2 corrected for the difference between the SST and the equilibrator.
        """

        sea_water_pco2_t_corr = equilibrator_pco2 * np.exp(0.0423 * (intake_temperature - equilibrator_temperature))
        if isinstance(sea_water_pco2_t_corr, np.float64 or np.float32 or np.float16):
            sea_water_pco2_t_corr = float(sea_water_pco2_t_corr)
        return sea_water_pco2_t_corr


    def calc_water_vapor_pressure(self, sea_water_temperature: Union[float,np.array],
                                  sea_water_practical_salinity: Union[float,np.array]) -> Union[float,np.array]:
        """
        Compute water vapor pressure for a given seawater temperature and salinity.
        References: Guide to Best Practices for Ocean CO2 Measurements, PICES Special Publication 3,
            IOCCP Report No. 8, Dickson et al, 2007.

        :param sea_water_temperature: The measured seawater temperature, usually from the intake sensor.
        :param sea_water_practical_salinity: The measured seawater practical salinity, from a sensor in flowthrough.
        :return: The water vapor pressure in atm.
        """

        a1 = -7.85951783
        a2 = 1.84408259
        a3 = -11.7866497
        a4 = 22.6807411
        a5 = -15.9618719
        a6 = 1.80122502
        tcrit = 647.096  # Critical point temperature in Kelvin
        pcrit = 22.064  # Critical point pressure in MPa
        tk = C2K(sea_water_temperature)  # Convert Celsius to Kelvin.
        eta = 1 - (tk/tcrit)
        a = a1 * eta + a2 * eta**1.5 + a3 * eta**3 + a4 * eta**3.5 + a5 * eta**4 + a6 * eta**7.5
        p_sigma = pcrit * np.exp((tcrit/tk)*a)
        m_total = ((31.998 * sea_water_practical_salinity)/
                   ((10**3) - (1.005 * sea_water_practical_salinity))) # total molality of dissolved species
        m_circ = 1
        phi = (0.90799 - 0.08992 * ((0.5 * m_total)/m_circ) + 0.18458 * ((0.5 * m_total)/m_circ) ** 2 -
               0.07395 * ((0.5 * m_total)/m_circ) ** 3 - 0.00221 * ((0.5 * m_total)/m_circ) ** 4)
        p_sigma_sw = p_sigma * np.exp(-0.0180 * phi * m_total/m_circ)  # in MPa.
        water_vapor_pressure = MPa2atm(p_sigma_sw)  # Convert from MPa to atm.
        if isinstance(water_vapor_pressure, np.float64 or np.float32 or np.float16):
            water_vapor_pressure = float(water_vapor_pressure)
        return water_vapor_pressure


    def calc_fugacity(self, pco2: Union[float,np.array],
                      xco2_corr:  Union[float,np.array],
                      barometric_pressure: Union[float,np.array],
                      temperature:  Union[float,np.array]) -> Union[float, np.array]:
        """
        Calculate the fugacity of atmospheric or seawater CO2.

        References: Wanninkhof et al, 2020 - https://essd.copernicus.org/preprints/essd-2019-245/essd-2019-245.pdf
                    Weiss, 1974

        :param pco2: The pCO2 of air or seawater in uatm.
        :param xco2_corr: The corrected xCO2 value of air or seawater in ppm.
        :param barometric_pressure: The barometric pressure at the sea surface.
        :param temperature: Temperature depends on the source of the measurement.
            It can either be the SST OR air temperature as measured by a sensor closest to the water.
        :return: The fugacity of CO2 in air or seawater.
        """

        xco2_atm = uatm2atm(xco2_corr) # Get xCO2 into atm.
        tk = C2K(temperature)  # Get temperature in Kelvin
        bpres_atm = mbar2atm(barometric_pressure)  # Get barometric pressure into atm.
        d_co2_air = 57.7 - 0.118 * tk  # in cm3/mol
        bt = -1636.75 + (12.040 * tk) - (3.27957 * 10 ** -2 * tk ** 2) + (3.16528 * 10 ** -5 * tk ** 3)  # in cm3/mol
        g = (bt + 2 * (1 - xco2_atm)**2 * d_co2_air) * (bpres_atm)/(self.R_OTHER * tk)
        fco2 = np.exp(g) * pco2  # In uatm.
        if isinstance(fco2, np.float64 or np.float32 or np.float16):
            fco2 = float(fco2)
        return fco2


    def calc_co2_flux(self, fco2a: Union[float,np.array],
                      fco2w: Union[float,np.array],
                      u10: Union[float,np.array],
                      sea_water_temperature: Union[float,np.array],
                      sea_water_practical_salinity: Union[float,np.array]) -> Union[float,np.array]:
        """
        Compute the flux of CO2 from the ocean to the atmosphere.
            Negative values indicate flux INTO the ocean.
            Positive values indicate flux INTO the atmosphere.
            This code is reworked from the Ocean Observatories Initiative Data Product Specification
            for the flux of CO2 from the ocean into the atmosphere (v1-03). The primary difference is that this
            implementation uses the fugacity rather than partial pressure of CO2.

        References: Weiss, 1974
                    Wanninkhof, 1992
                    Sweeney et al., 2007

        :param fco2a: The fugacity of CO2 in the atmosphere in uatm.
        :param fco2w: The fugacity of CO2 in seawater in uatm.
        :param u10: The wind speed at 10m above the sea surface in m/s.
        :param sea_water_temperature: The temperature of seawater in Celsius.
        :param sea_water_practical_salinity: The salinity of seawater in PSU.
        :return: The flux of CO2 across the air-sea interface.
        """

        Sc = (2073.1 - (125.62 * sea_water_temperature) + (3.6276 * sea_water_temperature ** 2) -
              (0.043219 * sea_water_temperature ** 3))  # Compute the Schmidt Number.
        k = 0.27 * (u10 ** 2) * np.sqrt(Sc/660) # Compute gas transfer velocity in cm/hr.
        k = k / 360000  # Convert to m/s.
        tk = C2K(sea_water_temperature) # Units of K.
        K0 = 1000 * np.exp(-58.0931 + (90.5069 * (100/tk)) +
                           (22.2940 * np.log(tk/100)) +
                           sea_water_practical_salinity * (0.027766 - (0.025888 * (tk/100)) +
                                                           (0.0050578 * (tk/100)**2)))  # Compute solubility.
        dfco2 = uatm2atm(fco2w) - uatm2atm(fco2a)  # Convert uatm to atm.
        flux = k * K0 * (dfco2)
        if isinstance(flux, np.float64 or np.float32 or np.float16):
            flux = float(flux)
        return flux

