import numpy as np
from typing import Union


# Temperature Conversions
def C2K(celsius: Union[float, np.array]) -> Union[float, np.array]:
    """
     Convert Celsius to Kelvin.

     :param celsius: The temperature in Celsius.
     :return: The temperature in Kelvin.
     """

    kelvin = celsius + 273.15
    return kelvin



# Pressure Conversions
def mbar2atm(mbar: Union[float, np.array]) -> Union[float, np.array]:
    """
     Convert millibar to atmosphere.

     :param mbar: The pressure in millibars.
     :return: The pressure in atmospheres.
     """

    atm = 0.000986923 * mbar
    return atm


def bar2atm(bar: Union[float, np.array]) -> Union[float, np.array]:
    """
     Convert bar to atmosphere.

     :param bar: The pressure in bars.
     :return: The pressure in atmospheres.
     """

    atm = mbar2atm(bar*1000)
    return atm




def MPa2atm(MPa: Union[float, np.array]) -> Union[float, np.array]:
    """
     Convert MegaPascals to atmosphere.

     :param MPa: The pressure in MegaPascals.
     :return: The pressure in atmospheres.
     """

    atm = 9.8692326716013 * MPa
    return atm


def uatm2atm(uatm: Union[float, np.array]) -> Union[float, np.array]:
    """
     Convert micro-atmospheres to atmosphere.

     :param uatm: The pressure in micro-atmospheres.
     :return: The pressure in atmospheres.
     """

    atm = uatm/1.0e6
    return atm