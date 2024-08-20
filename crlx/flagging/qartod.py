import numpy as np
import xarray as xr


class FLAG:
    PASS: int = 1
    NOT_EVALUATED: int = 2
    HIGH_INTEREST: int = 3
    SUSPECT: int = 3
    FAIL: int = 4
    MISSING_DATA: int = 9


def location_test(latitude: xr.DataArray, longitude: xr.DataArray) -> xr.DataArray:
    """
    Determine if a given latitude and longitude are valid.

    :param latitude: xr.DataArray of lat values with a 'time' coordinate.
    :param longitude: xr.DataArray of lon values with a 'time' coordinate that has the same timestamps as the lat.
    :return: xr.DataArray of test results.
    """
    results = xr.full_like(latitude, fill_value=FLAG.NOT_EVALUATED).astype('int8')  # Assign NOT_EVALUATED by default.
    results = results.where((np.abs(latitude) < 90) | (np.abs(longitude) < 180), FLAG.FAIL)
    results = results.where((np.abs(latitude) > 90) & (np.abs(longitude) > 180), FLAG.PASS)
    results = results.where(~np.isnan(latitude) | ~np.isnan(longitude), FLAG.MISSING_DATA)  # Set to nan if missing.

    results.attrs['test_name'] = 'QARTOD Location Test'
    results.attrs['is_qartod'] = "True"
    results.attrs['is_experimental'] = "False"
    results.attrs[
        'description'] = """The QARTOD Location Test verifies if a given latitude and longitude for an associated data 
        point are within the confines of reality. Reality is defined as -90 to 90 for latitude and -180 to 180 for 
        longitude. Implementation of this test does not currently support the flagging of suspect values."""
    results.attrs['valid_flags'] = "1 = PASS, 2 = NOT_EVALUATED, 4 = FAIL, 9 = MISSING_DATA"
    results.attrs['valid_latitude_range'] = [-90,90]
    results.attrs['valid_longitude_range'] = [-180,180]
    return results


def gross_range_test(data: xr.DataArray, sensor_min: float, sensor_max: float,
                     operator_min: float or None = None, operator_max: float or None = None) -> xr.DataArray:
    """
     Determine if data are within appropriate sensor or operator defined ranges.
    :param data: The input data.
    :param sensor_min: The minimum value the sensor can observe.
    :param sensor_max: The maximum value the sensor can observe.
    :param operator_min: An operator/user defined minimum value. Example: Minimum factory calibrated value.
    :param operator_max: An operator/user defined maximum value. Example: Maximum factory calibrated value.
    :return: An xr.DataArray of results.
    """

    results = xr.full_like(data, fill_value=FLAG.NOT_EVALUATED).astype('int8')
    results = results.where((data < sensor_min) & (data > sensor_max), FLAG.PASS)
    results = results.where((data > sensor_min) | (data < sensor_max), FLAG.FAIL)

    if operator_min is not None:
        if sensor_min != operator_min:
             results = results.where((data > operator_min) | (data < sensor_min), FLAG.SUSPECT)
    if operator_max is not None:
        if sensor_max != operator_max:
            results = results.where((data < operator_max) | (data > sensor_max), FLAG.SUSPECT)

    results = results.where(~np.isnan(data), FLAG.MISSING_DATA)

    results.attrs['test_name'] = f'QARTOD Gross Range Test - {data.name}'
    results.attrs['is_qartod'] = "True"
    results.attrs['is_experimental'] = "False"
    results.attrs[
        'description'] = """The QARTOD Gross Range Test verifies if a given data point is within the limits defined 
        by the sensor manufacturer (and operator)."""
    results.attrs[
        'valid_flags'] = "1 = PASS, 2 = NOT_EVALUATED, 3 = HIGH_INTEREST or SUSPECT, 4 = FAIL, 9 = MISSING_DATA"
    results.attrs['sensor_min'] = sensor_min
    results.attrs['sensor_max'] = sensor_max
    if operator_min is not None:
        results.attrs['operator_min'] = operator_min
    else:
        results.attrs['operator_min'] = 'No Operator Value Supplied'
    if operator_max is not None:
        results.attrs['operator_max'] = operator_max
    else:
        results.attrs['operator_max'] = 'No Operator Value Supplied'
    results.attrs['valid_min'] = sensor_min
    results.attrs['valid_max'] = sensor_max
    return results


def climatology_test(data: xr.DataArray,
                     latitude: xr.DataArray, longitude: xr.DataArray,
                     lookup: xr.Dataset) -> xr.DataArray:
    climatology = lookup.attrs['temporal_resolution']
    min_var = [v for v in list(lookup.data_vars) if 'min' in v]
    if len(min_var) > 1:
        raise LookupError('More than one minimum variable found in lookup dataset.')
    else:
        min_var = min_var[0]
    max_var = [v for v in list(lookup.data_vars) if 'max' in v]
    if len(max_var) > 1:
        raise LookupError('More than one maximum variable found in lookup dataset.')
    else:
        max_var = max_var[0]

    if climatology == 'dayofyear':  # Cells are assigned a time, making vectorized processing possible.
        cells = lookup.sel(dayofyear=data.time.dt.dayofyear, latitude=latitude, longitude=longitude, method='nearest')
    elif climatology == 'month':
        cells = lookup.sel(month=data.time.dt.month, latitude=latitude, longitude=longitude, method='nearest')
    elif climatology == 'season':
        cells = lookup.sel(season=data.time.dt.season, latitude=latitude, longitude=longitude, method='nearest')

    results = xr.full_like(data, FLAG.NOT_EVALUATED).astype('int8')
    results = results.where((cells[min_var] < data) | (cells[max_var] > data), FLAG.HIGH_INTEREST)
    results = results.where((cells[min_var] > data) | (cells[max_var] < data), FLAG.PASS)

    results = results.where((~np.isnan(data)), FLAG.MISSING_DATA)
    results = results.where((~np.isnan(cells[min_var])) | (~np.isnan(cells[max_var])), FLAG.NOT_EVALUATED)

    results = results.drop_vars(['latitude', 'longitude', 'dayofyear', 'month'], errors='ignore')

    results.attrs['test_name'] = f'QARTOD Climatology Test - {data.name}'
    results.attrs['is_qartod'] = "True"
    results.attrs['is_experimental'] = "False"
    results.attrs[
        'description'] = """The QARTOD Climatology Test verifies if a sample is within the minimum and maximum 
        expected value for a given time of the year and location. The nearest grid cell and time from the 
        reference dataset are used when flagging data."""
    results.attrs['valid_flags'] = "1 = PASS, 2 = NOT_EVALUATED, 3 = HIGH_INTEREST or SUSPECT, 9 = MISSING_DATA"
    results.attrs['reference_dataset'] = lookup.attrs['dataset']
    results.attrs['temporal_reference_resolution'] = lookup.attrs['temporal_resolution']
    results.attrs['spatial_reference_resolution'] = lookup.attrs['spatial_resolution']
    results.attrs['spatial_reference_units'] = lookup.attrs['spatial_resolution_units']
    results.attrs['reference_dataset'] = lookup.attrs['dataset']

    return results

def spike_test(data: xr.DataArray, std_window: int = 1 * 30 + 1, low_multiplier=3, high_multiplier=5):
    # Assume data is already sorted by time?????
    data = data.sortby('time')

    spkref_windows = data.rolling({'time': 3}, min_periods=1).construct('window_dim')
    spkref_left = spkref_windows[:, 0]
    spkref_right = spkref_windows[:, -1]
    spkref = (spkref_left + spkref_right) / 2

    sd = data.rolling({'time': std_window}, center=True, min_periods=1).std()
    threshold_low = low_multiplier * sd
    threshold_high = high_multiplier * sd


    results = xr.full_like(data, FLAG.NOT_EVALUATED).astype('int8')
    results = results.where(~(np.abs(data - spkref) < threshold_low) & ~(np.abs(data - spkref) > threshold_high),
                            FLAG.PASS)
    results = results.where((np.abs(data - spkref) < threshold_low) | (np.abs(data - spkref) > threshold_high),
                            FLAG.HIGH_INTEREST)
    results = results.where(~(np.abs(data - spkref) > threshold_high), FLAG.FAIL)
    results = results.where((~np.isnan(data)), FLAG.MISSING_DATA)
    results = results.where((~np.isnan(spkref)), FLAG.MISSING_DATA)
    results = results.where((~np.isnan(threshold_low)) | (~np.isnan(threshold_high)), FLAG.NOT_EVALUATED)

    results.attrs['test_name'] = f'QARTOD Spike Test - {data.name}'
    results.attrs['is_qartod'] = "True"
    results.attrs['is_experimental'] = "False"
    results.attrs[
        'description'] = """The QARTOD Spike Test uses the neighboring points (n-1, n+1) to determine if data (n) 
        is unusually high or low, indicative of a spike that could be representative of unusual environmental 
        variation or sensor failure depending on the sampling rate."""
    results.attrs[
        'valid_flags'] = "1 = PASS, 2 = NOT_EVALUATED, 3 = HIGH_INTEREST or SUSPECT, 4 = FAIL, 9 = MISSING_DATA"
    results.attrs['window_size'] = std_window
    results.attrs['lower_threshold_std_multiplier'] = low_multiplier
    results.attrs['upper_threshold_std_multiplier'] = high_multiplier
    return results

def rate_of_change_test(data: xr.DataArray, std_multiplier: int = 3):
    _delta_data = data.diff(dim='time', n=1)
    _delta_time = data.time.diff(dim='time').astype(float) / 1000000000
    rate = _delta_data / _delta_time
    rate = np.concatenate([np.array([np.nan]), rate])
    results = xr.full_like(data, FLAG.NOT_EVALUATED).astype('int8')
    results = results.where(np.abs(rate) > std_multiplier * rate.std(), FLAG.PASS)
    results = results.where(~(np.abs(rate) > std_multiplier * rate.std()), FLAG.HIGH_INTEREST)
    results = results.where((~np.isnan(data)), FLAG.MISSING_DATA)
    results = results.where((~np.isnan(rate)), FLAG.MISSING_DATA)

    results.attrs['test_name'] = f'QARTOD Rate of Change Test - {data.name}'
    results.attrs['is_qartod'] = "True"
    results.attrs['is_experimental'] = "False"
    results.attrs[
        'description'] = """The QARTOD Rate of Change Test assesses the rate of change of a variable over time. 
        Currently the implementation of this test estimates the rate of change between each data point, 
        regardless of the sampling rate. This test does not flag BAD or FAIL data."""
    results.attrs['valid_flags'] = "1 = PASS, 2 = NOT_EVALUATED, 3 = HIGH_INTEREST or SUSPECT, 9 = MISSING_DATA"
    results.attrs['std_multiplier'] = std_multiplier

    return results


def flat_line_test(data: xr.DataArray, fail_window_size: int, suspect_window_size: int, eps: float) -> xr.DataArray:
    windows = data.rolling({'time': fail_window_size}).construct('window_dim')  # Create a nd array of windows.
    points = windows[:, -1]  # Get the 'n' of each window.
    bools = np.abs(points - windows) < eps  # Determine if the rest of the window is within the noise threshold.

    results = xr.full_like(data, fill_value=FLAG.PASS).astype('int8')
    results = results.where(np.all(bools[:, :-1], axis=1) != True, FLAG.FAIL)
    results = results.where(np.all(bools[:, -suspect_window_size - 1:-1], axis=1) != True, FLAG.SUSPECT)
    results = results.where(~np.any(np.isnan(windows), axis=1), FLAG.NOT_EVALUATED)
    results = results.where(~np.isnan(points), FLAG.MISSING_DATA)

    results.attrs['test_name'] = f'QARTOD Flat Line Test - {data.name}'
    results.attrs['is_qartod'] = "True"
    results.attrs['is_experimental'] = "False"
    results.attrs[
        'description'] = """The QARTOD Flat Line Test assesses if data are reasonably variable. 
        Failure of the flat line may suggest that the sensor is repeating values or is obstructed."""
    results.attrs[
        'valid_flags'] = "1 = PASS, 2 = NOT_EVALUATED, 3 = HIGH_INTEREST or SUSPECT, 4 = FAIL, 9 = MISSING_DATA"
    results.attrs['fail_window_size'] = fail_window_size
    results.attrs['suspect_window_size'] = suspect_window_size
    results.attrs['variation_threshold'] = eps

    return results




def multi_variate_test(data1: xr.DataArray, data2: xr.DataArray,
                       data1_multiplier: int = 3, data2_multiplier: int = 2,
                       sel_method: str = 'nearest', sel_tolerance: str = '5s') -> xr.DataArray:
    if np.all(np.isnan(data2)):
        return xr.full_like(data1, fill_value=FLAG.MISSING_DATA).astype('int8')
    if np.isnan(data1_multiplier) or np.isnan(data2_multiplier):
        return xr.full_like(data1, fill_value=FLAG.NOT_EVALUATED).astype('int8')

    if sel_method is not None:  # If no method, it is assumed that data2 has the same timestamps as data1.
        data2 = data2.sel(time=data1.time, method=sel_method, tolerance=sel_tolerance)
    roc1 = rate_of_change_test(data1, data1_multiplier)
    roc2 = rate_of_change_test(data2, data2_multiplier)
    results = xr.where((roc1.values == 3) & (roc2.values == 1), 3, roc1)

    results.attrs['test_name'] = f'QARTOD Multi-Variate Test - {data1.name}'
    results.attrs['is_qartod'] = "True"
    results.attrs['is_experimental'] = "False"
    results.attrs[
        'description'] = """The QARTOD Multi-Variate Test assesses if variable #1 varies at similar rate of change 
        compared to variable #2. In this implementation of the test, the rate of change test is run on each variable. 
        If variable #2 passes the RoC test, but variable #1 does not, data are flagged as suspect. 
        This test does not fail data."""
    results.attrs['valid_flags'] = "1 = PASS, 2 = NOT_EVALUATED, 3 = HIGH_INTEREST or SUSPECT, 9 = MISSING_DATA"
    results.attrs['variable_1'] = data1.name
    results.attrs['variable_2'] = data2.name
    results.attrs['variable_1_multiplier'] = data1_multiplier
    results.attrs['variable_2_multiplier'] = data2_multiplier
    results.attrs['mismatch_method'] = sel_method
    results.attrs['mismatch_tolerance'] = sel_tolerance
    return results


def attenuated_signal_test(data: xr.DataArray, min_var_fail: float, min_var_warn: float)-> xr.DataArray:
    std = data.std()
    maxmin = data.max() - data.min()
    results = xr.full_like(data, FLAG.NOT_EVALUATED).astype('int8')
    results = results.where(~(std<min_var_fail) | ~(maxmin < min_var_fail), FLAG.FAIL)
    results = results.where(~(std<min_var_warn) | ~(maxmin < min_var_warn), FLAG.SUSPECT)
    results = results.where((std<min_var_warn) | (maxmin < min_var_warn), FLAG.PASS)
    results = results.where((~np.isnan(min_var_fail))| (~np.isnan(min_var_warn)), FLAG.NOT_EVALUATED)
    results = results.where((~np.isnan(data)), FLAG.MISSING_DATA)

    results.attrs['test_name'] = f'QARTOD Attenuated Signal Test - {data.name}'
    results.attrs['is_qartod'] = "True"
    results.attrs['is_experimental'] = "False"
    results.attrs[
        'description'] = "The QARTOD Attenuated Signal Test assesses if the range of data are acceptable."
    results.attrs['valid_flags'] = "1 = PASS, 2 = NOT_EVALUATED, 3 = HIGH_INTEREST or SUSPECT, 9 = MISSING_DATA"


    return results


def neighbor_test(data1: xr.DataArray, data2: xr.DataArray,
                  data1_multiplier: int = 3, data2_multiplier: int = 2,
                  sel_method: str = 'nearest', sel_tolerance: str = '5s') -> xr.DataArray:
    results = multi_variate_test(data1, data2, data1_multiplier, data2_multiplier, sel_method, sel_tolerance)

    results.attrs['test_name'] = f'QARTOD Neighbor Test - {data1.name}'
    results.attrs['is_qartod'] = "True"
    results.attrs['is_experimental'] = "False"
    results.attrs[
        'description'] = """The QARTOD Neighbor Test assesses if variable #1 varies at similar rate of change compared 
        to variable #2. It is similar to the multi-variate test, but the expectation is that both input variables 
        represent the same product from two different sensors. In this implementation of the test, the rate of change 
        test is run on each variable. If variable #2 passes the RoC test, but variable #1 does not, data are flagged 
        as suspect. This test does not fail data."""
    results.attrs['valid_flags'] = "1 = PASS, 2 = NOT_EVALUATED, 3 = HIGH_INTEREST or SUSPECT, 9 = MISSING_DATA"
    results.attrs['variable_1'] = data1.name
    results.attrs['variable_2'] = data2.name
    results.attrs['variable_1_multiplier'] = data1_multiplier
    results.attrs['variable_2_multiplier'] = data2_multiplier
    results.attrs['mismatch_method'] = sel_method
    results.attrs['mismatch_tolerance'] = sel_tolerance
    return results