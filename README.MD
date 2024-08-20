# TODO
- Improve typing hints.
- 






## Quality Assessment and Control (QAQC)


### QARTOD 

The Quality Assessment/Control of Real-Time Oceanographic Data (QARTOD) project provides a set of manuals that contain tests for the assessing the quality of certain oceanographic variables.
A quick overview of these manuals reveals that there are common or generic tests that can be performed on most oceanographic time-series data.
Each manual will state whether tests are required, strongly recommended, or suggested. Test conditions and thresholds are often location, time, and operator dependent.

Some tests require prior data or neighboring data points. It is recommended to perform these tests for a specific window of interest with a buffer added on either end, making the results of these tests have a lag of seconds to hours depending on your application.
Adding a buffer will reduce the number of data points flagged as NOT_EVALUATED or MISSING_DATA. The results of tests should be used to critically assess the quality of data and should not be used to immediately discard data. 
Not all tests are perfect and depending on the thresholds used and bias, test results may underestimate or overestimate false flags.

#### Valid Flags
- 1 = PASS
- 2 = NOT_EVALUATED
  - Data flagged with 2 indicate that the test did not have valid input conditions (e.g. no available minimum/maximum for the climatology test)
- 3 = HIGH_INTEREST
  - Data flagged with 3 should be considered HIGH_INTEREST and can indicate interesting phenomena or possible sensor malfunction. Not all tests support this flag.
- 4 = FAIL
  - Data flagged with 4 can typically be considered of poor quality. Not all tests support this flag.
- 9 = MISSING_DATA
    - Data flagged with 9 indicated that the test did not receive a valid data value (e.g. NaN or a fill_value).



#### Common/Generic Tests
Generic QARTOD tests appear in each of the QARTOD manuals and share similar pseudo-code, regardless of the variable being assessed.

- Gap Test 
  - The gap test assesses if the timestamp of data point is considered valid.
- Syntax Test
  - The syntax test assess whether data from the sensor is in the correct format or if the checksum is valid.
- Gross Range Test
  - The gross range test assess whether data falls within the valid sensor minimum and maximum. The operator may also supply a minimum and maximum to flag HIGH_INTEREST or SUSPECT data.
- Climatology Test (Requires a reference dataset.)
- Spike Test
- Rate of Change Test
- Flat Line Test
- Multi-Variate Test (Requires comparison of variable A to variable B, with the assumption that they share a relationship.)
- Attenuated Signal Test
- Neighbor Test


#### Variable Specific Tests
Variable specific QARTOD tests appear in respective QARTOD manuals, in addition to the generic tests.

##### GPS
- Location Test (Requires latitude and longitude.)
    - GPS data does not have its own defined manual. Instead, the Location Test is a required test for every QARTOD manual. In some data acquisition systems, GPS is recorded separately and may have different timestamps compared to the time-series of interest. Conceptually, running the Location Test on GPS first and then syncing by timestamps would satisfy the Location Test requirement for all manuals.

##### Temperature and Salinity
- TS Curve/Space Test
- Density Inversion Test

##### Ocean Optics
- Decreasing Radiance, Irradiance, and PAR Test
- Photic Zone Limit Test


##### ADCP
- Battery Voltage Test
- Sensor Tilt Test
- Speed of Sound Test
- Noise Floor Test
- Signal Strength Test
- Signal-to-Noise Test
- Correlation Magnitude Test
- Percent Good Test
- Current Speed Test
- Current Direction Test
- Horizontal Velocity Test
- Vertical Velocity Test
- Error Velocity Test
- Echo Intensity Test
- Echo Intensity Drop-off Test
- Current Gradient Test


##### Surface Waves
- Signal Strength Test
- Correlation Magnitude Test
- Acoustic Noise Test
- Signal-to-Noise Test
- Pressure Tracking Test
- Acoustic Velocity Min/Max Test
- Acoustic Velocity Mean Teest
- Sample Count Test
- Time Series Range Test
- Segment Shift Test
- Acceleration Test
- Check Factor Test
- Mean and Standard Deviation Test
- Frequency Range Test
- Low Frequency Energy Test
- Bulk Wave Parameters Test
- Neighbor Check Test

##### HF RADAR
- Signal-to-Noise Ratio Test
- Cross Spectra Covariance Eigenvalues Test
- Direction of Arrival Magnitude Test
- Direction of Arrival Widths Test
- Positive Definiteness Test
- Max Threshold Test
- Radial Count Test
- Spatial Median Filter Test
- Temporal Gradient Test
- Average Radial Bearing Test
- Baseline and Synthetic Radial Test
- Radial Stuck Test
- Phases Test
- Data Density Test
- GDOP Threshold Test
- Max Speed Test
- Spatial Median Comparison Test
- U Uncertainty Test
- V Uncertainty Test


### Custom Data Tests
- Marine Heatwave Test
  - The marine heatwave test requires a reference dataset that contains daily climatological percentiles for the location of interest (a common method/percentile is defined by Hobday et al., 2016).
  - If the current sample and the mean of the last X samples exceeds the percentile, and the data are flagged as HIGH_INTEREST.
- Marine Cold Spell Test
  - The marine cold spell test is similar to the marine heatwave test, but with a lower percentile.
  - If the current sample and the mean of the last X samples falls below a given reference value, the data are flagged as HIGH_INTEREST.


























# File Naming Convention

{VESSEL} _ {SYSTEM} _ {LOCATION} _ {SENSOR}


## VESSEL
- Currently only SIKULIAQ

## SYSTEM
Possible options include...
- ATMOSPHERIC
- FLOWTHROUGH
- NAVIGATION
- OTHER
- ROSETTE
- SONAR


## LOCATION
Possible options include...
- WET-LAB
- TECH-STORES
- TECH-SHOP
- STBD (Starboard)
- SCI-CTRL (Science Control Room)
- PORT
- NA (Not Applicable)
- MAIN-MAST
- MAIN-LAB
- HULL
- WINCH-ROOM
- FWD-STORES
- FWD-MAST
- COMP-LAB (Computer Lab)
- CTBD (Centerboard)
- ROSETTE ROOM (CTD Winch Room)
- BOW-THRUST (Bow Thruster Room)
- BALTIC-ROOM
- AFT-DECK
- LAB-03

## SENSOR
Syntax: {vendor_name}-X
where X indicates the 'order' of the sensors. 
