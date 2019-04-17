Back to [README](../README.md)
Back to [BACKGROUND](../docs/BACKGROUND.md)

## Solar Disaggregation

# Problem:
A significant amount of the solar generation in California is produced by behind the meter (BTM) PV panels. At the end of June 2018 the number of non-utility BTM rooftop solar installations has reached 6,200 MW in the Californian ISO’s balancing area. The variability of solar generation combined with the unobservability of BTM installations is challenging power systems
operators on several fronts. In the case of BTM PV generation, power system operators have only access to the metered
net load at the individual customer site. In order to accurately plan how to manage their
grid assets and perform real time operations such as switching, state-estimation and voltage management power system operators
need to have access to the solar generation both historically and in real time. 

# Solution:
The proposed method aims to separate the metered netload into two distinct signals: solar generation and  load consumption at the feeder and individual customer level.
This approach uses only existing AMI measurements, solar information, such as irradiance or monitored
solar generation at a specific site in the same region, outside temperatures and hour of the day. The disaggregation is performed both
historically and real time.

## csss:
Contains the main libraries to perform the solar disaggregation and utilities
## Custom_Functions:
Similar to utilities. They should be merged.
## Exploratory_work:
Contains notebooks related to exploratory work such as batch solar and non solar houses classification and signal to noise ratio.
## Solar Disaggregation Analysis:
Contains the original notebooks left by MT
## Validation:
Contains the main notebooks and python files used to perform the extended validation.
## General:
- SolarDisagg_Individual_Home_Historical
- SolarDisagg_Individual_Home_Historical_linear

are the main notebooks used to perform the historcal disaggregation.
