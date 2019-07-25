# VADER-Analytics
Visualization and Analytics for Distributed Energy Resources

### Introduction
In its current state, the distribution system is incapable of handling small to moderate amounts of PV penetration. This is because it was initially designed for handling passive loads which, at the level of a substation, have low variability and are forecastable with high accuracy. It has been an open loop system with little monitoring and control. With the addition of PV energy sources, the overall scenario will change dramatically due to (1) two way power flow on network and (2) high aggregate variability. Additionally, changes on the consumption side lead to a number of smart loads, Electric Vehicles (EVs) and Demand Response.

These fundamental changes in the characteristics of the generation and consumption of power will lead to a number of practical engineering problems that must be overcome to allow increased penetration of Distributed PV. Solving the specific engineering challenges which come at any moderate level of PV penetration requires closed loop integration of data from (1) PV sources, (2) customer load data from smart meters, (3) EV charging data, and (4) local and line mounted precision instruments.

These data are not traditionally used by utilities in operations since they are ”non-SCADA” and the current grid does not require such level of control.  To integrate this data and provide real time intelligence from these non-SCADA data, we propose the Visualization and Analytics of Distribution Systems with Deep Penetration of Distributed Energy Resources or VADER platform. VADER is a collection of analytics enabled by integration of massive and heterogeneous data streams for granular real-time monitoring, visualization and control of Distributed Energy Resources (DER) in distribution networks. VADER analytics enable utilities to have greater visibility into distributed energy resources. We’ve built several batch- and stream-analytics in VADER that help operators better understand the impact of distributed energy resources on the grid.

Refer to [BACKGROUND.md](../master/docs/BACKGROUND.md) for more information on project background and the problems we're trying to solve.

### Getting Started

#### Installation
For step-by-step installation instructions, refer to [INSTALL-GUIDE.md](../master/docs/INSTALL-GUIDE.md). This installation guide allows users to quickly get a copy of analytics up and running on their local machine. This is ideal for users looking to test and further develop one or more analytics shipped with this repository.

### Contributing
Please read [CONTRIBUTING.md](../master/docs/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

### Versioning
We use [SemVer](https://semver.org/) for versioning.

### Authors
* Bennet Meyers - [Github](http://github.com/bmeyers)
* Davide Innocenti - [Github](http://github.com/davideinn)
* Emre Can Kara - [Github](http://github.com/eckara)
* Jiafan Yu - [Github](http://github.com/palmdr)
* Jonathan Goncalves - [Github](http://github.com/jongoncalves)
* Mayank Malik - [Github](http://github.com/malikmayank)
* Michaelangelo Tabone - [Github](http://github.com/mtabone)
* Serhan Kiliccote - [Github](http://github.com/serhank989)
* Siobhan Powell - [Github](http://github.com/siobhanpowell)

### Acknowledgements
This work is supported by U.S. Department of Energy Contract DE-AC02-76-SF00515.
