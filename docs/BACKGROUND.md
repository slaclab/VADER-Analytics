Back to [README](../README.md)

### VADER Overview

In its current state, the distribution system is incapable of handling small to moderate amounts of PV penetration. This is because it was initially designed for handling passive loads which, at the level of a substation, have low variability and are forecastable with high accuracy. It has been an open loop system with little monitoring and control. With the addition of PV energy sources, the overall scenario will change dramatically due to (1) two way power flow on network and (2) high aggregate variability. Additionally, changes on the consumption side lead to a number of smart loads, Electric Vehicles (EVs) and Demand Response.

These fundamental changes in the characteristics of the generation and consumption of power will lead to a number of practical engineering problems that must be overcome to allow increased penetration of Distributed PV. Solving the specific engineering challenges which come at any moderate level of PV penetration requires closed loop integration of data from (1) PV sources, (2) customer load data from smart meters, (3) EV charging data, and (4) local and line mounted precision instruments.

These data are not traditionally used by utilities in operations since they are ”non-SCADA” and the current grid does not require such level of control.  To integrate this data and provide real time intelligence from these non-SCADA data, we propose the Visualization and Analytics of Distribution Systems with Deep Penetration of Distributed Energy Resources or VADER platform. VADER is a collection of analytics enabled by integration of massive and heterogeneous data streams for granular real-time monitoring, visualization and control of Distributed Energy Resources (DER) in distribution networks. VADER analytics enable utilities to have greater visibility into distributed energy resources. We’ve built several batch- and stream-analytics in VADER that help operators better understand the impact of distributed energy resources on the grid.

VADER ships with the following Analytics:
1. [Solar Disaggregation.md](../docs/SOLAR-DISAGGREGATION.md)
2. [ML-based Power Flow - Forward](../docs/ML-BASED-POWER-FLOW-F.md)
3. [ML-based Power Flow - Inverse](../docs/ML-BASED-POWER-FLOW-I.md)

Much of the research work done in VADER has been published. The list below is a comprehensive compilation of all journal papers, conference papers, and articles related to VADER:

* Kara et al. “Estimating Behind-the-meter Solar Generation with Existing Measurement Infrastructure (Short Paper)” , Buildsys’16 ACM International Conference on Systems for Energy-Efficient Built Environments (2016)
* Raffi Sevlian and Ram Rajagopal, "Distribution System Topology Detection Using Consumer Load and Line Flow Measurements", IEEE Transactions on Control of Network (to be submitted)
* Yizheng Liao,  Yang Weng,  and Ram Rajagopal,  “Urban Distribution Grid Topology Reconstruction via Lasso”, IEEE Power & Energy Society General Meeting, 17-21 July, 2016.
* Yizheng Liao, Yang Weng, Chin-Woo Tan, and Ram Rajagopal, “Urban Distribution Grid Line Outage Identification”, IEEE International Conference on Probabilistic Methods Applied to Power Systems,  17-20 October, 2016.
* Jiafan Yu, Junjie Qin, and Ram Rajagopal, “On Certainty Equivalence of Demand Charge Reduction Using Storage”, Proceedings of American Control Conference, Seattle, WA, 24-26 May, 2017.
* Bennet Meyers and Mark Mikofski, “Accurate Modeling of Partially Shaded PV Arrays”, Proceedings of Photovoltaic Specialists Conference (PVSC-44), Washington, DC, 25-30 June, 2017.
* Jiafan Yu, Yang Weng, and Ram Rajagopal, “Data-Driven Joint Topology and Line Parameter Estimation for Renewable Integration”, Proceedings of IEEE Power and Energy Society General Meeting, Chicago, IL, 16-20 July, 2017.
* Jiafan Yu, Yang Weng, and Ram Rajagopal, “Robust Mapping Rule Estimation for Power Flow Analysis in Distribution Grids”, North American Power Symposium, Morgantown, WV, 17-19 September, 2017.
* Yu, Jiafan, Yang Weng, and Ram Rajagopal. "Mapping Rule Estimation for Power Flow Analysis in Distribution Grids." arXiv preprint arXiv:1702.07948(2017).
* Yizheng Liao,  Yang Weng, and Ram Rajagopal,  “Distributed Energy Resources Topology Identification via Graphical modeling”, IEEE Transactions on Power Systems, 2017
* M. Malik et al. “A Common Data Architecture for Energy Data Analytics”, IEEE SmartGridComm 2018
* Jiafan Yu, Yang Weng, and Ram Rajagopal, “PaToPa: A Data-Driven Parameter and Topology Joint Estimation Framework in Distribution Grids”, IEEE Transactions on Power Systems 2018
* Bennet Meyers, Michaelangelo Tabone, and Emre Kara, “Statistical Clear Sky Fitting Algorithm”, World Conference on Photovoltaic Energy Conversion, 2018.
