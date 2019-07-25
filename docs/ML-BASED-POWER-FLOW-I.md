Back to [README](../README.md) > [BACKGROUND](../docs/BACKGROUND.md) >

## ML-based Power Flow - Inverse

#### Inverse Mapping: Power Injections or Loads to Voltage

The inverse mapping gives the voltage measurements as a function of the power injections in a 
network.

This package implements Linear Regression and/or Support Vector Regression to model the inverse
power flow problem of voltage estimation. The class was prepared with a specific use case in mind
which is outlined in Voltage_Estimation_Use_Case.ipynb, as described below. 

In VADER, working with real data from SCE we found that they do not have measurements at all of 
the interior points in their distribution network and do not have measurements of voltage phase 
angle at most of the buses in the distribution network. Without the phase angle measurements the 
forward ML power flow model can not be validated. However, we can still use this theory for voltage 
estimation; the inverse mapping model also discussed in the paper. 

The voltage estimation problem is laid out as follows: 
- Inputs: Real and reactive power injections measured at the leaf nodes (customer level data) and an aggregation point
- Outputs: Voltage magnitude at the aggregation point.

The Voltage_Estimation_Use_Case.ipynb example and InverseMLPF class in this package are built to 
address this case, as well as voltage estimation with any other data set - the same model and tools
can be used to predict voltage magnitude at all buses in the network (wherever suitable training
data exists).

### Classes and Structure

There are two main class in the module is InverseMLPF. InverseMLPF defines an object that lets you 
implement this ML model of the inverse power flow equations. Please see inverse_mlpf.py for more details on the methods and attributes of this class. 

Another class detailed in make_pf_data.py, GenerateDataMLPF, can be used to generate power
flow data from home load profiles (for example, profiles sampled from the Pecan Street database). 
This is demonstrated in Sample_Script.ipynb. Generating the power flow data enables
the user to run and test the ML power flow algorithms without having real measurement data
for voltage or power injections in a network, but it is only a substitute validation for working 
with real data. 

### Sample Application Code

The inverse mapping is used for voltage estimation in *Voltage_Estimation_Use_Case.ipynb*,
looking specifically at a particular utility setting where we try to estimate the substation level voltage 
from load data. 

It asks the following question. Given the: 

    - Load data at end points of the grid
    - Load data at the substation aggregation point
    - Voltage measurements at the substation aggregation point
    
**Can we build a model for the voltage measurements at the substation level?** 

The forward powerflow equations map voltage phase angle and magnitude to power injections 
in the network. The inverse mapping goes the other way, calculating the voltage from the power 
injections. We reduce this to calculate just the voltage magnitude from the loads in the network 
with good results, but in this use case with measurements only at the leaf nodes, will they be 
enough to build a good model?   

This is implemented and tested in the sample script, showing how to 
interact with InverseMLPF.
