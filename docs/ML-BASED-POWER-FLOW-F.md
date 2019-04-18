Back to [README](../README.md)
Back to [BACKGROUND](../docs/BACKGROUND.md)

# Machine Learning Power Flow

#### Forward Mapping: Voltage to Power Injections

The forward power flow equations map voltage magnitude and phase angle measurements to power injections
in a network.

This package uses a Support Vector Regression with a quadratic kernel to replace 
the physical power flow equations to model the relationships between power and voltage in
an electricity distribution network. The work is published in the proceedings of NAPS
2017 as "Robust mapping rule estimation for power flow analysis in distribution grids"
by Jiafan Yu, Yang Weng, and Ram Rajagopal.

The paper is available here:
https://www.researchgate.net/publication/321116252_Robust_mapping_rule_estimation_for_power_flow_analysis_in_distribution_grids

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


The inverse mapping use case for voltage estimation was prepared by Siobhan Powell.

#### Notes on the Overall Package

The original code for this work was developed by Jiafan Yu.
This package was prepared by Siobhan Powell with invaluable help and feedback from Mayank
Malik and Jonathan Goncalves, all as part of GISMo lab at SLAC: see gismo.slac.stanford.edu
for more information about our team and what we do.


## Classes and Structure

There are two classes in the module: ForwardMLPF and InverseMLPF. ForwardMLPF defines an object that lets you
implement this ML model of the forward power flow equations, and InverseMLPF likewise implements
 the reverse mapping. Please see mlpf.py for more details on the methods and attributes of this class.


## Sample Application Code

There are two sample scripts showing how to interact with the classes.  

#### Sample for ForwardMLPF

The notebook *Sample_Script.ipynb* shows how you can interact with the ForwardMLPF class and
gives examples of calling all the methods. The input data is four numpy arrays of shape
num_samples x num_bus, one for each the real power injection, reactive power injection,
voltage magnitude, and voltage phase angle. There are two ways you provide data to the model:
- Prepare and load your own measurement data.
- Provide home load data and use the functions provided (soon to be added)
to generate network measurement data through pandapower and your choice of standard test
networks.

Once the models are fit you can use the built-in methods to calculate the test errors on
the test set, but you can also apply the object to any new input sample that you want.

#### Sample for InverseMLPF

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
