Back to [README](../README.md) > [BACKGROUND](../docs/BACKGROUND.md) >

## ML-based Power Flow - Forward

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

### Classes and Structure

There are two main class in the module is ForwardMLPF. ForwardMLPF defines an object that lets you
implement this ML model of the inverse power flow equations. Please see forward_mlpf.py for more details on the methods and attributes of this class.

Another class detailed in make_pf_data.py, GenerateDataMLPF, can be used to generate power
flow data from home load profiles (for example, profiles sampled from the Pecan Street database).
This is demonstrated in Sample_Script.ipynb. Generating the power flow data enables
the user to run and test the ML power flow algorithms without having real measurement data
for voltage or power injections in a network, but it is only a substitute validation for working
with real data.

### Sample Application Code


The notebook *Sample_Script.ipynb* shows how you can interact with the ForwardMLPF class and
gives examples of calling all the methods. The input data is four numpy arrays of shape
num_samples x num_bus, one for each the real power injection, reactive power injection,
voltage magnitude, and voltage phase angle. There are two ways you provide data to the model:
- Prepare and load your own measurement data.
- Provide home load data and use the GenerateDataMLPF class in make_pf_data.py
to generate network measurement data through pandapower with your choice of standard test
network.

Once the models are fit you can use the built-in methods to calculate the test errors on
the test set, but you can also apply the object to any new input sample that you want.
