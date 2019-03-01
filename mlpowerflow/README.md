# Machine Learning Power Flow

~February 23, 2019~

This package uses a Support Vector Regression with a quadratic kernel to replace 
the physical power flow equations to model the relationships between power and voltage in 
an electricity distribution network. The work is published in the proceedings of NAPS 
2017 as "Robust mapping rule estimation for power flow analysis in distribution grids" 
by Jiafan Yu, Yang Weng, and Ram Rajagopal. 

The paper is available here: 
https://www.researchgate.net/publication/321116252_Robust_mapping_rule_estimation_for_power_flow_analysis_in_distribution_grids

The original code for this work was developed by Jiafan Yu. 
This package was prepared by Siobhan Powell with invaluable help and feedback from Mayank
Malik and Jonathan Goncalves, all as part of GISMo lab at SLAC: see gismo.slac.stanford.edu 
for more information about our team and what we do. 



## Objects

There is (so far) one class in the module. ForwardMLPF defines an object that lets you 
implement this ML model of the forward power flow equations. Please see mlpf.py for more
details on the methods and attributes of this class. 


## Sample Application Code

The notebook Sample_Script.ipynb shows how you can interact with this class and 
gives examples of calling all the methods. The input data is four numpy arrays of shape
num_samples x num_bus, one for each the real power injection, reactive power injection, 
voltage magnitude, and voltage phase angle. There are two ways you provide data to the model:
- Prepare and load your own measurement data.
- Provide home load data and use the functions provided (soon to be added) 
to generate network measurement data through pandapower and your choice of standard test 
networks.

Once the models are fit you can use the built in methods to calculate the test errors on 
the test set, but you can also apply the object to any new input sample that you want. 
