import pandapower as pp
import numpy as np
import pandas as pd
import pandapower.networks
import numpy.random


class GenerateDataMLPF:
    """ Use load data to create a set of power flow measurements required by the MLPF algorithms.

    This class uses the package Pandapower to run power flow calculations. Given home load data, it builds up a network
    load profile to match a given test network's behaviour, and then returns the corresponding voltage and power
    injection measurements at all buses in the network.
    """

    def __init__(self, network_name):
        """Initialize attributes of the object."""

        self.network_name = network_name

        if network_name in ['rural_1', 'rural_2', 'village_1', 'village_2', 'suburb_1']:
            self.pp_net = pp.networks.create_synthetic_voltage_control_lv_network(network_class=network_name)
            net = self.pp_net
        else:
            self.pp_net = getattr(pp.networks, network_name)
            net = self.pp_net()

        self.net = net
        self.load_buses = np.copy(net.load.bus.values)
        self.num_load_buses = np.shape(self.load_buses)[0]
        self.num_buses = net.bus.shape[0]
        self.num_times = 0
        self.num_homes_per_bus = np.zeros((1, 1))
        self.power_factors = np.zeros((1, 1))
        self.p_set = np.zeros((1, 1))
        self.q_set = np.zeros((1, 1))
        self.p_injection = np.zeros((1, 1))
        self.q_injection = np.zeros((1, 1))
        self.v_magnitude = np.zeros((1, 1))
        self.v_angle = np.zeros((1, 1))

    def prepare_loads(self, raw_load_data, reuse=True):
        """
        Prepare the raw home load data to match the chosen test network.

        The test network is shipped with values for each of the loads, so to preserve the state of the network we try
        to match these. Going through each load bus, homes are added from the raw home data until the maximum value
        in the applied load reaches its original value.

        Parameters
        ----------
        raw_load_data: array_like
            Real power load profiles for individual homes, shape (number of time steps, number of homes)
        reuse: bool
            If the raw_load_data has many more home load profiles than will be needed to build up the load of the
            network this can be False, otherwise this allows load profiles to be reused when building the network load.

        Attributes
        ----------
        p_set, q_set: array_like
            The prepared data sets, real and reactive loads at the load buses in the network.
        num_times: int
            The number of time stamps in the data set.
        self.power_factors: array_like
            The power factor assigned to each home, used to calculate the reactive power loads.
        self.num_homes_per_bus: array_like
            Keeps track of how many home load profiles were assigned at each load bus in the network.
        """

        net = self.net

        p_ref = np.copy(net.load['p_kw'].values)

        self.num_times = np.shape(raw_load_data)[0]
        num_homes = np.shape(raw_load_data)[1]

        p_set = np.zeros((self.num_times, self.num_load_buses))
        q_set = np.zeros((self.num_times, self.num_load_buses))

        self.power_factors = np.clip(0.9 + 0.05 * np.random.randn(self.num_load_buses), 0.0, 1.0)

        num_homes_per_bus = np.zeros((self.num_load_buses, 1))

        bus_set = np.arange(0, num_homes)
        for j in range(self.num_load_buses):
            while np.max(p_set[:, j]) < p_ref[j]:

                which_house = np.random.choice(bus_set)
                if not reuse:
                    bus_set.remove(which_house)
                p_set[:, j] += raw_load_data[:, which_house]
                num_homes_per_bus[j] += 1

            s_here_power = np.power(p_set[:, j] / self.power_factors[j], 2)
            p_here_power = np.power(p_set[:, j], 2).reshape(np.shape(s_here_power))
            q_set[:, j] = np.sqrt(s_here_power - p_here_power)

        self.p_set = np.copy(p_set)
        self.q_set = np.copy(q_set)
        self.num_homes_per_bus = num_homes_per_bus

    def evaluate_all_powerflows(self, display_counts=False):
        """
        For every time step in the data set, run the power flow on the network and capture the results.

        Parameters
        ----------
        display_counts: bool
            Option to add print statements giving updates on the progress working through the data set.

        Attributes
        ----------
        p_injection, q_injection, v_magnitude, v_angle: array_like
            The results of the power flow for each bus: real and reactive power injection, voltage magnitude and
            phase angle.
        """

        p_injection = np.zeros((self.num_times, self.num_buses))
        q_injection = np.zeros((self.num_times, self.num_buses))
        v_magnitude = np.zeros((self.num_times, self.num_buses))
        v_angle = np.zeros((self.num_times, self.num_buses))

        for t in range(self.num_times):
            p_out, q_out, vm_out, va_out = self.run_pf(self.p_set[t, :], self.q_set[t, :])

            p_injection[t, :] = np.copy(p_out)
            q_injection[t, :] = np.copy(q_out)
            v_magnitude[t, :] = np.copy(vm_out)
            v_angle[t, :] = np.copy(va_out)

            if display_counts:
                if np.mod(t, 50) == 0:
                    print('Done power flow calculation for time step ', t)

        self.p_injection = p_injection
        self.q_injection = q_injection
        self.v_magnitude = v_magnitude
        self.v_angle = v_angle

    def run_pf(self, p_load, q_load):
        """
        Run power flow in the network for one instant in time with these loads.

        To do this we first apply the loads, p_load and q_load, to the net.load dataframe, then we use the function
        pandapower.runpp to execute the power flow. The resulting power injections and voltage measurements at all of
        the network buses (not just the load buses) are extracted from net.res_bus and returned.

        Parameters
        ----------
        p_load, q_load: array_like
            The real and reactive power values of the loads in the network.

        Returns
        ----------
        p_inj, q_inj, vm, va: array_like
            The outputs of the power flow simulation: real and reactive power injection, voltage magnitude and phase
            angle. These arrays contain the values for each bus in the network.
        """

        net = self.net

        # Assign loads
        df = net.load
        for i in range(np.shape(self.load_buses)[0]):
            df.loc[lambda df: df['bus'] == self.load_buses[i], 'p_kw'] = p_load[i]
            df.loc[lambda df: df['bus'] == self.load_buses[i], 'q_kvar'] = q_load[i]
        net.load = df

        pp.runpp(net)

        p_inj = np.copy(net.res_bus.p_kw.values)
        q_inj = np.copy(net.res_bus.q_kvar.values)
        vm = np.copy(net.res_bus.vm_pu.values)
        va = np.copy(net.res_bus.va_degree.values)

        return p_inj, q_inj, vm, va
