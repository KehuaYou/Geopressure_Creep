"""
geopressure_creep.py
-----------------------------------
This module contains objects and methods to initialize, control, execute,
and plot simulations of pore pressure evolution during sedimentation
and compaction. The model includes the effect of creep and smectite-illite
transformation.

Model detail is described in the following reference:
You K., P. Flemings, A.R. Bhandari, M. Heidari and J. Germaine (2022), The role
of creep in geopressure development, Petroleum Geoscience, 28(3), petgeo2021-064
https://doi.org/10.1144/petgeo2021-064
-----------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
import copy

# constants, acceleration due to gravity
GRAVITY = 9.8


class Lithology(object):
    """
    Describe the general rock properties

    Attributes
    -----------------------------------
    bulk_density: float
        rock bulk density in kg/m3
    grain_density: float
        Grain density in kg/m3
    cp: float
        Compressibility of grain due to pressure change
    ct: float
        Compressibility of grain due to temperature change
    heat_conductivity: float
        Solid grain heat conductivity
    void_reference_sm: float
        Void ratio at reference point for smectite
    stress_reference_sm: float
        Effective  stress at reference point for smectite
    cc_sm: float
        Primary compression index for smectite
    ca_sm: float
        Secondary compression index for smectite
    void_reference_il: float
        Void ratio at reference point for illite
    stress_reference_il: float
        Effective  stress at reference point for illite
    cc_il: float
        Primary compression index for illite
    ca_il: float
        Secondary compression index for illite
    to: float
        the parameter to in creep model
    high_perm_a: float
        Parameter in high perm model k = 10^(high_perm_a * porosity + high_perm_b)
    high_perm_b: float
        Parameter in high perm model k = 10^(high_perm_a * porosity + high_perm_b)
    low_perm_a:
        Parameter in low perm model k = 10^(low_perm_a * porosity + low_perm_b)
    low_perm_b:
        Parameter in low perm model k = 10^(low_perm_a * porosity + low_perm_b)
    A_trans:
        Parameter in kinetics model of smectite to illite transformation
    D_trans:
        Parameter in kinetics model of smectite to illite transformation
    -----------------------------------
    """

    def __init__(self, ):
        super(Lithology, self).__init__()
        # Get the default lithology properties
        self._default_values()

    def _default_values(self):
        """
        Set the default lithology parameters
        """
        self.bulk_density = 2000
        self.grain_density = 2750
        self.cp = 0  # compressibility of solid grain due to pressure change
        self.ct = 0  # compressibility of solid grain due to temperature change
        self.heat_conductivity = 1.6  # solid grain heat conductivity

        # properties of Smectite
        self.void_reference_sm = 0.89  # void ratio at reference point
        self.stress_reference_sm = 1.7e6  # effective stress at reference point
        self.cc_sm = 0.5475  # primary compression index
        self.ca_sm = self.cc_sm * 0.0187  # secondary compression index

        # properties of Illite
        self.void_reference_il = 0.2  # void ratio at reference point
        self.stress_reference_il = 19.7e6  # effective stress at reference point
        self.cc_il = 0.5475  # primary compression index
        self.ca_il = self.cc_sm * 0.0187  # secondary compression index

        self.to = 85 * 60  # parameter to in creep model

        # parameters in permeability model
        self.high_perm_a = 10.6
        self.high_perm_b = -19.6
        self.low_perm_a = 10.6
        self.low_perm_b = -22.6

        # parameters in the kinetics model of smectite to illite transformation
        self.A_trans = 3.86e9
        self.D_trans = 19096


class Fluid(object):
    """
    Describe the fluid or water properties

    Attributes
    --------------------------------------
    density_reference: float
        Water density at reference pressure and temperature
    viscosity: float
        Water viscosity
    cfp: float
        Water compressibility due to pressure change
    cft: float
        Water compressibility due to temperature change
    heat_conductivity: float
        Water heat conductivity
    ----------------------------------------
    """

    def __init__(self, ):
        super(Fluid, self).__init__()
        # Get the default water properties
        self._default_values()

    def _default_values(self):
        """
        Set the default water properties
        """
        self.density_reference = 1025  # water density at reference pressure and temperature
        self.viscosity = 1.31e-3  # water viscosity
        self.cfp = 5e-10  # water compressibility due to pressure change
        self.cft = 0  # water compressibility due to temperature change
        self.heat_conductivity = 0.58  # water heat conductivity


class ModelGrid(object):
    """
    Describe the model grid

    Attributes:
    ----------------------------------------
    seafloor: float
        Seafloor depth in meters
    model_top: float
        depth of the model top below the seafloor in meters
    row: float
        Maximum number of grids you want to use in simulation
    dx_ini: float
        Vertical size of each grid when the grid cell is at the depth of model_top
    rocks: Lithology
        A Lithology object that contains base rock properties
    fluids: Fluid
        A Fluid object that contains default water properties
    sedimentation_rate: float
        Sedimentation rate at the seafloor
    age_top: float
        Age of sediment at the top of the modeling domain
    perm_indexes: list
        A list with sie row, indicating the layer with high permeability model with value = 1 and
        low permeability layer with value = 0
    frac_s: float
        Fraction of smectite when sediment is deposited at the seafloor
    T_surf: float
        Temperature in oC at seafloor
    T_grad: float
        Temperature gradient in oC/km
    q_heat: float
        Geothermal heat flux in W/m2
    creep: Boolean
        A index to indicate if creep is included or not in the simulation
    si_transfer: Boolean
        A index to indicate if smectite illite transfer is included in the simulation or not
    ----------------------------------------
    """

    def __init__(self, seafloor, model_top, row, dx_ini, rocks, fluids, sedimentation_rate, age_top,
                 perm_indexes, frac_sm, t_surf, t_grad, q_heat, creep, si_transfer):
        super(ModelGrid, self).__init__()

        # save the base properties
        self.seafloor = seafloor
        self.model_top = model_top
        self.row = row
        self.dx_ini = dx_ini
        self.rock = rocks
        self.fluid = fluids
        self.sedimentation_rate = sedimentation_rate
        self.age_top = age_top
        self.perm_index = perm_indexes
        self.frac_sm_ini = frac_sm
        self.t_surf = t_surf
        self.t_grad = t_grad
        self.q_heat = q_heat
        self.creep = creep
        self.si_transfer = si_transfer

        # build the initial grid cells
        self._build_grid()
        # set the initial fluid properties
        self._set_fluid_properties()
        # set the initial matrix properties
        self._set_rock_properties()

    def _build_grid(self):
        self.top_row = self.row - 1  # There are 2 grid cells in the simulation domain initially
        self.dx = self.dx_ini * np.ones((self.row,))
        temp_depth = [self.dx_ini * i for i in range(self.row)]
        self.depth = np.array(temp_depth) - temp_depth[-2] + self.model_top

    def _set_fluid_properties(self):
        self.temperature = np.maximum(self.t_surf + self.t_grad * self.depth,
                                      self.t_surf + self.t_grad * self.model_top)
        self.fluid_density = self.fluid.density_reference * np.ones((self.row,))
        self.pw = np.maximum(self.fluid.density_reference * GRAVITY * (self.depth + self.seafloor),
                             self.fluid.density_reference * GRAVITY * (self.model_top + self.seafloor))

    def _set_rock_properties(self):
        self.total_stress = np.maximum(self.fluid.density_reference * GRAVITY * self.seafloor
                                       + self.rock.bulk_density * GRAVITY * self.depth,
                                       self.fluid.density_reference * GRAVITY * self.seafloor
                                       + self.rock.bulk_density * GRAVITY * self.model_top)
        self.eff_stress = self.total_stress - self.pw
        self.frac_sm = self.frac_sm_ini * np.ones((self.row,))
        self.cc = self.frac_sm * self.rock.cc_sm + (1 - self.frac_sm) * self.rock.cc_il
        self.cr = self.cc / 6
        self.ca = self.frac_sm * self.rock.ca_sm + (1 - self.frac_sm) * self.rock.ca_il
        self.void_ratio_sm = self.rock.void_reference_sm - self.rock.cc_sm * np.log10(
            self.eff_stress / self.rock.stress_reference_sm)
        self.void_ratio_il = self.rock.void_reference_il - self.rock.cc_il * np.log10(
            self.eff_stress / self.rock.stress_reference_il)
        self.void_ratio = self.frac_sm * self.void_ratio_sm + (1 - self.frac_sm) * self.void_ratio_il
        for i in range(self.row):
            if np.isnan(self.void_ratio[i]):
                self.void_ratio[i] = 2.67
                self.eff_stress[i] = 0.0009555 * 1e6
        self.porosity = self.void_ratio / (1 + self.void_ratio)
        self.porosity_ini = self.porosity[0]
        self.te = np.zeros((self.row,))
        self.permeability = (
                np.power(10, self.rock.high_perm_a * self.porosity + self.rock.high_perm_b) * self.perm_index
                + np.power(10, self.rock.low_perm_a * self.porosity + self.rock.low_perm_b) * (1 - self.perm_index))
        self.age = self.age_top * np.ones((self.row,))
        self.age[-1] += (self.depth[-1] - self.depth[-2]) / self.sedimentation_rate

    def simulate(self, out_dir='../output/', base_name='creep_model', dt=86400 * 365 * 100, num_total=20000,
                 num_save=500, num_show=500, max_iter=15, t_max=86400 * 365 * 10e6):
        """
        We conduct the simulation to predict the sediment deposition from the model_top and the following compaction,
        pore pressure evolution here. We use the newton_raphson method to solve the nonlinear mass conversation equation
        for pore water

        :param out_dir: the directory where the outputs are stored
        :param base_name: the name of the model
        :param dt: time step in seconds
        :param num_total: maximum number of timestep we solve
        :param num_save: we save the results for every num_save timestep
        :param num_show: we print the results for every num_show timestep
        :param max_iter: the maximum number of iteration every timestep to solve the nonlinear equation
        :param t_max: the maximum time we solve in seconds
        :return: t_out - the time when we store our results
                top_row_out - the row number at the model_top, simulation is conducted on grids between top_row and row
                data_out - a list where we store the predicted results, each row is a  depth, each column is a
                            predicted result, the third dimension is a time
                            (depth, temperature, total stress, pore pressure, effective stress, void ratio, porosity,
                            smectite fraction, sediment age, equivalent time, fluid density, sediment permeability)

        """

        timestep = 0
        time = [0]
        self._copy_result()

        header, data = self.extract_solution()
        t_out = []
        data_out = []
        top_row_out = []
        t_out.append(time[-1])
        data_out.append(data)
        top_row_out.append(self.top_row)

        while timestep < num_total:
            self.dt = dt
            time.append(time[-1] + self.dt)
            timestep += 1
            print('Timestep ={} out of total {}'.format(timestep, num_total))
            self._copy_result()

            t_flag = False
            loop = True
            iteration = 0
            while loop:
                if iteration == 0:
                    self._update_grid()
                    self._update_boundary_condition()
                    self._update_permeability()
                    self._update_sediment_age()
                    if self.si_transfer:
                        self._sm_il_trans()
                        self._update_rock_compression_property()

                self._newton_raphson()
                for i in range(self.top_row, self.row):
                    self._ppt_update(i)

                iteration += 1
                if iteration > max_iter and not t_flag:
                    loop = False

                for i in range(self.top_row, self.row):
                    if self.pw[i] < 0 or self.eff_stress[i] < 0 or np.isnan(self.pw[i]):
                        t_flag = True
                if t_flag:
                    t_flag = False
                    loop = True
                    iteration = 0
                    time[-1] = time[-1] - self.dt
                    self.dt *= 0.1
                    time[-1] = time[-1] + self.dt
                    self._go_to_prev_time()

            self._update_permeability()
            if np.remainder(timestep, num_show) == 0:
                print('Present model solution:  ')
                self.print_data(row_start=self.top_row, row_end=self.row)
            if np.remainder(timestep, num_save) == 0:
                file_name = out_dir + base_name + '_%6.6f.dat' % (time[-1] / (86400 * 365 * 1e6))
                self.save_data(f_name=file_name, t=time[-1], top_row=self.top_row)
                header, data = self.extract_solution()
                t_out.append(time[-1])
                data_out.append(data)
                top_row_out.append(self.top_row)
            if time[-1] >= t_max:
                # Reached maximum desired time
                timestep = num_total
                print('Model reached desired maximum time (%g m.y.):' % (time[-1] / (86400 * 365 * 1e6)))

        header, data = self.extract_solution()
        t_out.append(time[-1])
        data_out.append(data)
        top_row_out.append(self.top_row)
        return t_out, top_row_out, data_out

    def _copy_result(self):
        self.total_stress_old = copy.deepcopy(self.total_stress)
        self.eff_stress_old = copy.deepcopy(self.eff_stress)
        self.pw_old = copy.deepcopy(self.pw)
        self.temperature_old = copy.deepcopy(self.temperature)
        self.fluid_density_old = copy.deepcopy(self.fluid_density)
        self.void_ratio_old = copy.deepcopy(self.void_ratio)
        self.void_ratio_sm_old = copy.deepcopy(self.void_ratio_sm)
        self.void_ratio_il_old = copy.deepcopy(self.void_ratio_il)
        self.porosity_old = copy.deepcopy(self.porosity)
        self.te_old = copy.deepcopy(self.te)
        self.depth_old = copy.deepcopy(self.depth)
        self.top_row_old = copy.deepcopy(self.top_row)
        self.age_old = copy.deepcopy(self.age)
        self.frac_sm_old = copy.deepcopy(self.frac_sm)

    def _go_to_prev_time(self):
        import copy
        self.total_stress = copy.deepcopy(self.total_stress_old)
        self.eff_stress = copy.deepcopy(self.eff_stress_old)
        self.pw = copy.deepcopy(self.pw_old)
        self.temperature = copy.deepcopy(self.temperature_old)
        self.fluid_density = copy.deepcopy(self.fluid_density_old)
        self.void_ratio = copy.deepcopy(self.void_ratio_old)
        self.void_ratio_sm = copy.deepcopy(self.void_ratio_sm_old)
        self.void_ratio_il = copy.deepcopy(self.void_ratio_il_old)
        self.porosity = copy.deepcopy(self.porosity_old)
        self.te = copy.deepcopy(self.te_old)
        self.depth = copy.deepcopy(self.depth_old)
        self.top_row = copy.deepcopy(self.top_row_old)
        self.age = copy.deepcopy(self.age_old)
        self.frac_sm = copy.deepcopy(self.frac_sm_old)

    def _update_grid(self):
        # deposit a new grid cell from the model_top
        if self.top_row > 1:
            if self.depth[self.top_row - 1] > 300:
                self.top_row -= 1
        # update dx for cells above the model top
        self.dx[0:(self.top_row - 1)] = self.dx_ini
        # update dx for cells below the model top
        self.dx[(self.top_row - 1):self.row] = self.dx_ini * (1 - self.porosity_ini) / (1 - self.porosity[(self.top_row
                                                                                                           - 1):self.row])
        # the sediment moving velocity for top cell below model top
        vs = self.sedimentation_rate * (1 - self.porosity_ini) / (1 - self.porosity[self.top_row - 1])
        # update the depth for top cell below model top
        self.depth[self.top_row - 1] += vs * self.dt
        # update the depth for cells below the model top
        self.depth[self.top_row:self.row] = self.depth[(self.top_row - 1):(self.row - 1)] + 0.5 * (
                self.dx[(self.top_row - 1):(self.row - 1)] + self.dx[self.top_row:self.row])
        # update the depth for cells above the model top
        for i in range(self.top_row - 1):
            self.depth[self.top_row - 2 - i] = (self.depth[self.top_row - 1 - i]
                                                - 0.5 * (self.dx[self.top_row - 2 - i] + self.dx[self.top_row - 1 - i]))

    def _update_boundary_condition(self):
        self.total_stress[self.top_row - 1] = (self.fluid.density_reference * GRAVITY * self.seafloor
                                               + self.rock.bulk_density * GRAVITY * self.depth[self.top_row - 1])
        for i in range(self.top_row, self.row):
            self.total_stress[i] = self.total_stress[i - 1] + (
                    (1 - self.porosity[i - 1]) * self.rock.grain_density + self.porosity[
                i - 1] * self.fluid_density[i - 1]) * GRAVITY * self.dx[i - 1] * 0.5 + (
                                           (1 - self.porosity[i]) * self.rock.grain_density + self.porosity[
                                       i] * self.fluid_density[i]) * GRAVITY * self.dx[i] * 0.5

        self.pw[self.top_row - 1] = self.fluid.density_reference * GRAVITY * (self.depth[self.top_row - 1]
                                                                              + self.seafloor)
        # Update temperature profile in the entire simulated domain, equilibrium with geothermal heat supply
        self.temperature[self.top_row - 1] = self.t_surf + self.t_grad * self.depth[self.top_row - 1]
        for i in range(self.top_row, self.row):
            self.temperature[i] = (self.temperature[i - 1]
                                   + self.q_heat / ((1 - self.porosity[i]) * self.rock.heat_conductivity
                                                    + self.porosity[i] * self.fluid.heat_conductivity)
                                   * (self.depth[i] - self.depth[i - 1]))

    def _update_permeability(self):
        self.permeability = (
                np.power(10, self.rock.high_perm_a * self.porosity + self.rock.high_perm_b) * self.perm_index
                + np.power(10, self.rock.low_perm_a * self.porosity + self.rock.low_perm_b) * (1 - self.perm_index))

    def _sm_il_trans(self):
        self.frac_sm[self.top_row - 1:self.row] = (self.frac_sm[self.top_row - 1:self.row]
                                                   - self.rock.A_trans * np.exp(
                    -self.rock.D_trans / (self.temperature[self.top_row - 1:self.row] + 273.15)) * self.frac_sm[
                                                                                                   self.top_row - 1:self.row] ** 5 * self.dt)

    def _update_rock_compression_property(self):
        self.cc = self.frac_sm * self.rock.cc_sm + (1 - self.frac_sm) * self.rock.cc_il
        self.cr = self.cc / 6
        self.ca = self.frac_sm * self.rock.ca_sm + (1 - self.frac_sm) * self.rock.ca_il

    def _update_sediment_age(self):
        self.age = self.age_old + self.dt * (self.depth > self.model_top)

    def _newton_raphson(self):
        eps_pw = 1
        temp = self.row - self.top_row
        residual_matrix = np.zeros([temp, 1])
        jacobi_matrix = np.zeros([temp, temp])

        for ii in range(self.top_row, self.row):
            ind = ii - self.top_row
            residual_0 = self._calc_residual(ii)
            residual_matrix[ind] = -residual_0

            if ii > self.top_row:
                jacobi_matrix[ind, ind - 1] = self._calc_derivative(cal_i=ii, i=ii - 1, eps=eps_pw,
                                                                    residual0=residual_0)
            jacobi_matrix[ind, ind] = self._calc_derivative(cal_i=ii, i=ii, eps=eps_pw, residual0=residual_0)
            if ii < self.row - 1:
                jacobi_matrix[ind, ind + 1] = self._calc_derivative(cal_i=ii, i=ii + 1, eps=eps_pw,
                                                                    residual0=residual_0)

        delta = solve(jacobi_matrix, residual_matrix)
        delta = np.real(delta)
        self.pw[self.top_row:self.row] += delta.flatten()

    def _calc_derivative(self, cal_i, i, eps, residual0):
        """
        Calculate the derivative of the residual function to pore pressure
        :param cal_i: residual at grid cell cal_i
        :param i: the derivative to the pore pressure at grid cell i
        :param eps: small change in pore pressure for derivative calculation
        :param residual0: residual at grid cell cal_i at current pore pressure
        :return: jac - the derivative of residual at grid cell cal_i to the pore pressure at grid cell i
        """
        self.pw[i] = self.pw[i] + eps
        self._ppt_update(i)
        residual = self._calc_residual(cal_i)
        jac = (residual - residual0) / eps
        self.pw[i] = self.pw[i] - eps
        self._ppt_update(i)
        return jac

    def _calc_residual(self, i):
        ss = 1 / (1 + self.void_ratio[i]) * (self.cc[i] / np.log(10) / self.eff_stress[i] + self.void_ratio[i]
                                             * (self.fluid.cfp - self.rock.cp))
        sn = 1 / (1 + self.void_ratio[i] * (self.fluid.cfp - self.rock.cp) / (self.cc[i] / np.log(10))
                  * self.eff_stress[i])
        storage_term = (self.pw[i] - self.pw_old[i]) / self.dt
        loading_term = sn * (self.total_stress[i] - self.total_stress_old[i]) / self.dt
        temperature_term = 0
        creep_term = (1 / (1 + self.void_ratio[i]) * (self.ca[i] / np.log(10)) / (self.rock.to + self.te[i]) / ss
                      * (self.te[i] - self.te_old[i]) / self.dt)
        transfer_term = (- (self.void_ratio_sm_old[i] - self.void_ratio_il_old[i]) / ((1 + self.void_ratio[i]) * ss)
                         * (self.frac_sm[i] - self.frac_sm_old[i]) / self.dt)

        head_middle = self.pw[i] - self.fluid_density[i] * GRAVITY * self.depth[i]
        if i == 1:
            head_lower = self.pw[i + 1] - self.fluid_density[i + 1] * GRAVITY * self.depth[i + 1]
            perm_lower = (2 * (self.permeability[i] / self.dx[i]) * (self.permeability[i + 1] / self.dx[i + 1])
                          / (self.permeability[i] / self.dx[i] + self.permeability[i + 1] / self.dx[i + 1]))
            dnw_lower = ((head_lower > head_middle) * self.fluid_density[i + 1]
                         + (head_lower <= head_middle) * self.fluid_density[i])

            head_upper = self.pw[i] - self.fluid_density[i] * GRAVITY * self.depth[i]
            perm_upper = 0
            dnw_upper = self.fluid_density[i]
        elif i == self.row - 1:
            head_upper = self.pw[i - 1] - self.fluid_density[i - 1] * GRAVITY * self.depth[i - 1]
            perm_upper = (2 * (self.permeability[i - 1] / self.dx[i - 1]) * (self.permeability[i] / self.dx[i])
                          / (self.permeability[i] / self.dx[i] + self.permeability[i - 1] / self.dx[i - 1]))
            dnw_upper = ((head_upper > head_middle) * self.fluid_density[i - 1]
                         + (head_upper <= head_middle) * self.fluid_density[i])

            head_lower = self.pw[i] - self.fluid_density[i] * GRAVITY * self.depth[i]
            perm_lower = 0
            dnw_lower = self.fluid_density[i]
        else:
            head_upper = self.pw[i - 1] - self.fluid_density[i - 1] * GRAVITY * self.depth[i - 1]
            perm_upper = (2 * (self.permeability[i - 1] / self.dx[i - 1]) * (self.permeability[i] / self.dx[i])
                          / (self.permeability[i] / self.dx[i] + self.permeability[i - 1] / self.dx[i - 1]))
            dnw_upper = ((head_upper > head_middle) * self.fluid_density[i - 1]
                         + (head_upper <= head_middle) * self.fluid_density[i])

            head_lower = self.pw[i + 1] - self.fluid_density[i + 1] * GRAVITY * self.depth[i + 1]
            perm_lower = (2 * (self.permeability[i] / self.dx[i]) * (self.permeability[i + 1] / self.dx[i + 1])
                          / (self.permeability[i] / self.dx[i] + self.permeability[i + 1] / self.dx[i + 1]))
            dnw_lower = ((head_lower > head_middle) * self.fluid_density[i + 1]
                         + (head_lower <= head_middle) * self.fluid_density[i])

        flow_term = 1 / (self.fluid_density[i] * ss) * (perm_lower * (head_lower - head_middle) * dnw_lower
                                                        - perm_upper * (head_middle - head_upper) * dnw_upper) / \
                    self.dx[i]
        residual = storage_term - loading_term - temperature_term - creep_term * self.creep - flow_term \
                   - transfer_term * self.si_transfer
        return residual

    def _ppt_update(self, i):
        self.eff_stress[i] = self.total_stress[i] - self.pw[i]
        self.fluid_density[i] = (self.fluid_density_old[i]
                                 + self.fluid.cfp * self.fluid_density_old[i] * (self.pw[i] - self.pw_old[i])
                                 - self.fluid.cft * self.fluid_density_old[i]
                                 * (self.temperature[i] - self.temperature_old[i]))
        if self.creep:
            te_temp = self.dt - self.rock.to + (self.te_old[i] + self.rock.to) * np.power(
                self.eff_stress[i] / self.eff_stress_old[i], -(self.cc[i] - self.cr[i]) / self.ca[i])
            self.te[i] = np.minimum(te_temp, self.age[i])
        else:
            self.te[i] = 0
        self.void_ratio_sm[i] = (self.rock.void_reference_sm
                                 - self.rock.cc_sm * np.log10(self.eff_stress[i] / self.rock.stress_reference_sm)
                                 - self.rock.ca_sm * np.log10((self.te[i] + self.rock.to) / self.rock.to))
        self.void_ratio_il[i] = (self.rock.void_reference_il
                                 - self.rock.cc_il * np.log10(self.eff_stress[i] / self.rock.stress_reference_il)
                                 - self.rock.ca_il * np.log10((self.te[i] + self.rock.to) / self.rock.to))
        self.void_ratio[i] = self.frac_sm[i] * self.void_ratio_sm[i] + (1 - self.frac_sm[i]) * self.void_ratio_il[i]
        self.porosity[i] = self.void_ratio[i] / (1 + self.void_ratio[i])

    def print_data(self, row_start, row_end):
        """
                Print to the screen a section of data from the present solution

                Print selected variables to the screen in the form as a table from the
                row row_start to the row row_end

        """
        # Make sure there are enough rows of data to print
        row_start = max(row_start, 1)
        row_end = min(self.row, row_end)
        num_rows = row_end - row_start + 1

        # Create an output array
        header = '    depth (km) | temperature (deg C) | total_stress (MPa) | '
        header += 'eff_stress (MPa) | pore_pressure (MPa) | porosity (--) | void_ratio (--) | '
        header += ' sed_age (m.y.)  | equi_time (m.y.) | smec_frac (%) \n'
        header += '    ---------- | ------------------- | ------------------ | '
        header += '---------------- | ------------------- | ------------- | --------------- | '
        header += '--------------- | ----------------- | --------------- '

        out_data = []
        out_rows = np.floor(np.linspace(row_start - 1, row_end - 1, num=num_rows,
                                        endpoint=True))

        for i in out_rows:
            i = int(i)
            out_data.append('%11.3g %16.4g %18.4g ' %
                            (self.depth[i] / 1000, self.temperature[i], self.total_stress[i] / 1e6))
            out_data.append('%19.4g %21.4g %20.3g %17.3g  ' %
                            (self.eff_stress[i] / 1e6, self.pw[i] / 1e6, self.porosity[i], self.void_ratio[i]))
            out_data.append('%17.4g %17.4g % 16.4g \n' %
                            (
                                self.age[i] / (86400 * 365 * 1e6), self.te[i] / (86400 * 365 * 1e6),
                                self.frac_sm[i] * 100))
        print(''.join(header))
        print(''.join(out_data))

    def extract_solution(self):
        """
        Create a matrix of data from the present solution for saving

        Extract a matrix of data with depth on the row-axis in the first
        column and containing the present model solution.  These are the
        data that would normally be saved to preserve a model solution.

        Returns
        -------
        header : str
            A string containing the header information
        data : list
            A two-dimensional list of data

        """
        header = 'Geo-pressure model with creep and smectite-illite transformation\n\n'
        header += 'Data are organized in columns, with each column defined as follows:\n\n'
        out_names = ['depth below seafloor (km)',
                     'temperature (deg C)',
                     'total stress (MPa)'
                     'pore water pressure (Mpa)',
                     'effective stress (MPa)',
                     'void ratio (--)',
                     'porosity (--)',
                     'smectite fraction (--)',
                     'sediment age (m.y.)',
                     'equivalent time  (m.y.)',
                     'fluid density (kg/m3)',
                     'sediment permeability (m2)']
        for i in range(len(out_names)):
            header += '    Col %3.3d = %s\n' % (i, out_names[i])

        data = []
        for i in range(self.row):
            data.append([self.depth[i], self.temperature[i], self.total_stress[i], self.pw[i],
                         self.eff_stress[i], self.void_ratio[i], self.porosity[i], self.frac_sm[i],
                         self.age[i], self.te[i], self.fluid_density[i], self.permeability[i]])

        return header, data

    def save_data(self, f_name, t=-999, top_row=-999):
        """
                Save the present model solution

                Use the `data_matrix` method to obtain a snapshot of the data and
                the header information, then pass this to `np.savetxt()` to save
                the data.  Note, this will only be the present model solution, not
                a time series of solution data.

                Parameters
                ----------
                fname : str
                    File name to save the data file
                t : float, default=-999
                    Time associated with the present solution.  This data is appended
                    to the header data.  The -999 flag means that the data were saved
                    independent of a model simulation and are not associated with any
                    particular time step (e.g., when testing input files for
                    reasonable initial conditions.)
                top_row: the row number at the top of the model domain
                """
        # Get the present data
        header, data = self.extract_solution()
        # Add the time information to the header
        header += '\nData correspond to model simulation time %g (m.y.)' % (t / (86400 * 365 * 1e6))
        header += '\nTop row number is %4.4d ' % top_row
        # Format the data as an array
        data = np.array(data)
        # Save the data to the given file
        np.savetxt(f_name, data, header=header)


if __name__ == '__main__':
    maximum_row_num = 1000  # maximum number of grid cell
    seafloor_simulate = 1500    # seafloor depth or water depth
    model_top_depth = 300       # depth of model top below the seafloor
    initial_smectite_fraction = 1   # initial fraction of smectite in sediments
    perm_index = np.zeros((maximum_row_num,))     # permeability index, 1 = high perm, 0 = low perm
    rock = Lithology()
    fluid = Fluid()

    # Create a one-dimensional simulation domain
    grid = ModelGrid(seafloor=seafloor_simulate, model_top=model_top_depth, row=maximum_row_num, dx_ini=25,
                     rocks=rock, fluids=fluid,
                     sedimentation_rate=1e-3 / (86400 * 365), age_top=model_top_depth / (1e-3 / (86400 * 365)),
                     perm_indexes=perm_index, frac_sm=initial_smectite_fraction, t_surf=4, t_grad=0.024, q_heat=0.03,
                     creep=True, si_transfer=True)

    # Print the initial condition to the screen
    grid.print_data(row_start=1, row_end=10)

    # Run the simulation, and save the results to "results", output time to "time" and top row number to "top_row_list"
    time, top_row_list, results = grid.simulate(out_dir='../output/', base_name='creep_model',
                                                dt=86400 * 365 * 100, num_total=100, num_save=50, num_show=50,
                                                max_iter=15, t_max=86400 * 365 * 10e6)

    results = np.array(results)
    # Extract the solution at final timestep
    start = top_row_list[-1] - 1
    depth = results[-1, start:, 0]
    total_stress = results[-1, start:, 2]
    pressure = results[-1, start:, 3]
    porosity = results[-1, start:, 6]
    # Calculate some baselines
    hydro_pressure = grid.fluid.density_reference * GRAVITY * (depth + seafloor_simulate)
    reduced_total_stress = total_stress - hydro_pressure
    over_pressure = pressure - hydro_pressure
    void_ratio_sm = rock.void_reference_sm - rock.cc_sm * np.log10(reduced_total_stress / rock.stress_reference_sm)
    void_ratio_il = rock.void_reference_il - rock.cc_il * np.log10(reduced_total_stress / rock.stress_reference_il)
    void_ratio_normal = initial_smectite_fraction * void_ratio_sm + (1 - initial_smectite_fraction) * void_ratio_il
    porosity_normal = void_ratio_normal / (1 + void_ratio_normal)

    # Plot the final results
    SMALL_SIZE = 12
    LARGE_SIZE = 16
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    plt.xticks(fontsize=SMALL_SIZE)
    plt.yticks(fontsize=SMALL_SIZE)

    ax[0].plot(reduced_total_stress / 1e6, depth, color='k', linewidth=2, label='total stress')
    ax[0].plot(over_pressure / 1e6, depth, color='b', linewidth=2, label='overpressure')
    ax[0].set_xlabel('Pressure, MPa', fontsize=SMALL_SIZE)
    ax[0].set_ylabel('Depth below seafloor, m', fontsize=SMALL_SIZE)
    ax[0].set_ylim(depth[-1], 300)
    ax[0].legend(loc='best')
    ti = str(time[-1] / (86400 * 365 * 1e3))
    output_time = 'Time = ' + ti + ' kyrs'
    ax[0].set_title(output_time, fontsize=LARGE_SIZE)

    ax[1].plot(porosity, depth, color='red', lw=2, label='porosity')
    ax[1].plot(porosity_normal, depth, color='green', lw=2, linestyle='dashed', label='porosity @ normal compaction')
    ax[1].set_xlim(left=0.2, right=0.6)
    ax[1].set_xlabel('Porosity', fontsize=SMALL_SIZE)
    ax[1].set_ylim(depth[-1], 300)
    ax[1].legend(loc='upper left')
    ax[1].set_title(output_time, fontsize=LARGE_SIZE)
    ax[1].set_xticks(ticks=[0.2, 0.3, 0.4, 0.5, 0.6], labels=['0.2', '0.3', '0.4', '0.5', '0.6'])

    plt.tight_layout()
    plt.show()
