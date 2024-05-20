# ------------------------------------------------------------------------ #
#
#       quail: A lightweight discontinuous Galerkin code for
#              teaching and prototyping
#		<https://github.com/IhmeGroup/quail>
#
#		Copyright (C) 2020-2021
#
#       This program is distributed under the terms of the GNU
#		General Public License v3.0. You should have received a copy
#       of the GNU General Public License along with this program.
#		If not, see <https://www.gnu.org/licenses/>.
#
# ------------------------------------------------------------------------ #

# ------------------------------------------------------------------------ #
#
#       File : src/physics/base/thermo.py
#
#       Contains functions for computing thermodynamic properties.
#
# ------------------------------------------------------------------------ #
from abc import ABC, abstractmethod
import numpy as np
from quail.external.optional_cantera import ct
from quail.external.optional_mutationpp import mpp


class ThermoBase(ABC):
    '''
    This is a base class for thermodynamic property calculations.

    Abstract Constants:
    -------------------
    NUM_SPECIES
        number of species
    NUM_ENERGY
        number of energies
    '''
    @property
    @abstractmethod
    def NUM_SPECIES(self):
        '''
        Number of species
        '''
        pass

    @property
    @abstractmethod
    def NUM_ENERGY(self):
        '''
        Number of energies
        '''
        pass

    def __init__(self, **kwargs):
        pass

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, T):
        self._T = T

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, p):
        self._p = p

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, rho):
        self._rho = rho

    @property
    def Y(self):
        return self._Y

    @Y.setter
    def Y(self, Y):
        self._Y = Y

    @property
    def e(self):
        return self._e

    @e.setter
    def e(self, e):
        self._e = e

    @property
    def cv(self):
        return self._cv

    @cv.setter
    def cv(self, cv):
        self._cv = cv

    @property
    def cp(self):
        return self._cp

    @cp.setter
    def cp(self, cp):
        self._cp = cp

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma

    @property
    def h(self):
        raise NotImplementedError("Current thermodynamic model does " +
                                  "not implement enthalpy calculation.")

    @property
    def s(self):
        raise NotImplementedError("Current thermodynamic model does " +
                                  "not implement entropy calculation.")

    @property
    def c(self):
        raise NotImplementedError("Current thermodynamic model does " +
                                  "not implement speed of sound calculation.")

    @property
    def dpdrho(self):
        raise NotImplementedError("Current thermodynamic model does " +
                                  "not implement pressure derivatives.")

    @property
    def dpde(self):
        raise NotImplementedError("Current thermodynamic model does " +
                                  "not implement pressure derivatives.")

    def set_state_from_rhoi_p(self, rhoi, p):
        pass

    def set_state_from_Y_T_p(self, Y, T, p):
        pass

    def set_state_from_rhoi_e(self, rhoi, e):
        pass

    def get_thermal_conductivity(self):
        pass

    def get_diffusion_coefficients(self):
        pass


class CaloricallyPerfectGas(ThermoBase):
    '''
    Calorically perfect gas
    '''
    NUM_SPECIES = 1
    NUM_ENERGY = 1

    def __init__(self, GasConstant=287.0, SpecificHeatRatio=1.4, **kwargs):
        super().__init__()
        self.gamma = SpecificHeatRatio
        self.R = GasConstant
        self.Y = 1.0
        self.inputs = None

        self.cv = self.R / (self.gamma - 1.0)
        self.cp = self.cv + self.R

    @property
    def h(self):
        return self.e + self.p / self.rho

    @property
    def s(self):
        # return R*(gamma/(gamma-1.)*np.log(T) - np.log(p))
        # Alternate way
        return np.log(self.p / self.rho**self.gamma)

    @property
    def c(self):
        return np.sqrt(self.gamma*self.p/self.rho)

    @property
    def p_jac(self):
        if self.inputs == 'rhoi_e':
            dpdrhoi = (self.gamma-1.0)*self.e
            dpde = (self.gamma-1.0)*self.rho
            return dpdrhoi, dpde
        elif self.inputs == 'rhoi_p':
            return 0.0, 1.0
        elif self.inputs == 'Y_T_P':
            return 0.0, 0.0, 1.0
        else:
            raise NotImplementedError

    @property
    def dpdrhoE(self):
        return self.gamma - 1.0

    def set_state_from_rhoi_p(self, rhoi, p):
        self.rho = rhoi.sum(axis=2, keepdims=True)
        self.p = p

        RT = self.p / self.rho

        self.T = RT / self.R
        self.e = RT / (self.gamma - 1.0)

        self.inputs = 'rhoi_p'

    def set_state_from_Y_T_p(self, Y, T, p):
        self.T = T
        self.p = p

        RT = self.R * self.T

        self.rho = p / RT
        self.e = self.R * self.T / (self.gamma - 1.0)

        self.inputs = 'Y_T_p'

    def set_state_from_rhoi_e(self, rhoi, e):
        self.rho = rhoi.sum(axis=2, keepdims=True)
        self.e = e
        self.p = (self.gamma - 1.0)*(self.rho * self.e)
        self.T = self.p / (self.R * self.rho)

        self.inputs = 'rhoi_e'


class CanteraThermo(ThermoBase):
    '''
    Interface to Cantera to compute transport properties.
    '''
    NUM_ENERGY = 1

    @property
    def NUM_SPECIES(self):
        return self.gas.n_species

    def __init__(self, Mechanism='air.yaml', **kwargs):
        super().__init__()
        self.R = ct.gas_constant

        # Initialize the gas phase
        self.gas = ct.Solution(Mechanism)

        self.solution = None

    @property
    def T(self):
        return self.solution.T

    @property
    def p(self):
        return self.solution.P

    @property
    def rho(self):
        return self.solution.density_mass

    @property
    def e(self):
        return self.solution.int_energy_mass

    @property
    def h(self):
        return self.solution.enthalpy_mass

    @property
    def Y(self):
        return self.solution.Y

    @property
    def cv(self):
        return self.solution.cv

    @property
    def cp(self):
        return self.solution.cp

    @property
    def gamma(self):
        return self.cp / self.cv

    @property
    def net_production_rates(self):
        return self.solution.net_production_rates

    def set_state_from_Y_T_p(self, Y, T, p):
        # Create a SolutionArray to handle the thermodynamic computations
        self.solution = ct.SolutionArray(self.gas, shape=T.shape)

        self.solution.TPY = T, p, Y

    def set_state_from_rhoi_e(self, rhoi, e):
        # Create a SolutionArray to handle the thermodynamic computations
        self.solution = ct.SolutionArray(self.gas, shape=e.shape)

        # Get species mass fractions and density
        rho = rhoi.sum(axis=2)
        v = 1.0 / rho
        Y = rhoi * v

        self.solution.UVY = e, v, Y
