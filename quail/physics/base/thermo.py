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
        return np.sqrt(self.gamma*self.dpdrho)

    @property
    def p_jac(self):
        raise NotImplementedError("Current thermodynamic model does " +
                                  "not implement pressure derivatives.")

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
        self.Y = self.default_Y = np.array([1.0])
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
    def dpdrho(self):
        return self.p / self.rho

    def set_state_from_rhoi_p(self, rhoi, p):
        self.rho = rhoi.sum(axis=2, keepdims=True)
        self.Y = rhoi / self.rho
        self.p = p

        RT = self.p / self.rho

        self.T = RT / self.R
        self.e = RT / (self.gamma - 1.0)

        self.inputs = 'rhoi_p'

    def set_state_from_Y_T_p(self, Y, T, p):
        self.Y = Y
        self.T = T
        self.p = p

        RT = self.R * self.T

        self.rho = p / RT
        self.e = RT / (self.gamma - 1.0)

        self.inputs = 'Y_T_p'

    def set_state_from_rhoi_e(self, rhoi, e):
        self.rho = rhoi.sum(axis=2, keepdims=True)
        self.Y = rhoi / self.rho
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
        self.species_names = self.gas.species_names
        self.default_Y = self.gas.Y

        self.solution = None

    @property
    def T(self):
        return self.solution.T[..., None]

    @property
    def p(self):
        return self.solution.P[..., None]

    @property
    def rho(self):
        return self.solution.density_mass[..., None]

    @property
    def e(self):
        return self.solution.int_energy_mass[..., None]

    @property
    def h(self):
        return self.solution.enthalpy_mass[..., None]

    @property
    def s(self):
        return self.solution.entropy_mass[..., None]

    @property
    def c(self):
        return np.sqrt(self.gamma*self.p/self.rho)

    @property
    def Y(self):
        return self.solution.Y

    @property
    def cv(self):
        return self.solution.cv[..., None]

    @property
    def cp(self):
        return self.solution.cp[..., None]

    @property
    def gamma(self):
        return self.cp / self.cv

    @property
    def dpdrho(self):
        return self.p / self.rho

    @property
    def net_production_rates(self):
        return self.solution.net_production_rates

    def set_state_from_rhoi_p(self, rhoi, p):
        # Create a SolutionArray to handle the thermodynamic computations
        self.solution = ct.SolutionArray(self.gas, shape=p.shape[:2])

        rho = rhoi.sum(axis=2, keepdims=True)
        Y = rhoi / rho

        self.solution.DPY = rho[..., 0], p[..., 0], Y

    def set_state_from_Y_T_p(self, Y, T, p):
        # Create a SolutionArray to handle the thermodynamic computations
        self.solution = ct.SolutionArray(self.gas, shape=T.shape[:2])

        self.solution.TPY = T[..., 0], p[..., 0], Y

    def set_state_from_rhoi_e(self, rhoi, e):
        # Create a SolutionArray to handle the thermodynamic computations
        self.solution = ct.SolutionArray(self.gas, shape=e.shape[:2])

        # Get species mass fractions and density
        rho = rhoi.sum(axis=2, keepdims=True)
        v = 1.0 / rho
        Y = rhoi * v

        self.solution.UVY = e[..., 0], v[..., 0], Y
