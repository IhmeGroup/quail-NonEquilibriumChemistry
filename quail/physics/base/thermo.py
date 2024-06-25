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
from quail.general import ThermoType
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

    @property
    @abstractmethod
    def THERMO_TYPE(self):
        '''
        Thermodynamic model type (general.ThermoType enum member)
        '''
        pass

    def __init__(self, **kwargs):
        self.init_params = kwargs
        pass

    def reinitialize(self):
        self.__init__(**self.init_params)
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
    THERMO_TYPE = ThermoType.CaloricallyPerfectGas

    def __init__(self, GasConstant=287.0, SpecificHeatRatio=1.4, **kwargs):
        super().__init__(GasConstant=GasConstant,
                         SpecificHeatRatio=SpecificHeatRatio)
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
    THERMO_TYPE = ThermoType.Cantera

    def __getstate__(self):
        state = self.__dict__.copy()
        # Workaround because Cantera objects cannot be pickled
        state.pop("gas", None)
        state.pop("solution", None)
        return state

    @property
    def NUM_SPECIES(self):
        return self.gas.n_species

    def __init__(self, Mechanism='air.yaml', OffsetEnergy=True, **kwargs):
        super().__init__(Mechanism=Mechanism, OffsetEnergy=OffsetEnergy)
        self.OffsetEnergy = OffsetEnergy
        self.Ru = ct.gas_constant

        # Initialize the gas phase
        self.gas = ct.Solution(Mechanism)
        self.gas.basis = 'mass'
        self.species_names = self.gas.species_names
        self.default_Y = self.gas.Y

        if self.OffsetEnergy:
            # Store the reference internal energies which bound the minimum
            # temperature for the reference pressure
            Tref = self.gas.min_temp
            self.gas.TP = Tref, self.gas.reference_pressure
            self.eref = (self.gas.partial_molar_int_energies / self.gas.molecular_weights)[None, None, :]

        self.solution = None

    @property
    def R(self):
        return self.Ru / np.atleast_3d(self.solution.mean_molecular_weight)

    @property
    def T(self):
        return np.atleast_3d(self.solution.T)

    @property
    def p(self):
        return np.atleast_3d(self.solution.P)

    @property
    def rho(self):
        return np.atleast_3d(self.solution.density_mass)

    @property
    def e0(self):
        if self.OffsetEnergy:
            return np.sum(self.Y * self.eref, axis=-1, keepdims=True)
        return 0.0

    @property
    def e(self):
        return np.atleast_3d(self.solution.int_energy_mass) - self.e0

    @property
    def h(self):
        return np.atleast_3d(self.solution.enthalpy_mass) - self.e0

    @property
    def s(self):
        return np.atleast_3d(self.solution.entropy_mass)

    @property
    def c(self):
        return np.sqrt(self.gamma*self.p/self.rho)

    @property
    def Y(self):
        return self.solution.Y

    @property
    def cv(self):
        return np.atleast_3d(self.solution.cv)

    @property
    def cp(self):
        return np.atleast_3d(self.solution.cp)

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
        if hasattr(T, 'shape'):
            shape = T.shape
            p = np.broadcast_to(p, shape)
        elif hasattr(p, 'shape'):
            shape = p.shape
            T = np.broadcast_to(T, shape)
        self.solution = ct.SolutionArray(self.gas, shape=shape[:2])

        self.solution.TPY = T[..., 0], p[..., 0], Y

    def set_state_from_rhoi_e(self, rhoi, e):
        # Create a SolutionArray to handle the thermodynamic computations
        self.solution = ct.SolutionArray(self.gas, shape=e.shape[:2])

        # Get species mass fractions and density
        rho = rhoi.sum(axis=2, keepdims=True)
        v = 1.0 / rho
        Y = rhoi * v

        # Add reference energy if needed
        if self.OffsetEnergy:
            e += np.sum(Y * self.eref, axis=-1, keepdims=True)

        self.solution.UVY = e[..., 0], v[..., 0], Y
