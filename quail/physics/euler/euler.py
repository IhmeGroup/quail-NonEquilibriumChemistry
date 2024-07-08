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
#       File : src/physics/euler/euler.py
#
#       Contains class definitions for 1D and 2D Euler equations.
#
# ------------------------------------------------------------------------ #
from enum import Enum
from functools import cached_property
import numpy as np

from quail import errors, general

from quail.physics.base import base
import quail.physics.base.functions as base_fcns
from quail.physics.base.functions import BCType as base_BC_type
from quail.physics.base.functions import ConvNumFluxType as base_conv_num_flux_type
from quail.physics.base.functions import FcnType as base_fcn_type

import quail.physics.euler.functions as euler_fcns
from quail.physics.euler.functions import BCType as euler_BC_type
from quail.physics.euler.functions import ConvNumFluxType as \
        euler_conv_num_flux_type
from quail.physics.euler.functions import FcnType as euler_fcn_type
from quail.physics.euler.functions import SourceType as euler_source_type


class Euler(base.PhysicsBase):
    '''
    This class corresponds to the compressible Euler equations for a
    calorically perfect gas. It inherits attributes and methods from the
    PhysicsBase class. See PhysicsBase for detailed comments of attributes
    and methods. This class should not be instantiated directly. Instead,
    the 1D and 2D variants, which inherit from this class (see below),
    should be instantiated.

    Additional methods and attributes are commented below.

    Attributes:
    -----------
    velocity: float
        local velocity for a given state vector
    '''
    PHYSICS_TYPE = general.PhysicsType.Euler

    def __getstate__(self):
        state = self.__dict__.copy()
        # Workaround because cached properties cannot be pickled
        state.pop("StateVariables", None)
        state.pop("AdditionalVariables", None)
        return state

    @property
    def NUM_STATE_VARS(self):
        return self.thermo.NUM_SPECIES + self.NDIMS + self.thermo.NUM_ENERGY

    def __init__(self, thermo, transport, NDIMS):
        super().__init__(thermo, transport, NDIMS)

    def set_maps(self):
        super().set_maps()

        self.BC_map.update({
            base_BC_type.StateAll : base_fcns.StateAll,
            base_BC_type.Extrapolate : base_fcns.Extrapolate,
            euler_BC_type.SlipWall : euler_fcns.SlipWall,
            euler_BC_type.PressureOutlet : euler_fcns.PressureOutlet,
        })

        if self.NDIMS == 1:
            # Define functions for 1D problem types
            d = {
                euler_fcn_type.SmoothIsentropicFlow:
                    euler_fcns.SmoothIsentropicFlow,
                euler_fcn_type.MovingShock: euler_fcns.MovingShock,
                euler_fcn_type.DensityWave: euler_fcns.DensityWave,
                euler_fcn_type.RiemannProblem: euler_fcns.RiemannProblem,
                euler_fcn_type.ShuOsherProblem:
                    euler_fcns.ShuOsherProblem,
            }

            self.source_map.update({
                euler_source_type.StiffFriction: euler_fcns.StiffFriction,
            })

            self.conv_num_flux_map.update({
                base_conv_num_flux_type.LaxFriedrichs:
                        euler_fcns.LaxFriedrichs,
                euler_conv_num_flux_type.Roe: euler_fcns.Roe1D,
            })
        elif self.NDIMS == 2:
            # Define functions for 2D problem types
            d = {
                euler_fcn_type.IsentropicVortex: euler_fcns.IsentropicVortex,
                euler_fcn_type.TaylorGreenVortex: euler_fcns.TaylorGreenVortex,
                euler_fcn_type.GravityRiemann: euler_fcns.GravityRiemann,
            }

            self.source_map.update({
                euler_source_type.StiffFriction: euler_fcns.StiffFriction,
                euler_source_type.TaylorGreenSource:
                        euler_fcns.TaylorGreenSource,
                euler_source_type.GravitySource: euler_fcns.GravitySource,
            })

            self.conv_num_flux_map.update({
                base_conv_num_flux_type.LaxFriedrichs:
                    euler_fcns.LaxFriedrichs,
                euler_conv_num_flux_type.Roe: euler_fcns.Roe2D,
            })

        self.IC_fcn_map.update(d)
        self.exact_fcn_map.update(d)
        self.BC_fcn_map.update(d)

    @cached_property
    def StateVariables(self):
        state_variable_list = {}

        # Handle multi-species
        if self.thermo.NUM_SPECIES == 1:
            state_variable_list["Densities"] = "\\rho"
        else:
            densities = []
            for sp in self.thermo.species_names:
                val = "\\rho Y_{%s}" % sp
                state_variable_list['rhoY%s' % sp] = val
                densities += [val]

            state_variable_list["Densities"] = densities

        # Handle multiple dimensions
        if self.NDIMS > 0:
            dim_list = [('X', 'u'), ('Y', 'v'), ('Z', 'w')]
            momenta = []
            for i in range(self.NDIMS):
                dim, vel = dim_list[i]
                val = "\\rho %s" % vel
                state_variable_list["%sMomentum" % dim] = val
                momenta += [val]

            state_variable_list["Momentum"] = momenta

        # Handle multiple energies
        if self.thermo.NUM_ENERGY == 1:
            state_variable_list["Energies"] = "\\rho E"
        else:
            energies = []
            for energy in self.thermo.energy_list:
                val = "\\rho %s" % energy
                state_variable_list['rho%s' % energy] = energy
                energies += [val]

            state_variable_list["Energies"] = energies

        return Enum("StateVariables", state_variable_list)

    def get_state_indices(self):
        irho = self.get_state_index("Densities")
        irhou = self.get_state_index("Momentum")
        irhoE = self.get_state_index("Energies")

        return irho, irhou, irhoE

    def get_state_slices(self):
        srho = self.get_state_slice("Densities")
        srhou = self.get_state_slice("Momentum")
        srhoE = self.get_state_slice("Energies")

        return srho, srhou, srhoE

    @cached_property
    def AdditionalVariables(self):
        additional_variable_list = {
            "Density": "\\rho",
            "Pressure": "p",
            "Temperature": "T",
            "Entropy": "s",
            "TotalEnergy": "\\rho E",
            "KineticEnergy": "\\rho u^2",
            "InternalEnergy": "\\rho e",
            "TotalEnthalpy": "H",
            "SoundSpeed": "c",
            "MaxWaveSpeed": "\\lambda",
        }

        # Handle multi-species
        if self.thermo.NUM_SPECIES > 1:
            Y_list = []
            for sp in self.thermo.species_names:
                val = "Y_{%s}" % sp
                additional_variable_list["MassFraction%s" % sp] = val
                Y_list += [val]
            additional_variable_list["MassFractions"] = Y_list

        # Handle multiple dimensions
        if self.NDIMS > 0:
            additional_variable_list['Velocity'] = "u"
            additional_variable_list['VelocityMagnitude'] = "|u|"

            dim_list = [('X', 'u'), ('Y', 'v'), ('Z', 'w')]
            for i in range(self.NDIMS):
                dim, vel = dim_list[i]
                additional_variable_list["%sVelocity" % dim] = vel

        return Enum('AdditionalVariables', additional_variable_list)

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, velocity):
        self._velocity = velocity

    @property
    def kinetic_energy(self):
        return self._kinetic_energy

    @kinetic_energy.setter
    def kinetic_energy(self, kinetic_energy):
        self._kinetic_energy = kinetic_energy

    def set_thermo_state(self, Uq, flag_non_physical=None):
        ''' Extract state variables '''
        srho, srhou, _ = self.get_state_slices()

        rhoi = Uq[:, :, srho]  # [n, nq, nsp]
        rhou = Uq[:, :, srhou] # [n, nq, ndim]

        ''' Flag non-physical state '''
        if flag_non_physical:
            if np.any(rhoi < 0.):
                raise errors.NotPhysicalError

        '''Compute density and energy'''
        rho = self.compute_additional_variable("Density", Uq, flag_non_physical)
        e = self.compute_additional_variable("InternalEnergy", Uq, flag_non_physical) / rho

        '''Set the thermodynamic state'''
        self.thermo.set_state_from_rhoi_e(rhoi, e)

        '''Compute and store velocity'''
        self.velocity = rhou / rho

    def compute_additional_variable(self, var_name, Uq, flag_non_physical):
        ''' Compute '''
        vname = self.AdditionalVariables[var_name].name

        # The following variables can be computed directly:
        if self.match_variable(vname, "Density"):
            srho = self.get_state_slice("Densities")
            return Uq[..., srho].sum(axis=2, keepdims=True)
        elif self.match_variable(vname, "TotalEnergy"):
            irhoE = self.get_state_index("Energies")
            return Uq[..., [irhoE]]
        elif self.match_variable(vname, "KineticEnergy"):
            rho = self.compute_variable("Density", Uq)
            rhou = self.compute_variable("Momentum", Uq)
            self.kinetic_energy = 0.5*(rhou * rhou).sum(axis=2, keepdims=True) / rho
            return self.kinetic_energy
        elif self.match_variable(vname, "InternalEnergy"):
            rhoE = self.compute_variable("TotalEnergy", Uq)
            ke = self.compute_variable("KineticEnergy", Uq)
            return rhoE - ke
        elif self.match_variable(vname, "Velocity"):
            rho = self.compute_variable("Density", Uq)
            rhou = self.compute_variable("Momentum", Uq)
            self.velocity = rhou / rho
            return self.velocity
        elif self.match_variable(vname, "VelocityMagnitude"):
            return np.linalg.norm(self.velocity, axis=2, keepdims=True)
        elif self.match_variable(vname, "XVelocity"):
            return self.velocity[:, :, [0]]
        elif self.match_variable(vname, "YVelocity"):
            return self.velocity[:, :, [1]]
        elif self.match_variable(vname, "ZVelocity"):
            return self.velocity[:, :, [2]]

        # For other quantities, first set the thermodynamic state:
        self.set_thermo_state(Uq, flag_non_physical)

        if self.match_variable(vname, "Pressure"):
            varq = self.thermo.p
        elif self.match_variable(vname, "Temperature"):
            varq = self.thermo.T
        elif self.match_variable(vname, "Entropy"):
            varq = self.thermo.s
        elif self.match_variable(vname, "TotalEnthalpy"):
            rhoE = self.compute_variable("TotalEnergy", Uq)
            varq = (rhoE + self.thermo.p) / self.thermo.rho
            # varq = self.thermo.h + self.kinetic_energy / self.thermo.rho
        elif self.match_variable(vname, "SoundSpeed"):
            varq = self.thermo.c
        elif self.match_variable(vname, "MaxWaveSpeed"):
            # |u| + c
            velmag = self.compute_variable("VelocityMagnitude", Uq)
            varq = velmag + self.thermo.c
        else:
            raise NotImplementedError

        return varq

    def compute_pressure_gradient(self, Uq, grad_Uq):
        '''
        Compute the gradient of pressure with respect to physical space. This is
        needed for pressure-based shock sensors.

        Inputs:
        -------
            Uq: solution in each element evaluated at quadrature points
            [ne, nq, ns]
            grad_Uq: gradient of solution in each element evaluted at quadrature
                points [ne, nq, ns, ndims]

        Outputs:
        --------
            array: gradient of pressure with respected to physical space
                [ne, nq, ndims]
        '''
        # Set the thermodynamic state
        self.set_thermo_state(Uq)

        # Compute the pressure derivatives w.r.t. the primitive variables
        dpdrhoi, dpde = self.thermo.p_jac

        # Compute the primitive variable derivatives w.r.t. the conservative variables
        srho, srhou, srhoE = self.get_state_slices()
        rho = self.thermo.rho
        rhou = Uq[:, :, srhou]

        dedrho = (self.kinetic_energy/rho - self.thermo.e) / rho
        dedrhou = -rhou / rho**2
        dedrhoE = 1.0 / rho

        # Compute dp/dU
        dpdU = np.empty_like(Uq)
        dpdU[:, :, srho] = dpdrhoi + dedrho * dpde
        dpdU[:, :, srhou] = dedrhou * dpde
        dpdU[:, :, srhoE] = dedrhoE * dpde

        # Multiply with dU/dx
        return np.einsum('ijk, ijkl -> ijl', dpdU, grad_Uq)

    def get_conv_flux_interior(self, Uq):
        # Set the thermodynamic state
        self.set_thermo_state(Uq)

        # Get indices of state variables
        srho, srhou, srhoE = self.get_state_slices()

        rho = self.thermo.rho
        u = self.velocity
        rhou = Uq[:, :, None, srhou] # [n, nq, 1, ndims]

        # Calculate pressure
        p = self.thermo.p

        # Get momentum flux matrix
        momentum_flux = u[..., :, None] * rhou  # [n, nq, ndims, ndims]

        # Add pressure to the diagonal
        idx_diag = 2*(tuple(range(self.NDIMS)),)
        momentum_flux[..., *idx_diag] += p

        # Get mass fraction(s)
        Y = self.thermo.Y[..., None]
        # Get total enthalpy
        h = (self.thermo.h + self.kinetic_energy / rho)[..., None]

        # Assemble flux matrix
        F = np.empty(Uq.shape + (self.NDIMS,)) # [n, nq, ns, ndims]
        F[:, :, srho, :] = Y * rhou        # Flux of mass in all directions
        F[:, :, srhou, :] = momentum_flux  # Flux of momentum
        F[:, :, srhoE, :] = h * rhou       # Flux of energy

        return F, (u, self.thermo)

    def get_conv_eigenvectors(self, U_bar):
        '''
        This function defines the convective eigenvectors for the
        1D euler equations. This is used with the WENO limiter to
        transform the system of equations from physical space to
        characteristic space.

        Inputs:
        -------
            U_bar: Average state [ne, 1, ns]

        Outputs:
        --------
            right_eigen: Right eigenvector matrix [ne, 1, ns, ns]
            left_eigen: Left eigenvector matrix [ne, 1, ns, ns]
        '''
        # Set the thermodynamic state
        self.set_thermo_state(U_bar)

        # Unpack
        ne = U_bar.shape[0]

        ns = self.NUM_STATE_VARS

        srho, srhou, srhoE = self.get_state_slices()

        # Get velocity
        u = self.velocity[..., None]
        # Get squared velocity
        u2 = u*u
        # Calculate pressure
        p = self.thermo.p[..., None] # [n, nq, 1]
        # Get total specific enthalpy
        H = (self.thermo.h + self.kinetic_energy / self.thermo.rho)[..., None]
        # Get sound speed
        a = self.thermo.c[..., None]

        gm1oa2 = (self.thermo.gamma - 1.) / (a * a)

        # Allocate the right and left eigenvectors
        right_eigen = np.zeros([ne, 1, ns, ns])
        left_eigen = np.zeros([ne, 1, ns, ns])

        # # Calculate the right and left eigenvectors
        right_eigen[:, :, srho, srho]  = 1.
        right_eigen[:, :, srho, srhou] = 1.
        right_eigen[:, :, srho, srhoE] = 1.

        right_eigen[:, :, srhou, srho]  = u - a
        right_eigen[:, :, srhou, srhou] = u + a
        right_eigen[:, :, srhou, srhoE] = u

        right_eigen[:, :, srhoE, srho]  = H - u*a
        right_eigen[:, :, srhoE, srhou] = H + u*a
        right_eigen[:, :, srhoE, srhoE] = 0.5 * u2

        left_eigen[:, :, srho, srho]  = 0.5 * (0.5*gm1oa2 * u2 + u/a)
        left_eigen[:, :, srho, srhou] = -0.5 * (gm1oa2 * u + 1./a)
        left_eigen[:, :, srho, srhoE] = 0.5 * gm1oa2

        left_eigen[:, :, srhou, srho]  = 0.5 * (0.5*gm1oa2 * u2 - u/a)
        left_eigen[:, :, srhou, srhou] = -0.5 * (gm1oa2 * u - 1./a)
        left_eigen[:, :, srhou, srhoE] = 0.5 * gm1oa2

        left_eigen[:, :, srhoE, srho]  = 1. - 0.5 * gm1oa2 * u2
        left_eigen[:, :, srhoE, srhou] = gm1oa2 * u
        left_eigen[:, :, srhoE, srhoE] = -1.*gm1oa2

        # Can uncomment line below to test l dot r = kronecker delta
        # test = np.einsum('elij,eljk->elik', left_eigen, right_eigen)

        return right_eigen, left_eigen # [ne, 1, ns, ns]
