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
#       File : src/physics/navierstokes/navierstokes.py
#
#       Contains class definitions for 1D and 2D Navier-Stokes equations.
#
# ------------------------------------------------------------------------ #
from enum import Enum
import numpy as np

from quail import errors, general

from quail.physics.base import base
import quail.physics.base.functions as base_fcns
from quail.physics.base.functions import BCType as base_BC_type
from quail.physics.base.functions import ConvNumFluxType as base_conv_num_flux_type
from quail.physics.base.functions import DiffNumFluxType as base_diff_num_flux_type
from quail.physics.base.functions import FcnType as base_fcn_type

from quail.physics.euler import euler
import quail.physics.euler.functions as euler_fcns
from quail.physics.euler.functions import BCType as euler_BC_type
from quail.physics.euler.functions import ConvNumFluxType as \
        euler_conv_num_flux_type
from quail.physics.euler.functions import FcnType as euler_fcn_type
from quail.physics.euler.functions import SourceType as euler_source_type

import quail.physics.navierstokes.functions as navierstokes_fcns
from quail.physics.navierstokes.functions import BCType as navierstokes_bc_type
from quail.physics.navierstokes.functions import FcnType as navierstokes_fcn_type
from quail.physics.navierstokes.functions import SourceType as \
        navierstokes_source_type


class NavierStokes(euler.Euler):
    '''
    This class corresponds to the compressible Navier-Stokes equations.
    It inherits attributes and methods from the Euler class. See Euler
    for detailed comments of attributes and methods. This class should
    not be instantiated directly. Instead, the 1D and 2D variants, which
    inherit from this class (see below), should be instantiated.

    Additional methods and attributes are commented below.

    Attributes:
    -----------
    R: float
        mass-specific gas constant
    gamma: float
        specific heat ratio
    '''
    PHYSICS_TYPE = general.PhysicsType.NavierStokes

    def set_maps(self):
        super().set_maps()

        self.diff_num_flux_map.update({
            base_diff_num_flux_type.SIP :
                base_fcns.SIP,
            })

        self.BC_map.update({
            base_BC_type.StateAll : base_fcns.StateAll,
            base_BC_type.Extrapolate : base_fcns.Extrapolate,
            navierstokes_bc_type.IsothermalWall :
                    navierstokes_fcns.IsothermalWall,
            navierstokes_bc_type.AdiabaticWall :
                    navierstokes_fcns.AdiabaticWall,
        })

        if self.NDIMS == 2:
            # Define functions for 2D problem types
            d = {
                navierstokes_fcn_type.TaylorGreenVortexNS :
                        navierstokes_fcns.TaylorGreenVortexNS,
                navierstokes_fcn_type.ManufacturedSolution :
                        navierstokes_fcns.ManufacturedSolution,
            }

            self.IC_fcn_map.update(d)
            self.exact_fcn_map.update(d)
            self.BC_fcn_map.update(d)

            self.source_map.update({
                navierstokes_source_type.ManufacturedSource :
                        navierstokes_fcns.ManufacturedSource,
            })

    def get_diff_flux_interior(self, Uq, gUq):
        # Set the thermodynamic state
        self.set_thermo_state(Uq)

        # Get indices/slices of state variables
        srho, srhou, srhoE = self.get_state_slices()

        # Calculate transport
        rho = self.thermo.rho
        mu = self.transport.get_viscosity(self.thermo)
        kappa = self.transport.get_thermal_conductivity(self.thermo)
        nu = mu / rho

        # Get velocity in each dimension
        u = self.velocity

        # Compute spatial gradient of temperature
        gT = self.compute_temperature_gradient(Uq, gUq)

        # Get density and momentum gradients
        grho = gUq[:, :, srho, :].sum(axis=2, keepdims=True)
        grhou = gUq[:, :, srhou, :]

        # Get the stress tensor (use product rules to write in
        # terms of the conservative gradients)
        # rho * dui/dxj = drhoui/dxj - ui * drho/dxj
        rho_gu = grhou - u[..., None]*grho
        tauij = rho_gu + np.swapaxes(rho_gu, 2, 3)

        # Subtract rho*duk/dxk = drhouk/dxk - uk*drho/dxk
        idx_diag = 2*(tuple(range(self.NDIMS)),)
        rhodiv = (grhou[:, :, *idx_diag] - u*grho[:, :, 0, :]
                  ).sum(axis=2, keepdims=True)
        tauij[:, :, *idx_diag] -= 2.0/3.0 * rhodiv

        # Multiply by mu / rho
        tauij *= nu[...,None]

        # Assemble flux matrix
        F = np.empty(Uq.shape + (self.NDIMS,)) # [n, nq, ns, ndims]
        F[:,:,srho, :] = 0.		# x,y-flux of rho (zero both dir)
        F[:,:,srhou,:] = tauij  # Stress tensor
        F[:,:,srhoE,:] = (u[..., None] * tauij).sum(axis=2, keepdims=True) + \
            (kappa * gT)[:,:,None,:]

        return F # [n, nq, ns, ndims]