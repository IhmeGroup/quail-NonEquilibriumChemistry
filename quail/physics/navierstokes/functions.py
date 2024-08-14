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
#       File : src/physics/navierstokes/functions.py
#
#       Contains definitions of Functions, boundary conditions, and source
#       terms for the Euler equations.
#
# ------------------------------------------------------------------------ #
from enum import Enum, auto
from quail.general import pi
from quail.backend import np
from quail.physics.base.data import FcnBase, SourceBase


class FcnType(Enum):
	'''
	Enum class that stores the types of analytical functions for initial
	conditions, exact solutions, and/or boundary conditions. These
	functions are specific to the available Euler equation sets.
	'''
	TaylorGreenVortexNS = auto()
	ManufacturedSolution = auto()

class BCType(Enum):
	'''
	Enum class that stores the types of boundary conditions. These
	boundary conditions are specific to the available Euler equation sets.
	'''
	pass


class SourceType(Enum):
	'''
	Enum class that stores the types of source terms. These
	source terms are specific to the available Euler equation sets.
	'''
	ManufacturedSource = auto()


'''
---------------
State functions
---------------
These classes inherit from the FcnBase class. See FcnBase for detailed
comments of attributes and methods. Information specific to the
corresponding child classes can be found below. These classes should
correspond to the FcnType enum members above.
'''

class ManufacturedSolution(FcnBase):
	'''
	Manufactured solution to the Navier-Stokes equations used for
	verifying the order of accuracy of a given scheme.

	Script to generate sources is located in examples/navierstokes/
	2D/manufacturedNS2D
	'''
	def __init__(self):
		pass
	def get_state(self, physics, x, t):
		# Unpack
		gamma = physics.thermo.gamma

		srho, srhou, srhoE = physics.get_state_slices()

		x1 = x[:, :, [0]]
		x2 = x[:, :, [1]]

		''' Fill state '''
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])

		# Generated initial condition from sympy

		Uq[:, :, srho] = 0.1*np.sin(pi*x1) + 0.1* \
			np.cos(pi*x1)*np.cos(pi*x2) -  \
			0.2*np.cos(pi*x2) + 1.0

		S_rhou = (0.1*np.sin(pi*x1) + 0.1* \
			np.cos(pi*x1)*np.cos(pi*x2) - 0.2* \
			np.cos(pi*x2) + 1.0)*(0.3*np.sin(3* \
			pi*x1) + 0.3*np.cos(pi*x1)* \
			np.cos(pi*x2) + 0.3*np.cos(pi*x2) + 2.0)

		S_rhov = (0.1*np.sin(pi*x1) + 0.1* \
			np.cos(pi*x1)*np.cos(pi*x2) - 0.2* \
			np.cos(pi*x2) + 1.0)*(0.3*np.sin(pi*x2) \
			+ 0.3*np.cos(pi*x1)*np.cos(pi*x2) + 0.3* \
			np.cos(pi*x1) + 2.0)

		Uq[:, :, srhou] = np.concatenate([S_rhou, S_rhov], axis=2)

		Uq[:, :, srhoE] = (4.0*(0.15*np.sin(3*pi*x1) + 0.15* \
			np.cos(pi*x1)*np.cos(pi*x2) + 0.15* \
			np.cos(pi*x2) + 1)**2 + 4.0*(0.15* \
			np.sin(pi*x2) + 0.15*np.cos(pi*x1)* \
			np.cos(pi*x2) + 0.15*np.cos(pi*x1) + 1)**2) \
			*(0.05*np.sin(pi*x1) + 0.05*np.cos(pi*x1)* \
			np.cos(pi*x2) - 0.1*np.cos(pi*x2) + 0.5) + \
			(1.0*np.sin(pi*x2) + 0.5*np.cos(pi*x1)* \
			np.cos(pi*x2) + 1.0*np.cos(2*pi*x1) + \
			10.0)/(gamma - 1)
		# End generated code

		return Uq # [ne, nq, ns]

class TaylorGreenVortexNS(FcnBase):
	'''
	2D Taylor Green Vortex Case
	'''
	def __init__(self, state={}):
		# Set default values
		self.init_state = {
			'rhoi': np.array([1.0]),
			'T': 1.0,
			'u': 0.0,
			'v': 0.0,
			'Ma': 0.01,
		} | state  # Override values provided in input deck

		# Remove redundant values
		if 'Y' in state.keys():
			self.init_state.pop('rhoi')

		# Turn all scalar values into arrays
		for key, value in self.init_state.items():
			self.init_state[key] = np.atleast_1d(value)

	def get_state(self, physics, x, t):
		# Unpack
		u = self.init_state['u']
		v = self.init_state['v']
		Ma = self.init_state['Ma']
		x1 = x[:, :, [0]] - u*t
		x2 = x[:, :, [1]] - v*t

		# Get the initial unperturbed state
		physics.get_state_from_primitives(**self.init_state)
		thermo = physics.thermo
		Y0 = thermo.Y
		rho0 = thermo.rho
		T0 = thermo.T
		RT = thermo.R * T0
		P0 = thermo.p

		gamma = thermo.gamma
		V0 = gamma*Ma*Ma*RT

		mu = physics.transport.get_viscosity(physics.thermo)
		nu = mu/rho0

		F = np.exp(-8. * pi*pi * nu * t)

		''' Fill state '''
		state = {
			'rhoi': rho0*Y0,
			'p': P0 + 0.25 * rho0 * (np.cos(4.*pi * x1) + np.cos(4.*pi * x2)) * F * F,
			'u': u + V0 * np.sin(2.*pi * x1) * np.cos(2.*pi * x2) * F,
			'v': v - V0 * np.cos(2.*pi * x1) * np.sin(2.*pi * x2) * F,
		}

		return physics.get_state_from_primitives(**state)


'''
---------------------
Source term functions
---------------------
These classes inherit from the SourceBase class. See SourceBase for detailed
comments of attributes and methods. Information specific to the
corresponding child classes can be found below. These classes should
correspond to the SourceType enum members above.
'''
class ManufacturedSource(SourceBase):
	'''
	Generated source term for the manufactured solution of the
	Navier-Stokes equations. Generated using script in
	examples/navierstokes/2D/manufactured_solution. Exact solution can
	be found in the following paper.
		[1] Dumbser, M. (2010)
	'''
	def get_source(self, physics, Uq, x, t):
		# Unpack
		gamma = physics.thermo.gamma
		R = physics.thermo.R

		srho, srhou, srhoE = physics.get_state_slices()
		x1 = x[:, :, [0]]
		x2 = x[:, :, [1]]

		mu = physics.transport.get_viscosity(physics.thermo)
		kappa = physics.transport.get_thermal_conductivity(physics.thermo)
		# import code; code.interact(local=locals())
		Sq = np.zeros_like(Uq)
		Sq[:, :, srho], S_rhou, S_rhov, Sq[:, :, srhoE] = \
			self.manufactured_source(x1, x2, t, gamma, kappa, mu, R)

		Sq[:, :, srhou] = np.concatenate([S_rhou, S_rhov], axis=2)

		return Sq # [ne, nq, ns]

	def manufactured_source(self, x1, x2, t, gamma, kappa, mu, R):


		# The following lines of code are generated using sympy
		S_rho = (-0.3*pi*np.sin(pi*x1)*np.cos(pi*x2) + \
			0.9*pi*np.cos(3*pi*x1))*(0.1*np.sin(pi*x1) \
			+ 0.1*np.cos(pi*x1)*np.cos(pi*x2) - 0.2* \
			np.cos(pi*x2) + 1.0) + (-0.1*pi* \
			np.sin(pi*x1)*np.cos(pi*x2) + 0.1*pi* \
			np.cos(pi*x1))*(0.3*np.sin(3*pi*x1) + \
			0.3*np.cos(pi*x1)*np.cos(pi*x2) + 0.3* \
			np.cos(pi*x2) + 2.0) + (-0.3*pi* \
			np.sin(pi*x2)*np.cos(pi*x1) + 0.3*pi* \
			np.cos(pi*x2))*(0.1*np.sin(pi*x1) + 0.1* \
			np.cos(pi*x1)*np.cos(pi*x2) - 0.2* \
			np.cos(pi*x2) + 1.0) + (-0.1*pi* \
			np.sin(pi*x2)*np.cos(pi*x1) + 0.2*pi* \
			np.sin(pi*x2))*(0.3*np.sin(pi*x2) + \
			0.3*np.cos(pi*x1)*np.cos(pi*x2) + \
			0.3*np.cos(pi*x1) + 2.0)

		S_rhou = -mu*(-0.2*pi**2*np.sin(pi*x1)* \
			np.sin(pi*x2) - 3.6*pi**2*np.sin(3*pi*x1) \
			 - 0.4*pi**2*np.cos(pi*x1)*np.cos(pi*x2)) \
			- mu*(0.3*pi**2*np.sin(pi*x1)*np.sin(pi*x2) \
			- 0.3*pi**2*np.cos(pi*x1)*np.cos(pi*x2) - \
			0.3*pi**2*np.cos(pi*x2)) + 4.0*(-0.3*pi* \
			np.sin(pi*x1)*np.cos(pi*x2) + 0.9*pi* \
			np.cos(3*pi*x1))*(0.1*np.sin(pi*x1) + \
			0.1*np.cos(pi*x1)*np.cos(pi*x2) - 0.2* \
			np.cos(pi*x2) + 1.0)*(0.15*np.sin(3*pi*x1) \
			+ 0.15*np.cos(pi*x1)*np.cos(pi*x2) + 0.15* \
			np.cos(pi*x2) + 1) + 4.0*(-0.1*pi* \
			np.sin(pi*x1)*np.cos(pi*x2) + 0.1*pi* \
			np.cos(pi*x1))*(0.15*np.sin(3*pi*x1) + 0.15* \
			np.cos(pi*x1)*np.cos(pi*x2) + 0.15* \
			np.cos(pi*x2) + 1)**2 + (-0.3*pi* \
			np.sin(pi*x2)*np.cos(pi*x1) - 0.3* \
			pi*np.sin(pi*x2))*(0.1*np.sin(pi*x1) + \
			0.1*np.cos(pi*x1)*np.cos(pi*x2) - 0.2* \
			np.cos(pi*x2) + 1.0)*(0.3*np.sin(pi*x2) + \
			0.3*np.cos(pi*x1)*np.cos(pi*x2) + 0.3* \
			np.cos(pi*x1) + 2.0) + (-0.3*pi* \
			np.sin(pi*x2)*np.cos(pi*x1) + 0.3*pi* \
			np.cos(pi*x2))*(0.1*np.sin(pi*x1) + 0.1* \
			np.cos(pi*x1)*np.cos(pi*x2) - 0.2* \
			np.cos(pi*x2) + 1.0)*(0.3*np.sin(3*pi*x1) \
			+ 0.3*np.cos(pi*x1)*np.cos(pi*x2) + 0.3* \
			np.cos(pi*x2) + 2.0) + (-0.1*pi* \
			np.sin(pi*x2)*np.cos(pi*x1) + 0.2*pi* \
			np.sin(pi*x2))*(0.3*np.sin(3*pi*x1) + 0.3* \
			np.cos(pi*x1)*np.cos(pi*x2) + 0.3* \
			np.cos(pi*x2) + 2.0)*(0.3*np.sin(pi*x2) + \
			0.3*np.cos(pi*x1)*np.cos(pi*x2) + 0.3* \
			np.cos(pi*x1) + 2.0) - 0.5*pi* \
			np.sin(pi*x1)*np.cos(pi*x2) - \
			2.0*pi*np.sin(2*pi*x1)

		S_rhov = -mu*(-0.2*pi**2*np.sin(pi*x1)* \
			np.sin(pi*x2) - 0.4*pi**2*np.sin(pi*x2) \
			- 0.4*pi**2*np.cos(pi*x1)*np.cos(pi*x2)) \
			 - mu*(0.3*pi**2*np.sin(pi*x1)* \
		 	np.sin(pi*x2) - 0.3*pi**2* \
		 	np.cos(pi*x1)*np.cos(pi*x2) - 0.3*pi**2* \
		 	np.cos(pi*x1)) + (-0.3*pi*np.sin(pi*x1)* \
		 	np.cos(pi*x2) - 0.3*pi*np.sin(pi*x1))* \
		 	(0.1*np.sin(pi*x1) + 0.1*np.cos(pi*x1)* \
	 		np.cos(pi*x2) - 0.2*np.cos(pi*x2) + 1.0)* \
	 		(0.3*np.sin(3*pi*x1) + 0.3*np.cos(pi*x1)* \
 			np.cos(pi*x2) + 0.3*np.cos(pi*x2) + 2.0) + \
 			(-0.3*pi*np.sin(pi*x1)*np.cos(pi*x2) + \
			0.9*pi*np.cos(3*pi*x1))*(0.1*np.sin(pi*x1) \
			+ 0.1*np.cos(pi*x1)*np.cos(pi*x2) - \
			0.2*np.cos(pi*x2) + 1.0)*(0.3*np.sin(pi*x2) + \
			0.3*np.cos(pi*x1)*np.cos(pi*x2) + 0.3* \
			np.cos(pi*x1) + 2.0) + (-0.1*pi* \
			np.sin(pi*x1)*np.cos(pi*x2) + 0.1*pi* \
			np.cos(pi*x1))*(0.3*np.sin(3*pi*x1) + 0.3* \
			np.cos(pi*x1)*np.cos(pi*x2) + 0.3* \
			np.cos(pi*x2) + 2.0)*(0.3*np.sin(pi*x2) + 0.3 \
			*np.cos(pi*x1)*np.cos(pi*x2) + 0.3* \
			np.cos(pi*x1) + 2.0) + 4.0*(-0.3*pi* \
			np.sin(pi*x2)*np.cos(pi*x1) + 0.3*pi* \
			np.cos(pi*x2))*(0.1*np.sin(pi*x1) + 0.1* \
			np.cos(pi*x1)*np.cos(pi*x2) - 0.2* \
			np.cos(pi*x2) + 1.0)*(0.15*np.sin(pi*x2) + \
			0.15*np.cos(pi*x1)*np.cos(pi*x2) + 0.15* \
			np.cos(pi*x1) + 1) + 4.0*(-0.1*pi* \
			np.sin(pi*x2)*np.cos(pi*x1) + 0.2*pi* \
			np.sin(pi*x2))*(0.15*np.sin(pi*x2) + 0.15* \
			np.cos(pi*x1)*np.cos(pi*x2) + 0.15* \
			np.cos(pi*x1) + 1)**2 - 0.5*pi* \
			np.sin(pi*x2)*np.cos(pi*x1) + \
			1.0*pi*np.cos(pi*x2)

		S_rhoE = -mu*(-0.3*pi*np.sin(pi*x1)* \
			np.cos(pi*x2) - 0.3*pi*np.sin(pi*x1)) \
			*(-0.3*pi*np.sin(pi*x1)*np.cos(pi*x2) \
			- 0.3*pi*np.sin(pi*x1) - 0.3*pi* \
			np.sin(pi*x2)*np.cos(pi*x1) - 0.3*pi* \
			np.sin(pi*x2)) - mu*(-0.3*pi* \
			np.sin(pi*x1)*np.cos(pi*x2) + 0.9*pi* \
			np.cos(3*pi*x1))*(-0.4*pi*np.sin(pi*x1)* \
			np.cos(pi*x2) + 0.2*pi*np.sin(pi*x2)* \
			np.cos(pi*x1) + 1.2*pi*np.cos(3*pi*x1) - \
			0.2*pi*np.cos(pi*x2)) - mu*(-0.3*pi* \
			np.sin(pi*x2)*np.cos(pi*x1) - 0.3*pi* \
			np.sin(pi*x2))*(-0.3*pi*np.sin(pi*x1)* \
			np.cos(pi*x2) - 0.3*pi*np.sin(pi*x1) - \
			0.3*pi*np.sin(pi*x2)*np.cos(pi*x1) - 0.3* \
			pi*np.sin(pi*x2)) - mu*(-0.3*pi* \
			np.sin(pi*x2)*np.cos(pi*x1) + 0.3*pi* \
			np.cos(pi*x2))*(0.2*pi*np.sin(pi*x1)* \
			np.cos(pi*x2) - 0.4*pi*np.sin(pi*x2)* \
			np.cos(pi*x1) - 0.6*pi*np.cos(3*pi*x1) \
			+ 0.4*pi*np.cos(pi*x2)) - mu*(-0.2*pi**2* \
			np.sin(pi*x1)*np.sin(pi*x2) - 3.6*pi**2* \
			np.sin(3*pi*x1) - 0.4*pi**2*np.cos(pi*x1)* \
			np.cos(pi*x2))*(0.3*np.sin(3*pi*x1) + 0.3* \
			np.cos(pi*x1)*np.cos(pi*x2) + 0.3* \
			np.cos(pi*x2) + 2.0) - mu*(-0.2*pi**2* \
			np.sin(pi*x1)*np.sin(pi*x2) - 0.4*pi**2* \
			np.sin(pi*x2) - 0.4*pi**2*np.cos(pi*x1)* \
			np.cos(pi*x2))*(0.3*np.sin(pi*x2) + 0.3* \
			np.cos(pi*x1)*np.cos(pi*x2) + 0.3* \
			np.cos(pi*x1) + 2.0) - mu*(0.3*pi**2* \
			np.sin(pi*x1)*np.sin(pi*x2) - 0.3*pi**2* \
			np.cos(pi*x1)*np.cos(pi*x2) - 0.3*pi**2* \
			np.cos(pi*x1))*(0.3*np.sin(pi*x2) + 0.3* \
			np.cos(pi*x1)*np.cos(pi*x2) + 0.3* \
			np.cos(pi*x1) + 2.0) - mu*(0.3*pi**2* \
			np.sin(pi*x1)*np.sin(pi*x2) - 0.3*pi**2* \
			np.cos(pi*x1)*np.cos(pi*x2) - 0.3*pi**2* \
			np.cos(pi*x2))*(0.3*np.sin(3*pi*x1) + 0.3* \
			np.cos(pi*x1)*np.cos(pi*x2) + 0.3* \
			np.cos(pi*x2) + 2.0) + (-0.3*pi* \
			np.sin(pi*x1)*np.cos(pi*x2) + 0.9*pi* \
			np.cos(3*pi*x1))*((4.0*(0.15* \
			np.sin(3*pi*x1) + 0.15*np.cos(pi*x1)* \
			np.cos(pi*x2) + 0.15*np.cos(pi*x2) + 1)**2 \
			+ 4.0*(0.15*np.sin(pi*x2) + 0.15* \
			np.cos(pi*x1)*np.cos(pi*x2) + 0.15* \
			np.cos(pi*x1) + 1)**2)*(0.05*np.sin(pi*x1) \
			+ 0.05*np.cos(pi*x1)*np.cos(pi*x2) - 0.1* \
			np.cos(pi*x2) + 0.5) + 1.0*np.sin(pi*x2) + \
			0.5*np.cos(pi*x1)*np.cos(pi*x2) + 1.0* \
			np.cos(2*pi*x1) + 10.0 + (1.0*np.sin(pi*x2) \
			+ 0.5*np.cos(pi*x1)*np.cos(pi*x2) + 1.0* \
			np.cos(2*pi*x1) + 10.0)/(gamma - 1)) + (-0.3* \
			pi*np.sin(pi*x2)*np.cos(pi*x1) + 0.3*pi* \
			np.cos(pi*x2))*((4.0*(0.15*np.sin(3*pi*x1) + \
			0.15*np.cos(pi*x1)*np.cos(pi*x2) + 0.15* \
			np.cos(pi*x2) + 1)**2 + 4.0*(0.15* \
			np.sin(pi*x2) + 0.15*np.cos(pi*x1)* \
			np.cos(pi*x2) + 0.15*np.cos(pi*x1) + 1)**2) \
			*(0.05*np.sin(pi*x1) + 0.05*np.cos(pi*x1)* \
			np.cos(pi*x2) - 0.1*np.cos(pi*x2) + 0.5) + 1.0 \
			*np.sin(pi*x2) + 0.5*np.cos(pi*x1)* \
			np.cos(pi*x2) + 1.0*np.cos(2*pi*x1) + 10.0 \
			+ (1.0*np.sin(pi*x2) + 0.5*np.cos(pi*x1)* \
			np.cos(pi*x2) + 1.0*np.cos(2*pi*x1) + 10.0) \
			/(gamma - 1)) + (0.3*np.sin(3*pi*x1) + 0.3* \
			np.cos(pi*x1)*np.cos(pi*x2) + 0.3* \
			np.cos(pi*x2) + 2.0)*((4.0*(-0.3*pi* \
			np.sin(pi*x1)*np.cos(pi*x2) - 0.3*pi* \
			np.sin(pi*x1))*(0.15*np.sin(pi*x2) + 0.15* \
			np.cos(pi*x1)*np.cos(pi*x2) + 0.15* \
			np.cos(pi*x1) + 1) + 4.0*(-0.3*pi* \
			np.sin(pi*x1)*np.cos(pi*x2) + 0.9* \
			pi*np.cos(3*pi*x1))*(0.15*np.sin(3*pi*x1) \
			+ 0.15*np.cos(pi*x1)*np.cos(pi*x2) + 0.15* \
			np.cos(pi*x2) + 1))*(0.05*np.sin(pi*x1) + \
			0.05*np.cos(pi*x1)*np.cos(pi*x2) - 0.1* \
			np.cos(pi*x2) + 0.5) + (-0.05*pi* \
			np.sin(pi*x1)*np.cos(pi*x2) + 0.05*pi* \
			np.cos(pi*x1))*(4.0*(0.15*np.sin(3*pi*x1) + \
			0.15*np.cos(pi*x1)*np.cos(pi*x2) + 0.15* \
			np.cos(pi*x2) + 1)**2 + 4.0*(0.15* \
			np.sin(pi*x2) + 0.15*np.cos(pi*x1)* \
			np.cos(pi*x2) + 0.15*np.cos(pi*x1) + 1)**2) \
			- 0.5*pi*np.sin(pi*x1)*np.cos(pi*x2) - \
			2.0*pi*np.sin(2*pi*x1) + (-0.5*pi* \
			np.sin(pi*x1)*np.cos(pi*x2) - 2.0*pi* \
			np.sin(2*pi*x1))/(gamma - 1)) + (0.3* \
			np.sin(pi*x2) + 0.3*np.cos(pi*x1)* \
			np.cos(pi*x2) + 0.3*np.cos(pi*x1) + \
			2.0)*((4.0*(-0.3*pi*np.sin(pi*x2)* \
			np.cos(pi*x1) - 0.3*pi*np.sin(pi*x2))* \
			(0.15*np.sin(3*pi*x1) + 0.15*np.cos(pi*x1)* \
			np.cos(pi*x2) + 0.15*np.cos(pi*x2) + 1) + \
			4.0*(-0.3*pi*np.sin(pi*x2)*np.cos(pi*x1) \
			+ 0.3*pi*np.cos(pi*x2))*(0.15*np.sin(pi*x2) \
			+ 0.15*np.cos(pi*x1)*np.cos(pi*x2) + 0.15* \
			np.cos(pi*x1) + 1))*(0.05*np.sin(pi*x1) + \
			0.05*np.cos(pi*x1)*np.cos(pi*x2) - 0.1* \
			np.cos(pi*x2) + 0.5) + (-0.05*pi* \
			np.sin(pi*x2)*np.cos(pi*x1) + 0.1*pi* \
			np.sin(pi*x2))*(4.0*(0.15*np.sin(3*pi*x1) + \
			0.15*np.cos(pi*x1)*np.cos(pi*x2) + 0.15* \
			np.cos(pi*x2) + 1)**2 + 4.0*(0.15* \
			np.sin(pi*x2) + 0.15*np.cos(pi*x1)* \
			np.cos(pi*x2) + 0.15*np.cos(pi*x1) + 1)**2) \
			- 0.5*pi*np.sin(pi*x2)*np.cos(pi*x1) + \
			1.0*pi*np.cos(pi*x2) + (-0.5*pi* \
			np.sin(pi*x2)*np.cos(pi*x1) + 1.0*pi* \
			np.cos(pi*x2))/(gamma - 1)) - pi**2*kappa* \
			(-0.2*(0.5*np.sin(pi*x1)*np.cos(pi*x2) + 2.0* \
			np.sin(2*pi*x1))*(np.sin(pi*x1)* \
			np.cos(pi*x2) - np.cos(pi*x1))/(0.1* \
			np.sin(pi*x1) + 0.1*np.cos(pi*x1)* \
			np.cos(pi*x2) - 0.2*np.cos(pi*x2) + 1.0) \
			+ (0.02*(np.sin(pi*x1)*np.cos(pi*x2) - \
			np.cos(pi*x1))**2/(0.1*np.sin(pi*x1) + 0.1 \
			*np.cos(pi*x1)*np.cos(pi*x2) - 0.2* \
			np.cos(pi*x2) + 1.0) + 0.1*np.sin(pi*x1) + \
			0.1*np.cos(pi*x1)*np.cos(pi*x2))*(1.0* \
			np.sin(pi*x2) + 0.5*np.cos(pi*x1)* \
			np.cos(pi*x2) + 1.0*np.cos(2*pi*x1) + 10.0) \
			/(0.1*np.sin(pi*x1) + 0.1*np.cos(pi*x1)* \
			np.cos(pi*x2) - 0.2*np.cos(pi*x2) + 1.0) - \
			0.5*np.cos(pi*x1)*np.cos(pi*x2) - 4.0* \
			np.cos(2*pi*x1))/(R*(0.1*np.sin(pi*x1) + \
			0.1*np.cos(pi*x1)*np.cos(pi*x2) - 0.2* \
			np.cos(pi*x2) + 1.0)) - pi**2*kappa*(-2* \
			(0.5*np.sin(pi*x2)*np.cos(pi*x1) - 1.0* \
			np.cos(pi*x2))*(0.1*np.cos(pi*x1) - 0.2)* \
			np.sin(pi*x2)/(0.1*np.sin(pi*x1) + 0.1* \
			np.cos(pi*x1)*np.cos(pi*x2) - 0.2* \
			np.cos(pi*x2) + 1.0) + ((0.2*np.cos(pi*x1) \
			- 0.4)*np.sin(pi*x2)**2/(0.1*np.sin(pi*x1) \
			+ 0.1*np.cos(pi*x1)*np.cos(pi*x2) - 0.2* \
			np.cos(pi*x2) + 1.0) + np.cos(pi*x2))*(0.1* \
			np.cos(pi*x1) - 0.2)*(1.0*np.sin(pi*x2) + 0.5 \
			*np.cos(pi*x1)*np.cos(pi*x2) + 1.0* \
			np.cos(2*pi*x1) + 10.0)/(0.1*np.sin(pi*x1) \
			+ 0.1*np.cos(pi*x1)*np.cos(pi*x2) - 0.2* \
			np.cos(pi*x2) + 1.0) - 1.0*np.sin(pi*x2) - \
			0.5*np.cos(pi*x1)*np.cos(pi*x2))/(R*(0.1* \
			np.sin(pi*x1) + 0.1*np.cos(pi*x1)* \
			np.cos(pi*x2) - 0.2*np.cos(pi*x2) + 1.0))
		# End of generated code


		return S_rho, S_rhou, S_rhov, S_rhoE

'''
-------------------
Boundary conditions
-------------------
These classes inherit from either the BCWeakRiemann or BCWeakPrescribed
classes. See those parent classes for detailed comments of attributes
and methods. Information specific to the corresponding child classes can be
found below. These classes should correspond to the BCType enum members
above.
'''


'''
------------------------
Numerical flux functions
------------------------
These classes inherit from the ConvNumFluxBase or DiffNumFluxBase class.
See ConvNumFluxBase/DiffNumFluxBase for detailed comments of attributes
and methods. Information specific to the corresponding child classes can
be found below. These classes should correspond to the ConvNumFluxType
or DiffNumFluxType enum members above.
'''