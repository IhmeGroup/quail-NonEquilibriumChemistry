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
#       File : src/numerics/timestepping/tools.py
#
#       Contains helper functions for the stepper class.
#
# ------------------------------------------------------------------------ #
from abc import ABC, abstractmethod
import math
import numpy as np

from quail.general import StepperType

import quail.numerics.basis.tools as basis_tools
from quail.numerics.helpers import helpers
import quail.numerics.timestepping.stepper as stepper_defs

import quail.solver.tools as solver_tools


def set_stepper(params, U):
	'''
	Given the TimeStepper parameter, set the stepper object

	Inputs:
	-------
		params: list of parameters for solver
		U: solution vector for instantiaing stepper class [num_elems, nb, ns]

	Outputs:
	--------
	    stepper: instantiated stepper object
	'''
	time_stepper = params["TimeStepper"]

	if StepperType[time_stepper] == StepperType.FE:
		stepper = stepper_defs.FE(U)
	elif StepperType[time_stepper] == StepperType.RK4:
		stepper = stepper_defs.RK4(U)
	elif StepperType[time_stepper] == StepperType.LSRK4:
		stepper = stepper_defs.LSRK4(U)
	elif StepperType[time_stepper] == StepperType.SSPRK3:
		stepper = stepper_defs.SSPRK3(U)
	elif StepperType[time_stepper] == StepperType.SSPRK3_4S:
		stepper = stepper_defs.SSPRK3_4S(U)
	elif StepperType[time_stepper] == StepperType.LSSSPRK3:
		stepper = stepper_defs.LSSSPRK3(U)
	# If setting a splitting scheme select solvers for the splits
	elif StepperType[time_stepper] == StepperType.Strang:
		stepper = stepper_defs.Strang(U)
		stepper.set_split_schemes(params["OperatorSplittingExplicit"],
			params["OperatorSplittingImplicit"], U)
	elif StepperType[time_stepper] == StepperType.Simpler:
		stepper = stepper_defs.Simpler(U)
		stepper.set_split_schemes(params["OperatorSplittingExplicit"],
			params["OperatorSplittingImplicit"], U)
	elif StepperType[time_stepper] == StepperType.ODEIntegrator:
		stepper = stepper_defs.ODEIntegrator(U)
		stepper.set_ode_integrator(params["ODEScheme"], U)
	else:
		raise NotImplementedError("Time scheme not supported")
	return stepper

def set_source_treatment(physics):
	'''
	Allows user to define how source terms are treated in both splitting
	and ADERDG schemes. Can select 'Explicit' or 'Implicit' depending on
	the stiffness of the source term.

	Inputs:
	-------
		physics: physics object

	Outputs:
	--------
		Constructs explicit_sources and implicit_sources
	'''
	physics.explicit_sources = []
	physics.implicit_sources = []
	for source in physics.source_terms:
		if source.source_treatment == 'Explicit':
			physics.explicit_sources.append(source)
		elif source.source_treatment == 'Implicit':
			physics.implicit_sources.append(source)

def set_time_stepping_approach(stepper, params):
	'''
	Sets stepper.get_time_step method given input parameters

	Inputs:
	-------
		stepper: stepper object (e.g., FE, RK4, etc...)
		params: list of parameters for solver

	Outputs:
	--------
		get_time_step: method selected to calculate dt
		num_time_steps: number of time steps for the solution
	'''

	# Unpack time stepping settings
	cfl = params["CFL"]
	ramp_inputs = [params["InitialCFL"], params["FinalCFL"],
				   params["RampIterations"]]
	dt = params["TimeStepSize"]
	num_time_steps = params["NumTimeSteps"]
	tfinal = params["FinalTime"]
	stepper.tfinal = tfinal

	'''
	Hierarchy for cases goes:
		1. number of time steps and tfinal
		2. time step size and tfinal
		3. CFL number
		4. number of time steps and time step size
	'''
	if num_time_steps != None and tfinal != None:
		stepper.get_time_step = get_dt_from_num_time_steps
		stepper.num_time_steps = num_time_steps
	elif dt != None and tfinal != None:
		stepper.get_time_step = get_dt_from_timestepsize
		stepper.num_time_steps = math.ceil(tfinal/dt)
	elif cfl != None or all([val is not None for val in ramp_inputs]):
		stepper.get_time_step = get_dt_from_cfl
		stepper.num_time_steps = 1
	elif num_time_steps != None and tfinal == None:
		stepper.get_time_step = get_dt_from_timestepsize_and_numtimesteps
		stepper.num_time_steps = num_time_steps

def get_dt_from_num_time_steps(stepper, solver):
	'''
	Calculates dt from the specified number of time steps

	Inputs:
	-------
		stepper: stepper object (e.g., FE, RK4, etc...)
		solver: solver object (e.g., DG, ADERDG, etc...)

	Outputs:
	--------
		dt: time step for the solver
	'''
	num_time_steps = stepper.num_time_steps
	tfinal = solver.params["FinalTime"]
	stepper.tfinal = tfinal
	time = solver.time

	# only needs to be set once per simulation
	if stepper.dt == 0.0 and num_time_steps != 0:
		return (tfinal - time) / num_time_steps
	elif num_time_steps == 0:
		return 1.0
	else:
		return stepper.dt


def get_dt_from_timestepsize(stepper, solver):
	'''
	Sets dt directly based on input deck specification of
	params["TimeStepSize"].

	Inputs:
	-------
		stepper: stepper object (e.g., FE, RK4, etc...)
		solver: solver object (e.g., DG, ADERDG, etc...)

	Outputs:
	--------
		dt: time step for the solver
	'''
	time = solver.time
	dt = solver.params["TimeStepSize"]
	tfinal = solver.params["FinalTime"]
	stepper.tfinal = tfinal

	# logic to ensure final time step yields FinalTime
	if time + dt < tfinal:
		return dt
	else:
		return tfinal - time


def get_dt_from_cfl(stepper, solver):
	'''
	Calculates dt using a specified CFL number, or by calculating based
	on ramp inputs. Updates at everytime step to ensure solution remains
	within the CFL bound.

	With ramping, the CFL can be ramped up or down, depending on the
	relative values of the initial and final CFL specified by the user.
	Once the ramp iterations have been exceeded, CFL will be held constant
	at its final ramp value.

	Inputs:
	-------
		stepper: stepper object (e.g., FE, RK4, etc...)
		solver: solver object (e.g., DG, ADERDG, etc...)

	Outputs:
	--------
		dt: time step for the solver
	'''
	mesh = solver.mesh
	ndims = mesh.ndims

	physics = solver.physics
	U = solver.state_coeffs

	time = solver.time
	tfinal = solver.params["FinalTime"]
	stepper.tfinal = tfinal
	cfl = solver.params["CFL"]
	if cfl is None:
		itime_initial = solver.itime_initial
		itime = solver.itime
		ramp_iters = solver.params["RampIterations"]
		cfl_initial = solver.params["InitialCFL"]
		cfl_final = solver.params["FinalCFL"]
		cfl = (
			(cfl_final - cfl_initial) / ramp_iters
			* (itime - itime_initial) + cfl_initial
		)
		# don't go beyond initial or final bounds
		# handles ramping up or down
		cfl = np.minimum(cfl, np.maximum(cfl_initial, cfl_final))
		cfl = np.maximum(cfl, np.minimum(cfl_initial, cfl_final))
		print("%d: CFL = %g - CFL_i = %g - CFL_f = %g" % (
			  itime - itime_initial + 1, cfl, cfl_initial, cfl_final))

	elem_helpers = solver.elem_helpers
	vol_elems = elem_helpers.vol_elems
	a = np.zeros([mesh.num_elems, U.shape[1], 1])
	basis_val = solver.elem_helpers.basis_val

	# Interpolate state at quad points
	Uq = helpers.evaluate_state(U, basis_val,
			skip_interp=solver.basis.skip_interp) # [ne, nq, ns]

	# Calculate max wavespeed
	a = physics.compute_variable("MaxWaveSpeed", Uq,
			flag_non_physical=True)
	# Calculate the dt for each element
	dt_elems = cfl*vol_elems**(1./ndims)/a

	# take minimum to set appropriate dt
	dt = np.min(dt_elems)

	# logic to ensure final time step yields FinalTime
	if time + dt < tfinal:
		stepper.num_time_steps += 1
		return dt
	else:
		return tfinal - time

def get_dt_from_timestepsize_and_numtimesteps(stepper, solver):
	'''
	Sets dt directly based on input deck specification of
	params["TimeStepSize"].

	Inputs:
	-------
		stepper: stepper object (e.g., FE, RK4, etc...)
		solver: solver object (e.g., DG, ADERDG, etc...)

	Outputs:
	--------
		dt: time step for the solver
	'''
	time = solver.time
	dt = solver.params["TimeStepSize"]
	num_time_steps = stepper.num_time_steps

	tfinal = dt*num_time_steps
	stepper.tfinal = tfinal
	# logic to ensure final time step yields FinalTime
	if time + dt < tfinal:
		return dt
	else:
		return tfinal - time
