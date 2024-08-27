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
#       File : src/solver/tools.py
#
#       Contains additional methods (tools) for the DG solver class
#
# ------------------------------------------------------------------------ #
import numpy as np
import scipy
from scipy import optimize
import sys

from quail import general
import quail.numerics.basis.tools as basis_tools
from quail.numerics.helpers import helpers
import quail.solver.tools as solver_tools
import quail.numerics.limiting.tools as limiter_tools


def set_function_definitions(solver, params):
	'''
	This function sets the necessary functions for the given case
	dependent upon setter flags in the input deck (primarily for
	the diffusive flux definitions)

	Inputs:
	-------
		solver: solver object
		params: dict with solver parameters
	'''
	if solver.physics.diff_flux_fcn:
		solver.evaluate_gradient = helpers.evaluate_gradient
		solver.ref_to_phys_grad = helpers.ref_to_phys_grad
		solver.calculate_boundary_flux_integral_sum = \
			solver_tools.calculate_boundary_flux_integral_sum
	else:
		solver.evaluate_gradient = general.pass_function
		solver.ref_to_phys_grad = general.pass_function
		solver.calculate_boundary_flux_integral_sum = \
			general.zero_function


def calculate_volume_flux_integral(solver, elem_helpers, Fq):
	'''
	Calculates the volume flux integral for the DG scheme

	Inputs:
	-------
		solver: solver object
		elem_helpers: helpers defined in ElemHelpers
		Fq: flux array evaluated at the quadrature points [ne, nq, ns, ndims]

	Outputs:
	--------
		res_elem: calculated residual array
			[ne, nb, ns]
	'''
	quad_wts = elem_helpers.quad_wts # [nq, 1]
	basis_phys_grad_elems = elem_helpers.basis_phys_grad_elems
			# [ne, nq, nb, ndims]
	djac_elems = elem_helpers.djac_elems # [ne, nq, 1]

	# Calculate flux quadrature
	F_quad = np.einsum('ijkl, jm, ijm -> ijkl', Fq, quad_wts, djac_elems)
			# [ne, nq, ns, ndims]
	# Calculate residual
	res_elem = np.einsum('ijnl, ijkl -> ink', basis_phys_grad_elems, F_quad)
			# [ne, nb, ns]
	return res_elem # [ne, nb, ns]


def calculate_boundary_flux_integral(basis_val, quad_wts, Fq):
	'''
	Calculates the boundary flux integral for the DG scheme

	Inputs:
	-------
		basis_val: basis function for the interior element [nf, nq, nb]
		quad_wts: quadrature weights [nq, 1]
		Fq: flux array evaluated at the quadrature points [nf, nq, ns]

	Outputs:
	--------
		resB: residual contribution (from boundary face) [nf, nb, ns]
	'''
	# Calculate flux quadrature
	Fq_quad = np.einsum('ijk, jm -> ijk', Fq, quad_wts) # [nf, nq, ns]

	# Calculate residual
	resB = np.einsum('ijn, ijk -> ink', basis_val, Fq_quad) # [nf, nb, ns]

	return resB # [nf, nb, ns]


def calculate_boundary_flux_integral_sum(basis_ref_grad, quad_wts, Fq):
	'''
	Calculates the directional boundary flux integrals for diffusion fluxes

	Inputs:
	-------
		basis_ref_grad: evaluated gradient of the basis function in
			reference space [nq, nb, ndims]
		quad_wts: quadrature weights [nq, 1]
		Fq: Direction diffusion flux contribution [nf, nq, ns, ndims]

	Outputs:
	--------
		resB: residual contribution (from boundary face) [nf, nb, ns]
	'''

	# Calculate flux quadrature
	Fq_quad = np.einsum('ijkl, jm -> ijkl', Fq, quad_wts) # [nf, nq, ns, ndims]

	# Calculate residual
	resB = np.einsum('ijnl, ijkl -> ink', basis_ref_grad, Fq_quad)

	return resB # [nf, nb, ns]


def calculate_source_term_integral(elem_helpers, Sq):
	'''
	Calculates the source term volume integral for the DG scheme

	Inputs:
	-------
		elem_helpers: helpers defined in ElemHelpers
		Sq: source term array evaluated at the quadrature points [ne, nq, ns]

	Outputs:
	--------
		res_elem: calculated residual array (for volume integral of all elements)
		[ne, nb, ns]
	'''
	quad_wts = elem_helpers.quad_wts # [nq, 1]
	basis_val = elem_helpers.basis_val # [nq, nb]
	djac_elems = elem_helpers.djac_elems # [ne, nq, 1]

	# Calculate source term quadrature
	Sq_quad = np.einsum('ijk, jm, ijm -> ijk', Sq, quad_wts, djac_elems)
			# [ne, nq, ns]

	# Calculate residual
	res_elem = np.einsum('jn, ijk -> ink', basis_val, Sq_quad) # [ne, nb, ns]

	return res_elem # [ne, nb, ns]


def initialize_artificial_viscosity(solver):
	'''
	Initializes a solver subroutine for AV smoothing. Uses the same mesh and basis
	functions for approximating a solution as the larger flow solver. Used to solve
	the AV elliptic PDE-smoothing equation given by Eric Ching:
		Eric J. Ching, Yu Lv, Peter Gnoffo, Michael Barnhardt, Matthias Ihme, Shock
		capturing for discontinuous Galerkin methods with application to predicting
		heat transfer in hypersonic flows, Journal of Computational Physics.
	Inputs:
	-------
		solver: solver object
	Outputs:
	--------
		av_solver: solver initialized for the AV pde-smoothing equation
	'''
	# Initialize AV-smoothing physics and solver
	import quail.defaultparams as default_deck
	from quail.physics.scalar import scalar
	from quail.physics.navierstokes import navierstokes
	from quail.solver import DG

	# AV is only defined for Navier Stokes flow physics
	if not isinstance(solver.physics, navierstokes.NavierStokes):
		raise NotImplementedError

	# Extract mesh
	mesh = solver.mesh
	# Create physics object (using scalar diffusive physics)
	av_physics = scalar.ConstAdvDiffScalar2D(NDIMS=mesh.ndims)
	av_physics.set_conv_num_flux("LaxFriedrichs")
	av_physics.set_diff_num_flux("SIP")
	pparams = {"ConstXVelocity": 0., "ConstYVelocity": 0.,
		"DiffCoefficientX": (1/24)**2, "DiffCoefficientY": (1/24)**2}
	av_physics.set_physical_params(**pparams)
	# Initial condition (empty AV field)
	iparams = {"state": [0]}
	av_physics.set_IC(IC_type="Uniform", **iparams)

	# Boundary conditions
	boundary_names = mesh.boundary_groups.keys()
	av_physics.BCs = dict.fromkeys(boundary_names)
	for bname in boundary_names:
		# Check if this is a wall boundary by checking the class name - should
		# find a better way than this.
		if "Wall" in repr(solver.physics.BCs[bname]):
			# Set Dirichlet along walls
			av_physics.set_BC(bname, "StateAll", "Uniform", **{"state": [0]})
		else:
			# Neumann along other boundaries
			av_physics.set_BC(bname, "Extrapolate")

	# Numerics parameters (same as larger flow solver)
	numerics_params = default_deck.Numerics.copy()
	numerics_params["SolutionOrder"] = solver.order
	numerics_params["SolutionBasis"] = type(solver.basis).__name__
	# Timestepping parameters (default, not used)
	stepper_params = default_deck.TimeStepping.copy()
	# Output parameters (default, not used)
	output_params = default_deck.Output.copy()
	solver_params = {**stepper_params, **numerics_params, **output_params}
	solver_params["RestartFile"] = None
	# Create solver object
	av_solver = DG.DG(solver_params, av_physics, mesh)

	return av_solver


def calculate_artificial_viscosity(solver, mesh, shock_indicator=None, av_param=None):
	'''
	Calculates the basis function coefficients approximating the AV field across the
	flow domain. Uses the shock detector, Riemannian metric tensor, and PDE-smoothing
	equation given by Eric Ching:
		Eric J. Ching, Yu Lv, Peter Gnoffo, Michael Barnhardt, Matthias Ihme, Shock
		capturing for discontinuous Galerkin methods with application to predicting
		heat transfer in hypersonic flows, Journal of Computational Physics.
	Inputs:
	-------
		solver: solver object
		mesh: mesh object
		shock_indicator: the shock-indicator to be used for element evaluation
		av_param: av constant term
	Outputs:
	--------
		avc: basis function coefficients that approximate the smooth AV field
	'''
	# Unpack
	physics = solver.physics
	elem_helpers = solver.elem_helpers
	basis_val = elem_helpers.basis_val
	basis_phys_grad_elems = elem_helpers.basis_phys_grad_elems
	quad_pts = elem_helpers.quad_pts
	quad_wts = elem_helpers.quad_wts
	iMM_elems = elem_helpers.iMM_elems
	djacs = elem_helpers.djac_elems
	vols = elem_helpers.vol_elems

	# Set defaults if not provided
	p = solver.order
	if shock_indicator is None:
		shock_indicator = limiter_tools.ejc_shock_indicator
	if av_param is None:
		av_param = 0.25

	# Shock detection
	Uq = helpers.evaluate_state(solver.state_coeffs, basis_val)
	shock_elems = shock_indicator(physics, elem_helpers, Uq)
	# Mesh metric
	Nd = mesh.ndims; Nu = Nd+2
	h_bar = mesh.length_scale
	# Calculate max wavespeed
	U_bar = helpers.get_element_mean(Uq, quad_wts, djacs, vols)
	a_max = physics.compute_variable("MaxWaveSpeed", U_bar).flatten()

	# Compute elementwise-AV
	av0 = av_param*a_max*h_bar/np.maximum(1, p)*shock_elems
	# Extend to quadrature points
	nq = quad_wts.shape[0]
	av0 = np.expand_dims(av0, axis=(1, 2))
	av0 = np.repeat(av0, nq, axis=1)

	# Convert to coefficients (elementwise-constant solution only)
	# av0c = np.zeros_like(solver.av_solver.state_coeffs)
	# solver_tools.L2_projection(mesh, iMM_elems, solver.basis,
	# 	quad_pts, quad_wts, av0, av0c)
	# solver.av_solver.state_coeffs = av0c
	# return av0c

	# Update AV-smoothing solver
	av_solver = solver.av_solver
	av_physics = av_solver.physics
	sparams = {"eta_0": av0}
	av_physics.set_source(source_type="ArtificialViscosity", **sparams)
	# Make initial coefficients guess from prior solution
	avc = av_solver.state_coeffs
	# Solve the nonlinear PDE system to get the new smooth AV solution
	res = np.zeros_like(avc)
	def av_res(U, av_solver, res):
		dt = 1
		U = U.reshape(res.shape)
		res = av_solver.get_residual(U, res)
		dU = solver_tools.mult_inv_mass_matrix(mesh, av_solver, dt, res)

		# Return residual
		av_res = dU-U
		return av_res.flatten()

	sol = scipy.optimize.root(av_res, avc, args=(av_solver, res), method='krylov')
	avc = sol.x.reshape(res.shape)

	# Evaluate AV at quadrature points
	av = helpers.evaluate_state(avc, solver.elem_helpers.basis_val)
	# Create filter bounds
	av_max_elems = np.max(av, axis=1).flatten()
	S_high = np.maximum(av_max_elems, a_max*h_bar/np.maximum(1, p))
	S_low = 0.01*S_high
	# Extend to quadrature points
	S_high = np.repeat(S_high[:, None, None], nq, axis=1)
	S_low = np.repeat(S_low[:, None, None], nq, axis=1)
	# Filter smoothed AV
	av_low = av < S_low
	av[av_low] = 0
	av_high = av > S_high
	av[av_high] = 0.5*S_high[av_high] * (1+np.sin( np.pi/2*(2*av[av_high]-(S_high[av_high]+S_low[av_high]))/(S_high[av_high]-S_low[av_high]) ))

	# Project onto the basis state from the final quadrature point values
	solver_tools.L2_projection(mesh, iMM_elems, solver.basis,
		quad_pts, quad_wts, av, avc)
	av_solver.state_coeffs = avc

	return avc


def calculate_artificial_viscosity_flux(mesh, physics, Uq, gUq, av, IDs=None):
	'''
	Calculates artificial viscosity flux given by Eric Ching:
		Eric J. Ching, Yu Lv, Peter Gnoffo, Michael Barnhardt, Matthias Ihme, Shock
		capturing for discontinuous Galerkin methods with application to predicting
		heat transfer in hypersonic flows, Journal of Computational Physics.
	Inputs:
	-------
		mesh: mesh object
		physics: physics object
		Uq: flow state evaluated at points
		gUq: state gradient evaluated at points
		av: artificial viscosity field evaluated at points
		IDs: the node IDs used for the current residual term
	Outputs:
	--------
		F: the artificial viscosity flux
	'''
	if IDs is None:
		# Slice over all elements
		IDs = np.s_[:]

	# Mesh metric
	h = mesh.inv_metric_tensor[IDs]
	h_bar = mesh.length_scale[IDs]

	# Extend dimensions
	h_bar = np.expand_dims(h_bar, axis=(1, 2))

	# Apply enthalpy-preservation correction to gradient
	irhoE = physics.get_state_index("Energies")
	dP = physics.compute_pressure_gradient(Uq, gUq)
	gUq[:, :, irhoE, :] += dP

	# Compute AV flux
	F = (av/h_bar)[..., None] * np.einsum('ilm,ijkm->ijkl', h, gUq)

	return F # [ne, nq, ns, ndims]


def calculate_artificial_viscosity_integral(physics, elem_helpers, Uc, av_param, p):
	'''
	Calculates the artificial viscosity volume integral, given in:
		Hartmann, R. and Leicht, T, "Higher order and adaptive DG methods for
		compressible flows", p. 92, 2013.

	Inputs:
	-------
		physics: physics object
		elem_helpers: helpers defined in ElemHelpers
		Uc: state coefficients of each element
		av_param: artificial viscosity parameter
		p: solution basis order

	Outputs:
	--------
		res_elem: artificial viscosity residual array for all elements
		[ne, nb, ns]
	'''
	# Unpack
	quad_wts = elem_helpers.quad_wts # [nq, 1]
	basis_phys_grad_elems = elem_helpers.basis_phys_grad_elems
			# [ne, nq, nb, dim]
	basis_val = elem_helpers.basis_val # [nq, nb]
	djac_elems = elem_helpers.djac_elems # [ne, nq, 1]
	vol_elems = elem_helpers.vol_elems # [ne]
	ndims = basis_phys_grad_elems.shape[3]

	# Evaluate solution at quadrature points
	Uq = helpers.evaluate_state(Uc, basis_val)
	# Evaluate solution gradient at quadrature points
	grad_Uq = np.einsum('ijnl, ink -> ijkl', basis_phys_grad_elems, Uc)
	# Compute pressure
	pressure = physics.compute_additional_variable("Pressure", Uq,
			flag_non_physical=False)[:, :, 0]
	# For Euler equations, use pressure as the smoothness variable
	if physics.PHYSICS_TYPE == general.PhysicsType.Euler:
		# Compute pressure gradient
		grad_p = physics.compute_pressure_gradient(Uq, grad_Uq)
		# Compute its magnitude
		norm_grad_p = np.linalg.norm(grad_p, axis = 2)
		# Calculate smoothness switch
		f = norm_grad_p / (pressure + 1e-12)
	# For everything else, use the first solution variable
	else:
		U0 = Uq[:, :, 0]
		grad_U0 = grad_Uq[:, :, 0]
		norm_grad_U0 = np.linalg.norm(grad_U0, axis = 2)
		# Calculate smoothness switch
		f =  norm_grad_U0 / (U0 + 1e-12)

	# Compute s_k
	s = np.zeros((Uc.shape[0], ndims))
	# Loop over dimensions
	for k in range(ndims):
		# Loop over number of faces per element
		for i in range(elem_helpers.normals_elems.shape[1]):
			# Integrate normals
			s[:, k] += np.einsum('jx, ij -> i', elem_helpers.face_quad_wts,
					np.abs(elem_helpers.normals_elems[:, i, :, k]))
		s[:, k] = 2 * vol_elems / s[:, k]
	# Compute h_k (the length scale in the kth direction)
	h = np.empty_like(s)
	# Loop over dimensions
	for k in range(ndims):
		h[:, k] = s[:, k] * (vol_elems / np.prod(s, axis=1))**(1/3)
	# Scale with polynomial order
	h_tilde = h / (p + 1)
	# Compute dissipation scaling
	epsilon = av_param *  np.einsum('ij, il -> ijl', f, h_tilde**3)
	# Calculate integral, with state coeffs factored out
	integral = np.einsum('ijm, ijpm, ijnm, jx, ijx -> ipn', epsilon,
				basis_phys_grad_elems, basis_phys_grad_elems, quad_wts,
				djac_elems)
	# Calculate residual
	res_elem = np.einsum('ipn, ipk -> ink', integral, Uc)

	return res_elem # [ne, nb, ns]


def calculate_dRdU(elem_helpers, Sjac):
	'''
	Helper function for ODE solvers that calculates the derivative of
	the source term integral with respect to the solution state.

	Inputs:
	-------
		elem_helpers: object containing precomputed element helpers
		Sjac: element source term Jacobian [ne, nq, ns, ns]

	Outputs:
	--------
		dRdU: derivative of the source term integral
			[ne, nb, nb, ns, ns]
	'''
	quad_wts = elem_helpers.quad_wts
	basis_val = elem_helpers.basis_val
	djac_elems = elem_helpers.djac_elems

	a = np.einsum('eijk, il, eil -> eijk', Sjac, quad_wts, djac_elems)

	return np.einsum('bq, ql, eqts -> eblts', basis_val.transpose(),
			basis_val, a)
		# [ne, nb, nb, ns, ns]


def mult_inv_mass_matrix(mesh, solver, dt, res):
	'''
	Multiplies the residual array with the inverse mass matrix

	Inputs:
		mesh: mesh object
		solver: solver object (e.g., DG, ADER-DG, etc...)
		dt: time step
		res: residual array

	Outputs:
		U: solution array
	'''
	physics = solver.physics
	iMM_elems = solver.elem_helpers.iMM_elems

	return dt*np.einsum('ijk, ikl -> ijl', iMM_elems, res)


def L2_projection(mesh, iMM, basis, quad_pts, quad_wts, f, U):
	'''
	Performs an L2 projection

	Inputs:
	-------
		mesh: mesh object
		iMM: space-time inverse mass matrix
		basis: basis object
		quad_pts: quadrature coordinates in reference space
		quad_wts: quadrature weights
		f: array of values to be projected from

	Outputs:
	--------
		U: array of values to be projected to
	'''
	if basis.basis_val.shape[0] != quad_wts.shape[0]:
		basis.get_basis_val_grads(quad_pts, get_val=True)

	for elem_ID in range(U.shape[0]):
		djac, _, _ = basis_tools.element_jacobian(mesh, elem_ID, quad_pts,
				get_djac=True)
		rhs = np.matmul(basis.basis_val.transpose(),
				f[elem_ID, :, :]*quad_wts*djac) # [nb, ns]

		U[elem_ID, :, :] = np.matmul(iMM[elem_ID], rhs)


def interpolate_to_nodes(f, U):
	'''
	Interpolates directly to the nodes of the element

	Inputs:
	-------
		f: array of values to be interpolated from

	Outputs:
	--------
		U: array of values to be interpolated onto
	'''
	U[:, :, :] = f


def get_ip_eta(mesh, order):
	i = order

	if i > 8:
		i = 8
	etas = np.array([1., 4., 12., 12., 20., 30., 35., 45., 50.])

	return etas[i] * mesh.gbasis.NFACES


def update_progress(progress):
	'''
	Displays or updates a console progress bar.
	Accepts a float between 0 and 1. Any int will be converted to a float.
	A value under 0 represents a 'halt'.
	A value at 1 or bigger represents 100%.

	Inputs:
	-------
		progress: value representing the progress, scaled from 0 to 1
	'''
	# Length of the progress bar
	bar_length = 55

	status = ""
	# Convert ints
	if isinstance(progress, int):
		progress = float(progress)
	# Make sre it's a number
	if not isinstance(progress, float):
		progress = 0
		status = "error: progress var must be float\r\n"
	# Less than 0 'halts' the progress
	if progress < 0:
		progress = 0
		status = "Halt...\r\n"
	# Cap the progress at 100%
	if progress >= 1:
		progress = 1
		status = "Done...\r\n"

	# Compute number of blocks
	block = int(round(bar_length*progress))
	# Figure out the color
	if progress < .25:
		color = '\033[0;31m' # Dark red
	elif progress < .5:
		color = '\033[1;31m' # Light red
	elif progress < .75:
		color = '\033[0;33m' # Yellow
	elif progress < 1:
		color = '\033[0;32m' # Dark green
	else:
		color = '\033[1;32m' # Light green
	reset_color = '\033[0m'
	# Write out the text
	text = color + '\rPercent: [{0}] {1}% {2}'.format( "#"*block + "-"*(bar_length-block),
			int(round(progress*100)), status) + reset_color
	sys.stdout.write(text)
	sys.stdout.flush()
