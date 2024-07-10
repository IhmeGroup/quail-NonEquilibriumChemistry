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
#       File : src/numerics/limiting/positivitypreserving.py
#
#       Contains class definitions for positivity-preserving limiters.
#
# ------------------------------------------------------------------------ #
from abc import ABC, abstractmethod
import numpy as np

from quail import errors
from quail import general

import quail.meshing.tools as mesh_tools

from quail.numerics.helpers import helpers
from quail.numerics.limiting import base


POS_TOL = 1.e-10


def trunc(a, decimals=8):
	'''
	This function truncates a float to a specified decimal place.
	Adapted from:
	https://stackoverflow.com/questions/42021972/
	truncating-decimal-digits-numpy-array-of-floats

	Inputs:
	-------
		a: value(s) to truncate
		decimals: truncated decimal place

	Outputs:
	--------
		truncated float
	'''
	return np.trunc(a*10**decimals)/(10**decimals)


class PositivityPreserving(base.LimiterBase):
	'''
	This class corresponds to the positivity-preserving limiter for the
	Euler equations. It inherits from the LimiterBase class. See
	See LimiterBase for detailed comments of attributes and methods. See
	the following references:
		[1] X. Zhang, C.-W. Shu, "On positivity-preserving high order
		discontinuous Galerkin schemes for compressible Euler equations
		on rectangular meshes," Journal of Computational Physics.
		229:8918–8934, 2010.
		[2] C. Wang, X. Zhang, C.-W. Shu, J. Ning, "Robust high order
		discontinuous Galerkin schemes for two-dimensional gaseous
		detonations," Journal of Computational Physics, 231:653-665, 2012.

	Attributes:
	-----------
	var_name1: str
		name of first variable involved in limiting (density)
	var_name2: str
		name of second variable involved in limiting (pressure)
	elem_vols: numpy array
		element volumes
	basis_val_elem_faces: numpy array
		stores basis values for element and faces
	quad_wts_elem: numpy array
		quadrature points for element
	djac_elems: numpy array
		stores Jacobian determinants for each element
	'''
	COMPATIBLE_PHYSICS_TYPES = [general.PhysicsType.Euler,
								general.PhysicsType.NavierStokes]

	def __init__(self, physics_type):
		super().__init__(physics_type)
		self.var_name1 = "Densities"
		self.var_name2 = "Pressure"
		self.elem_vols = np.zeros(0)
		self.basis_val_elem_faces = np.zeros(0)
		self.quad_wts_elem = np.zeros(0)
		self.djac_elems = np.zeros(0)

	def precompute_helpers(self, solver):
		# Unpack
		elem_helpers = solver.elem_helpers
		int_face_helpers = solver.int_face_helpers
		self.elem_vols, _ = mesh_tools.element_volumes(solver.mesh, solver)

		# Basis values in element interior and on faces
		if not solver.basis.skip_interp:
			basis_val_faces = int_face_helpers.faces_to_basisL.copy()
			bshape = basis_val_faces.shape
			basis_val_faces.shape = (bshape[0]*bshape[1], bshape[2])
			self.basis_val_elem_faces = np.vstack((elem_helpers.basis_val,
					basis_val_faces))
		else:
			self.basis_val_elem_faces = elem_helpers.basis_val

		# Jacobian determinant
		self.djac_elems = elem_helpers.djac_elems

		# Element quadrature weights
		self.quad_wts_elem = elem_helpers.quad_wts

	def limit_solution(self, solver, Uc):
		# Unpack
		physics = solver.physics
		elem_helpers = solver.elem_helpers
		int_face_helpers = solver.int_face_helpers
		basis = solver.basis

		djac = self.djac_elems

		# Interpolate state at quadrature points over element and on faces
		U_elem_faces = helpers.evaluate_state(Uc, self.basis_val_elem_faces,
				skip_interp=basis.skip_interp)
		nq_elem = self.quad_wts_elem.shape[0]
		U_elem = U_elem_faces[:, :nq_elem, :]
		ne = self.elem_vols.shape[0]
		# Average value of state
		U_bar = helpers.get_element_mean(U_elem, self.quad_wts_elem, djac,
				self.elem_vols)

		# Density and pressure from averaged state
		rho_bar = physics.compute_variable(self.var_name1, U_bar)
		p_bar = physics.compute_variable(self.var_name2, U_bar)

		if np.any(rho_bar < -POS_TOL):
			raise errors.NotPhysicalError("Negative element species partial density.")
		elif np.any(p_bar < -POS_TOL):
			raise errors.NotPhysicalError("Negative element pressure.")

		# Ignore divide-by-zero
		np.seterr(divide='ignore')

		''' Limit density '''
		# Compute density at quadrature points
		rho_elem_faces = physics.compute_variable(self.var_name1,
				U_elem_faces)
		# Check if limiting is needed
		theta = np.abs((rho_bar - POS_TOL)/(rho_bar - rho_elem_faces))
		# Truncate theta1; otherwise, can get noticeably different
		# results across machines, possibly due to poor conditioning in its
		# calculation
		theta1 = trunc(np.minimum(1., np.min(theta, axis=1, keepdims=True)))

		srho = physics.get_state_slice(self.var_name1)
		# Get IDs of elements that need limiting
		elem_IDs = np.unique(np.where(theta1 < 1.)[0])
		if len(elem_IDs) != 0:
			# Modify density coefficients
			if basis.MODAL_OR_NODAL == general.ModalOrNodal.Nodal:
				Uc[elem_IDs, :, srho] = theta1[elem_IDs]*Uc[elem_IDs, :, srho] \
						+ (1. - theta1[elem_IDs])*rho_bar[elem_IDs]
			elif basis.MODAL_OR_NODAL == general.ModalOrNodal.Modal:
				Uc[elem_IDs, :, srho] *= theta1[elem_IDs]
				Uc[elem_IDs, 0, srho] += (1. - theta1[elem_IDs, 0, :])*rho_bar[
						elem_IDs, 0, :]
			else:
				raise NotImplementedError

			# Intermediate limited solution
			U_elem_faces = helpers.evaluate_state(Uc,
					self.basis_val_elem_faces,
					skip_interp=basis.skip_interp)

		''' Limit pressure '''
		# Compute pressure at quadrature points
		p_elem_faces = physics.compute_variable(self.var_name2, U_elem_faces)
		theta = np.ones(p_elem_faces.shape)
		# Indices where pressure is negative
		negative_p_indices = np.where(p_elem_faces < 0.)
		elem_IDs = negative_p_indices[0]
		i_neg_p  = negative_p_indices[1]

		theta[elem_IDs, i_neg_p, :] = (p_bar[elem_IDs, 0, :] - POS_TOL) / (
				p_bar[elem_IDs, 0, :] - p_elem_faces[elem_IDs, i_neg_p, :])

		# Truncate theta2; otherwise, can get noticeably different
		# results across machines, possibly due to poor conditioning in its
		# calculation
		theta2 = trunc(np.min(theta, axis=1, keepdims=True))
		# Get IDs of elements that need limiting
		elem_IDs = np.unique(np.where(theta2 < 1.)[0])
		if len(elem_IDs) != 0:
			# Modify coefficients
			if basis.MODAL_OR_NODAL == general.ModalOrNodal.Nodal:
				Uc[elem_IDs] = np.einsum('imk, ijk -> ijk', theta2[elem_IDs],
						Uc[elem_IDs]) + np.einsum('imk, ijk -> ijk', 1 - theta2[
						elem_IDs], U_bar[elem_IDs])
			elif basis.MODAL_OR_NODAL == general.ModalOrNodal.Modal:
				Uc[elem_IDs] *= np.expand_dims(theta2[elem_IDs], axis=2)
				Uc[elem_IDs, [0]] += np.einsum('im, ijk -> ik', 1 - theta2[
						elem_IDs], U_bar[elem_IDs])
			else:
				raise NotImplementedError

		np.seterr(divide='warn')

		return Uc # [ne, nq, ns]


class PositivityPreservingEnergy(PositivityPreserving):
	'''
    Class: PPLimiter
    ------------------
    This class contains information about the positivity preserving limiter
    '''

	COMPATIBLE_PHYSICS_TYPES = [general.PhysicsType.Euler,
								general.PhysicsType.NavierStokes,
								general.PhysicsType.Chemistry]

	def __init__(self, physics_type):
		'''
		Method: __init__
		-------------------
		Initializes PPLimiter object
		'''
		super().__init__(physics_type)
		self.var_name1 = "Densities"
		self.var_name2 = "InternalEnergy"

	def limit_solution(self, solver, Uc):
		# Unpack
		physics = solver.physics
		basis = solver.basis

		djac = self.djac_elems

		# Interpolate state at quadrature points over element and on faces
		U_elem_faces = helpers.evaluate_state(Uc, self.basis_val_elem_faces,
				skip_interp=basis.skip_interp)
		nq_elem = self.quad_wts_elem.shape[0]
		U_elem = U_elem_faces[:, :nq_elem, :]

		# Average value of state
		U_bar = helpers.get_element_mean(U_elem, self.quad_wts_elem, djac,
				self.elem_vols)

		# Density and energy from averaged state
		srhoi = physics.get_state_slice(self.var_name1)
		rhoi_bar = physics.compute_variable(self.var_name1, U_bar)

		# srhoE = physics.get_state_slice(self.var_name2)
		rhoE_bar = physics.compute_variable(self.var_name2, U_bar)

		if np.any(rhoi_bar < -POS_TOL):
			raise errors.NotPhysicalError("Negative element species partial density.")
		elif np.any(rhoE_bar < -POS_TOL):
			raise errors.NotPhysicalError("Negative element internal energy.")

		# Ignore divide-by-zero
		np.seterr(divide='ignore')

		''' Limit partial densities '''
		# Compute partial densities
		rhoi_elem_faces = physics.compute_variable(self.var_name1, U_elem_faces)
		theta1 = np.abs((rhoi_bar-POS_TOL)/(rhoi_bar-rhoi_elem_faces))
		# Truncate theta2; otherwise, can get noticeably different
		# results across machines, possibly due to poor conditioning in its
		# calculation
		theta1 = trunc(np.minimum(1., np.min(theta1, axis=1, keepdims=True)))

		# Get IDs of elements that need limiting
		elem_IDs = np.unique(np.where(theta1 < 1.)[0])
		# Modify density coefficients
		if basis.MODAL_OR_NODAL == general.ModalOrNodal.Nodal:
			Uc[elem_IDs, :, srhoi] = theta1[elem_IDs]*Uc[elem_IDs, :,
					srhoi] + (1. - theta1[elem_IDs])*rhoi_bar[elem_IDs]
		elif basis.MODAL_OR_NODAL == general.ModalOrNodal.Modal:
			Uc[elem_IDs, :, srhoi] *= theta1[elem_IDs]
			Uc[elem_IDs, 0, srhoi] += (1. - theta1[elem_IDs])*rhoi_bar[elem_IDs]
		else:
			raise NotImplementedError

		if np.any(theta1 < 1.):
			U_elem_faces = helpers.evaluate_state(Uc,
					self.basis_val_elem_faces,
					skip_interp=basis.skip_interp)

		''' Limit energies '''
		# Compute energies
		rhoE_elem_faces = physics.compute_variable(self.var_name2, U_elem_faces)
		theta2 = np.abs((rhoE_bar-POS_TOL)/(rhoE_bar-rhoE_elem_faces))
		# Truncate theta2; otherwise, can get noticeably different
		# results across machines, possibly due to poor conditioning in its
		# calculation
		theta2 = trunc(np.minimum(1., np.min(theta2, axis=1, keepdims=True)))

		# Get IDs of elements that need limiting
		elem_IDs = np.unique(np.where(theta2 < 1.)[0])
		# Modify density coefficients
		if basis.MODAL_OR_NODAL == general.ModalOrNodal.Nodal:
			# Uc[elem_IDs, :, srhoE] = theta2[elem_IDs]*Uc[elem_IDs, :,
			# 		srhoE] + (1. - theta2[elem_IDs])*rhoE_bar[elem_IDs]
			Uc[elem_IDs] = (theta2[elem_IDs]*Uc[elem_IDs] +
				      (1. - theta2[elem_IDs])*U_bar[elem_IDs])
		elif basis.MODAL_OR_NODAL == general.ModalOrNodal.Modal:
			# Uc[elem_IDs, :, srhoE] *= theta2[elem_IDs]
			# Uc[elem_IDs, 0, srhoE] += (1. - theta2[elem_IDs])*rhoE_bar[elem_IDs]
			Uc[elem_IDs] *= theta2[elem_IDs]
			Uc[elem_IDs, 0, :] += (1. - theta2[elem_IDs, 0, :])*U_bar[elem_IDs, 0, :]
		else:
			raise NotImplementedError

		if np.any(theta2 < 1.):
			U_elem_faces = helpers.evaluate_state(Uc,
					self.basis_val_elem_faces,
					skip_interp=basis.skip_interp)

		np.seterr(divide='warn')

		return Uc # [ne, nq, ns]
