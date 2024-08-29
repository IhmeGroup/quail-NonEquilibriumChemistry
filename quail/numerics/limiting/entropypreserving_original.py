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

import quail.numerics.helpers.helpers as helpers
import quail.numerics.limiting.base as base


POS_TOL = 1.e-10
POS_TOL2 = 1.e-10

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



class EntropyPreserving(base.LimiterBase):
	'''
	This class corresponds to the positivity-preserving limiter for the
	Euler equations. It inherits from the LimiterBase class. With the original method,the entropy with the polynomai linterpolation can be nan
	due to the negative pressure in the log function. Therefore, we put the threshold value at 1.e-10 for pressure. 
	See LimiterBase for detailed comments of attributes and methods. See
	the following references:
		[1] Invariant-region-preserving DG methods for multi-dimensional 
		hyperbolic conservation law systems, with an application to compressible Euler equations

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
	COMPATIBLE_PHYSICS_TYPES = [general.PhysicsType.Euler, \
		general.PhysicsType.NavierStokes]

	def __init__(self, physics_type): 
		super().__init__(physics_type)
		self.var_name1 = "Density"
		self.var_name2 = "Pressure"
		self.var_name3 = "Entropy"
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

	def limit_solution(self, solver, Uc,rho_n,press_n):
    	# Uc is a conservative variables vector
		# limit conservative variables with the convexity of p, rho, and entropy
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

		# entropy_n = physics.compute_variable(self.var_name3,U_n) #[ne,3,1]
		entropy_n = np.log(np.maximum(press_n,1.e-10)/np.power(rho_n,1.4)) # gamma is hard coded
		rhos_n = entropy_n
		# is_nan = np.isnan(entropy_n)
		# nan_indecies = np.where(is_nan)
		min_entropy_incell = np.min(rhos_n,axis=1) #[ne,1] 
		min_entropy = np.min(min_entropy_incell)
		alpha = 0.0

		# Density and pressure from averaged state
		rho_bar = physics.compute_variable(self.var_name1, U_bar)
		p_bar = physics.compute_variable(self.var_name2, U_bar)
		s_bar = np.log(p_bar/np.power(rho_bar,1.4))
		rhos_bar = rho_bar*(s_bar-min_entropy) # cell average whose size is (200,1)

		if np.any(rho_bar < 0.) or np.any(p_bar < 0.):
			raise errors.NotPhysicalError

		# Ignore divide-by-zero
		np.seterr(divide='ignore')

		''' Limit density '''
		# Compute density at quadrature points
		rho_elem_faces = physics.compute_variable(self.var_name1,
				U_elem_faces)
		p_elem_faces = physics.compute_variable(self.var_name2, U_elem_faces)
		s_elem_faces = physics.compute_variable(self.var_name3, U_elem_faces)
		rhos_elem_faces = rho_elem_faces*(s_elem_faces-min_entropy)
		# Check if limiting is needed
		# theta = np.abs((rho_bar - POS_TOL)/(rho_bar - rho_elem_faces))
		theta1 = np.abs((rho_bar - alpha*rho_bar)/(rho_bar - rho_elem_faces+POS_TOL))
		theta2 = np.abs((p_bar - alpha*p_bar)/(p_bar - p_elem_faces+POS_TOL))
		theta3 = np.abs((rhos_bar - alpha*rhos_bar)/(rhos_bar - rhos_elem_faces+POS_TOL))
		
		# Truncate theta1; otherwise, can get noticeably different
		# results across machines, possibly due to poor conditioning in its
		# calculation

		theta1 = trunc(np.minimum(1., np.min(theta1, axis=1)))
		theta2 = trunc(np.minimum(1., np.min(theta2, axis=1)))
		theta3 = trunc(np.minimum(1., np.min(theta3, axis=1)))
		
		theta = np.minimum(theta1,theta2,theta3)
		
		# irho = physics.get_state_index(self.var_name1)

		# Get IDs of elements that need limiting
		elem_IDs = np.where(theta < 1.)[0]
		# Modify density coefficients
		if basis.MODAL_OR_NODAL == general.ModalOrNodal.Nodal:
			Uc[elem_IDs,:,:] = np.einsum('im, ijk -> ijk', theta[elem_IDs], 
					Uc[elem_IDs,:,:]) + np.einsum('im, ijk -> ijk', 1 - theta[
					elem_IDs], U_bar[elem_IDs,:,:])
		else:
			raise NotImplementedError

		np.seterr(divide='warn')

		return Uc # [ne, nq, ns]
