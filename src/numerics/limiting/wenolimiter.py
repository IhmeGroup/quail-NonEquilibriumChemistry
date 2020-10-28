# ------------------------------------------------------------------------ #
#
#       File : src/numerics/limiting/wenolimiter.py
#
#       Contains class definitions for weno limiters.
#      
# ------------------------------------------------------------------------ #
from abc import ABC, abstractmethod
import numpy as np 

import errors
import general

import meshing.tools as mesh_tools

import numerics.helpers.helpers as helpers
import numerics.limiting.base as base

import numerics.limiting.tools as limiter_tools


class ScalarWENO(base.LimiterBase):
	'''
	This class corresponds to the scalar WENO limiter. It inherits from 
	the LimiterBase class. See LimiterBase for detailed comments of 
	attributes and methods. See the following references:
		[1] 

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
	COMPATIBLE_PHYSICS_TYPES = general.PhysicsType.Burgers

	def __init__(self, physics_type):
		super().__init__(physics_type)
		self.var_name1 = "Scalar"
		self.elem_vols = np.zeros(0)
		self.basis_val_elem_faces = np.zeros(0)
		self.quad_wts_elem = np.zeros(0)
		self.djac_elems = np.zeros(0)

	def precompute_helpers(self, solver):
		# Unpack
		mesh = solver.mesh
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

		# identify neighboring elements
		elemP_IDs = int_face_helpers.elemL_ID.copy()
		elemM_IDs = int_face_helpers.elemR_ID.copy()

		for elem_ID in range(self.elem_vols.shape[0]):
			elemP_IDs[elem_ID] = mesh.elements[elem_ID].face_to_neighbors[1]
			elemM_IDs[elem_ID] = mesh.elements[elem_ID].face_to_neighbors[0]
	
		self.elemP_IDs = elemP_IDs
		self.elemM_IDs = elemM_IDs

	def get_nonlinearwts(self, p, gamma, basis_phys_grad, quad_wts, vol):

		# calculate the smoothness indicator 
		s = 1

		basis_p = (np.matmul(basis_phys_grad[:, :,:,0],p)**2).transpose(0,2,1)
		basis_p_qwts = np.einsum('ikj,jl -> ik',basis_p, quad_wts)
		beta = vol**(2*s-1) * basis_p_qwts.reshape(p.shape[0])

		eps = 1.0e-6

		return gamma / (eps + beta)**2

	def limit_element(self, solver, elem_ID, Uc):
		
		elem_helpers = solver.elem_helpers
		vol = self.elem_vols[elem_ID]
		basis_phys_grad = elem_helpers.basis_phys_grad_elems[elem_ID]
		quad_wts = elem_helpers.quad_wts
		
		weno_wts = np.zeros([3])
		# determine if the element requires limiting
		shock_indicated = self.shock_indicator(self, solver, elem_ID, Uc)

		if not shock_indicated:
			return Uc
		# unpack limiter info
		if self.u_elem is not None:
			p0 = self.um_elem
			p1 = self.u_elem
			p2 = self.up_elem

			p0_bar = self.um_bar
			p1_bar = self.u_bar
			p2_bar = self.up_bar

	def limit_element(self, solver, Uc):
		# Unpack
		elem_helpers = solver.elem_helpers
		vols = self.elem_vols
		basis_phys_grads = elem_helpers.basis_phys_grad_elems
		quad_wts = elem_helpers.quad_wts
		
		weno_wts = np.zeros([Uc.shape[0], 3])

		# Determine if the elements requires limiting
		shock_indicated = self.shock_indicator(self, solver, Uc)
	
		# unpack limiter info
		if self.U_elem is not None:
			p0 = self.Um_elem
			p1 = self.U_elem
			p2 = self.Up_elem

			p0_bar = self.Um_bar
			p1_bar = self.U_bar
			p2_bar = self.Up_bar

		else:
			raise NotImplementedError

		p0_tilde = p0 - p0_bar + p1_bar
		p2_tilde = p2 - p2_bar + p1_bar
		
		# Currently only implemented for P1 (Need to add Hessian for P2)
		if solver.order > 1:
			raise NotImplementedError

		# Calculate non-linear weights
		weno_wts[:, 0] = self.get_nonlinearwts(p0, 0.001, basis_phys_grads, quad_wts, vols)
		weno_wts[:, 1] = self.get_nonlinearwts(p1, 0.998, basis_phys_grads, quad_wts, vols)
		weno_wts[:, 2] = self.get_nonlinearwts(p2, 0.001, basis_phys_grads, quad_wts, vols)

		normal_wts = weno_wts / np.sum(weno_wts, axis=1).reshape(weno_wts.shape[0], 1)

		# Update state_coeffs for indicated elements
		Uc[shock_indicated] = np.einsum('i, ijk -> ijk', normal_wts[shock_indicated, 0], 
				p0_tilde[shock_indicated]) + \
							  np.einsum('i, ijk -> ijk', normal_wts[shock_indicated, 1],
				p1[shock_indicated]) + \
							  np.einsum('i, ijk -> ijk', normal_wts[shock_indicated, 2],
				p2_tilde[shock_indicated])
		
		return Uc