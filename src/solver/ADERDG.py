import code
import numpy as np
from scipy.linalg import solve_sylvester

from data import GenericData
import errors

from general import ModalOrNodal, StepperType

import meshing.meshbase as mesh_defs
import meshing.tools as mesh_tools

import numerics.basis.tools as basis_tools
import numerics.basis.ader_tools as basis_st_tools

import numerics.helpers.helpers as helpers
import numerics.limiting.tools as limiter_tools

import numerics.timestepping.tools as stepper_tools
import numerics.timestepping.stepper as stepper_defs

global echeck
echeck = -1

import solver.ader_tools as solver_tools
import solver.tools as dg_tools
import solver.base as base
import solver.DG as DG


class ElemHelpersADER(DG.ElemHelpers):

	def get_basis_and_geom_data(self, mesh, basis, order):

		dim = mesh.dim 
		quad_pts = self.quad_pts 
		num_elems = mesh.num_elems 
		nq = quad_pts.shape[0]
		nb = basis.nb

		self.jac_elems = np.zeros([num_elems,nq,dim,dim])
		self.ijac_elems = np.zeros([num_elems,nq,dim,dim])
		self.djac_elems = np.zeros([num_elems,nq,1])
		self.x_elems = np.zeros([num_elems,nq,dim])
		self.basis_phys_grad_elems = np.zeros([num_elems,nq,nb,dim])

		basis.get_basis_val_grads(quad_pts, get_val=True, get_ref_grad=True)

		self.basis_val = basis.basis_val
		self.basis_ref_grad = basis.basis_ref_grad

		for elem in range(mesh.num_elems):
			# get jacobian data
			djac, jac, ijac = basis_tools.element_jacobian(mesh, elem,
					quad_pts, get_djac=True, get_jac=True, get_ijac=True)
			# store the jacobian data
			self.jac_elems[elem] = jac
			self.ijac_elems[elem] = ijac
			self.djac_elems[elem] = djac

			# Physical coordinates of quadrature points
			x = mesh_tools.ref_to_phys(mesh, elem, quad_pts)
			# Store
			self.x_elems[elem] = x
			# Physical gradient
			# basis.get_basis_val_grads(quad_pts, get_phys_grad=True, 
			#		ijac=ijac) # gPhi is [nq,nb,dim]
			# self.basis_phys_grad_elems[elem] = basis.basis_phys_grad

class InteriorFaceHelpersADER(DG.InteriorFaceHelpers):

	def get_gaussian_quadrature(self, mesh, physics, basis, order):
		
		gbasis = mesh.gbasis
		quad_order = gbasis.FACE_SHAPE.get_quadrature_order(mesh, 
				order, physics=physics)
		self.quad_pts, self.quad_wts = \
				basis.FACE_SHAPE.get_quadrature_data(quad_order)

	def get_basis_and_geom_data(self, mesh, basis, order):

		# unpack
		dim = mesh.dim
		quad_pts = self.quad_pts 
		nq = quad_pts.shape[0]
		nb = basis.get_num_basis_coeff(order)
		nfaces_per_elem = mesh.gbasis.NFACES + 2

		# allocate
		self.faces_to_basisL = np.zeros([nfaces_per_elem,nq,nb])
		self.faces_to_basisR = np.zeros([nfaces_per_elem,nq,nb])
		self.normals_ifaces = np.zeros([mesh.num_interior_faces,nq,dim])

		for f in range(nfaces_per_elem):
			# left
			basis.get_basis_face_val_grads(mesh, f, quad_pts, 
					basis, get_val=True)
			self.faces_to_basisL[f] = basis.basis_val
			# right
			basis.get_basis_face_val_grads(mesh, f, quad_pts[::-1], 
					basis, get_val=True)
			self.faces_to_basisR[f] = basis.basis_val

		i = 0
		for IFace in mesh.interior_faces:
			# Normals
			# normals = mesh_defs.iface_normal(mesh, IFace, quad_pts)
			normals = mesh.gbasis.calculate_normals(mesh, IFace.elemL_id, 
				IFace.faceL_id, quad_pts)
			self.normals_ifaces[i] = normals
			i += 1

class BoundaryFaceHelpersADER(InteriorFaceHelpersADER):

	def get_basis_and_geom_data(self, mesh, basis, order):
		# separate these later

		# Unpack
		dim = mesh.dim
		quad_pts = self.quad_pts 
		nq = quad_pts.shape[0]
		nb = basis.get_num_basis_coeff(order)
		# nFacePerElem = mesh.nFacePerElem + 2
		nfaces_per_elem = mesh.gbasis.NFACES + 2

		# Allocate
		self.faces_to_basis = np.zeros([nfaces_per_elem, nq, nb])
		self.faces_to_xref = np.zeros([nfaces_per_elem, nq, dim+1])
		self.normals_bfgroups = []
		self.x_bfgroups = []


		for f in range(nfaces_per_elem):
			# Left
			self.faces_to_xref[f] = basis.get_elem_ref_from_face_ref(f, 
					quad_pts)
			basis.get_basis_face_val_grads(mesh, f, quad_pts, basis, 
					get_val=True)
			self.faces_to_basis[f] = basis.basis_val

		i = 0
		for BFG in mesh.boundary_groups.values():
			self.normals_bfgroups.append(np.zeros([BFG.num_boundary_faces,nq,
					dim]))
			self.x_bfgroups.append(np.zeros([BFG.num_boundary_faces,nq,dim]))
			normal_bfgroup = self.normals_bfgroups[i]
			x_bfgroup = self.x_bfgroups[i]
			j = 0
			for boundary_face in BFG.boundary_faces:
				# normals
				nvec = mesh.gbasis.calculate_normals(mesh, 
						boundary_face.elem_id, boundary_face.face_id, 
						quad_pts)
				normal_bfgroup[j] = nvec

				# Physical coordinates of quadrature points
				x = mesh_tools.ref_to_phys(mesh, boundary_face.elem_id, 
						self.faces_to_xref[boundary_face.face_id])
				# Store
				x_bfgroup[j] = x

				# Increment
				j += 1
			i += 1

	def alloc_other_arrays(self, physics, basis, order):
		quad_pts = self.quad_pts 
		nq = quad_pts.shape[0]
		ns = physics.NUM_STATE_VARS

		self.UqI = np.zeros([nq, ns])
		self.UqB = np.zeros([nq, ns])
		self.Fq = np.zeros([nq, ns])


class ADERHelpers(object):
	def __init__(self):
		self.MM = None
		self.iMM = None
		self.iMM_elems = None
		self.K = None
		self.iK = None 
		self.FTL = None 
		self.FTR = None
		self.SMT = None
		self.SMS_elems = None	
		self.jac_elems = None 
		self.ijac_elems = None 
		self.djac_elems = None 
		self.x_elems = None

	def calc_ader_matrices(self, mesh, basis, basis_st, dt, order):

		dim = mesh.dim
		nb = basis_st.nb
		SMS_elems = np.zeros([mesh.num_elems,nb,nb,dim])
		iMM_elems = np.zeros([mesh.num_elems,nb,nb])
		# Get flux matrices in time
		FTL = basis_st_tools.get_temporal_flux_ader(mesh, basis_st, basis_st,
				order, elem=0, PhysicalSpace=False)
		FTR = basis_st_tools.get_temporal_flux_ader(mesh, basis_st, basis,
				order, elem=0, PhysicalSpace=False)

		# Get stiffness matrix in time
		SMT = basis_st_tools.get_stiffness_matrix_ader(mesh, basis, basis_st,
				order, dt, elem=0, gradDir=1, PhysicalSpace=False)

		# Get stiffness matrix in space
		for elem in range(mesh.num_elems):
			SMS = basis_st_tools.get_stiffness_matrix_ader(mesh, basis,
					basis_st, order, dt, elem, gradDir=0, PhysicalSpace=True)
			SMS_elems[elem,:,:,0] = SMS.transpose()

			iMM = basis_st_tools.get_elem_inv_mass_matrix_ader(mesh,
					basis_st, order, elem, PhysicalSpace=True)
			iMM_elems[elem] = iMM

		iMM = basis_st_tools.get_elem_inv_mass_matrix_ader(mesh, basis_st,
				order, elem=-1, PhysicalSpace=False)
		MM = basis_st_tools.get_elem_mass_matrix_ader(mesh, basis_st, order,
				elem=-1, PhysicalSpace=False)

		self.FTL = FTL
		self.FTR = FTR
		self.SMT = SMT
		self.SMS_elems = SMS_elems
		self.MM = MM
		self.iMM = iMM
		self.iMM_elems = iMM_elems
		self.K = FTL - SMT
		self.iK = np.linalg.inv(self.K) 


	def get_geom_data(self, mesh, basis, order):

		# Unpack
		shape_name = basis.__class__.__bases__[1].__name__

		dim = mesh.dim 
		num_elems = mesh.num_elems 
		nb = basis.nb
		gbasis = mesh.gbasis
		xnode = gbasis.get_nodes(order)
		nnode = xnode.shape[0]

		# Allocate
		self.jac_elems = np.zeros([num_elems,nb,dim,dim])
		self.ijac_elems = np.zeros([num_elems,nb,dim,dim])
		self.djac_elems = np.zeros([num_elems,nb,1])
		self.x_elems = np.zeros([num_elems,nb,dim])


		for elem in range(mesh.num_elems):
			# Jacobian
			djac, jac, ijac = basis_tools.element_jacobian(mesh, elem, xnode,
				 	get_djac=True, get_jac=True, get_ijac=True)

			if shape_name is 'HexShape':
				self.jac_elems[elem] = np.tile(jac,(int(np.sqrt(nnode)),1,1))
				self.ijac_elems[elem] = np.tile(ijac,
						(int(np.sqrt(nnode)),1,1))
				self.djac_elems[elem] = np.tile(djac,(int(np.sqrt(nnode)),1))

				# Physical coordinates of nodal points
				x = mesh_tools.ref_to_phys(mesh, elem, xnode)
				# Store
				self.x_elems[elem] = np.tile(x,(int(np.sqrt(nnode)),1))

			elif shape_name is 'QuadShape':
				self.jac_elems[elem] = np.tile(jac,(nnode,1,1))
				self.ijac_elems[elem] = np.tile(ijac,(nnode,1,1))
				self.djac_elems[elem] = np.tile(djac,(nnode,1))

				# Physical coordinates of nodal points
				x = mesh_tools.ref_to_phys(mesh, elem, xnode)
				# Store
				self.x_elems[elem] = np.tile(x,(nnode,1))

	def compute_operators(self, mesh, physics, basis, basis_st, dt, order):
		self.calc_ader_matrices(mesh, basis, basis_st, dt, order)
		self.get_geom_data(mesh, basis_st, order)


class ADERDG(base.SolverBase):
    '''
    ADERDG inherits attributes and methods from the SolverBase class.
    See SolverBase for detailed comments of attributes and methods.

    Additional methods and attributes are commented below.
    '''
	def __init__(self,Params,physics,mesh):
		super().__init__(Params, physics, mesh)

		ns = physics.NUM_STATE_VARS

		TimeScheme = Params["TimeScheme"]
		if StepperType[TimeScheme] != StepperType.ADER:
			raise errors.IncompatibleError

		self.Stepper = stepper_defs.ADER(physics.U)
		stepper_tools.set_time_stepping_approach(self.Stepper, Params)

		# Set the space-time basis functions for the solver
		basis_name  = Params["InterpBasis"]
		self.basis_st = basis_st_tools.set_basis_spacetime(mesh, 
				physics.order, basis_name)

		# Set quadrature for space-time basis
		self.basis_st.set_elem_quadrature_type(Params["ElementQuadrature"])
		self.basis_st.set_face_quadrature_type(Params["FaceQuadrature"])

		self.basis.force_nodes_equal_quad_pts(Params["NodesEqualQuadpts"])
		self.basis_st.force_nodes_equal_quad_pts(Params["NodesEqualQuadpts"])

		# Allocate array for predictor step in ADER-Scheme
		physics.Up = np.zeros([self.mesh.num_elems, 
				self.basis_st.get_num_basis_coeff(physics.order), 
				physics.NUM_STATE_VARS])

		# Set predictor function
		SourceTreatment = Params["SourceTreatment"]
		self.calculate_predictor_elem = solver_tools.set_source_treatment(ns,
				SourceTreatment)

		# Check validity of parameters
		self.check_compatibility()

		# Precompute operators
		self.precompute_matrix_operators()

		if self.limiter is not None:
			self.limiter.precompute_operators(self)

		physics.ConvFluxFcn.alloc_helpers(np.zeros([self.iface_operators_st.
			quad_wts.shape[0], physics.NUM_STATE_VARS]))

		# Initialize state
		if Params["RestartFile"] is None:
			self.init_state_from_fcn()
	
	def __repr__(self):
		return '{self.__class__.__name__}(Physics: {self.physics},\n   \
				Basis: {self.basis}, \n   Basis_st: {self.basis_st},\n   \
				Stepper: {self.Stepper})'.format(self=self)
	
	def check_compatibility(self):
		super().check_compatibility()

		basis = self.basis
		Params = self.Params

		if Params["InterpolateFlux"] and 
				basis.MODAL_OR_NODAL != ModalOrNodal.Nodal:
			raise errors.IncompatibleError

		if Params["CFL"] != None:
			print('Error Message')
			print('-------------------------------------------------------')
			print('CFL-based time-stepping not supported in ADERDG')
			print('')
			raise errors.IncompatibleError

	def precompute_matrix_operators(self):
		mesh = self.mesh 
		physics = self.physics

		basis = self.basis
		basis_st = self.basis_st
		Stepper = self.Stepper

		self.elem_operators = DG.ElemHelpers()
		self.elem_operators.compute_operators(mesh, physics, basis, physics,
				order)
		self.iface_operators = DG.InteriorFaceHelpers()
		self.iface_operators.compute_operators(mesh, physics, basis, physics,
				order)
		self.bface_operators = DG.BoundaryFaceHelpers()
		self.bface_operators.compute_operators(mesh, physics, basis,
				physics.order)

		# Calculate ADER specific space-time operators
		self.elem_operators_st = ElemHelpersADER()
		self.elem_operators_st.compute_operators(mesh, physics, basis_st,
				physics.order)
		self.iface_operators_st = InteriorFaceHelpersADER()
		self.iface_operators_st.compute_operators(mesh, physics, basis_st,
				physics.order)
		self.bface_operators_st = BoundaryFaceHelpersADER()
		self.bface_operators_st.compute_operators(mesh, physics, basis_st,
				physics.order)

		Stepper.dt = Stepper.get_time_step(Stepper, self)
		dt = Stepper.dt
		self.ader_operators = ADERHelpers()
		self.ader_operators.compute_operators(mesh, physics, basis, 
				basis_st, dt, physics.order)

	def calculate_predictor_step(self, dt, W, Up):
		'''
		Calls the predictor step for each element

		Inputs:
		-------
			dt: time step 
			W: previous time step solution in space only

		Outputs:
		--------
			Up: predicted solution in space-time
		'''
		mesh = self.mesh
		physics = self.physics

		for elem in range(mesh.num_elems):
			Up[elem] = self.calculate_predictor_elem(self, elem, dt, W[elem],
					Up[elem])

		return Up

	def get_element_residual(self, elem, Up, ER):

		physics = self.physics
		mesh = self.mesh
		ns = physics.NUM_STATE_VARS
		dim = physics.dim

		elem_ops = self.elem_operators
		elem_ops_st = self.elem_operators_st

		quad_wts = elem_ops.quad_wts
		quad_wts_st = elem_ops_st.quad_wts
		quad_pts_st = elem_ops_st.quad_pts

		basis_val_st = elem_ops_st.basis_val
		x_elems = elem_ops.x_elems
		x = x_elems[elem]

		nq = quad_wts.shape[0]
		nq_st = quad_wts_st.shape[0]

		# interpolate state and gradient at quad points
		Uq = helpers.evaluate_state(Up, basis_val_st)

		if self.Params["ConvFluxSwitch"] == True:
			# evaluate the inviscid flux integral.
			Fq = physics.ConvFluxInterior(Uq) # [nq,sr,dim]
			ER += solver_tools.calculate_inviscid_flux_volume_integral(self, 
					elem_ops, elem_ops_st, elem, Fq)
		
		if self.Params["SourceSwitch"] == True:
			# evaluate the source term integral
			t = np.zeros([nq_st,dim])
			TimePhiData = None
			t, TimePhiData = solver_tools.ref_to_phys_time(mesh, elem, 
					self.time, self.Stepper.dt, TimePhiData, quad_pts_st, 
					t, None)
			
			Sq = elem_ops_st.Sq
			Sq[:] = 0.
			Sq = physics.SourceState(nq_st, x, t, Uq, Sq) # [nq,sr,dim]

			ER += solver_tools.calculate_source_term_integral(elem_ops, 
					elem_ops_st, elem, Sq)

		if elem == echeck:
			code.interact(local=locals())

		return ER

	def get_interior_face_residual(self, iiface, UpL, UpR, RL, RR):

		mesh = self.mesh
		physics = self.physics
		IFace = mesh.interior_faces[iiface]
		elemL = IFace.elemL_id
		elemR = IFace.elemR_id
		faceL_id = IFace.faceL_id
		faceR_id = IFace.faceR_id

		if faceL_id == 0:
			faceL_id_st = 3
		elif faceL_id == 1:
			faceL_id_st = 1
		else:
			return IncompatibleError
		if faceR_id == 0:
			faceR_id_st = 3
		elif faceR_id == 1:
			faceR_id_st = 1
		else:
			return IncompatibleError

		iface_ops = self.iface_operators
		iface_ops_st = self.iface_operators_st
		quad_wts_st = iface_ops_st.quad_wts

		faces_to_basisL = iface_ops.faces_to_basisL
		faces_to_basisR = iface_ops.faces_to_basisR

		faces_to_basisL_st = iface_ops_st.faces_to_basisL
		faces_to_basisR_st = iface_ops_st.faces_to_basisR

		UqL = iface_ops.UqL
		UqR = iface_ops.UqR
		Fq = iface_ops.Fq

		basis_valL = faces_to_basisL[faceL_id]
		basis_valR = faces_to_basisR[faceR_id]

		basis_valL_st = faces_to_basisL_st[faceL_id_st]
		basis_valR_st = faces_to_basisR_st[faceR_id_st]

		UqL = helpers.evaluate_state(UpL, basis_valL_st)
		UqR = helpers.evaluate_state(UpR, basis_valR_st)

		normals_ifaces = iface_ops.normals_ifaces
		normals = normals_ifaces[iiface]

		if self.Params["ConvFluxSwitch"] == True:

			Fq = physics.ConvFluxNumerical(UqL, UqR, normals) # [nq_st,ns]
			RL -= solver_tools.calculate_inviscid_flux_boundary_integral(
					basis_valL, quad_wts_st, Fq)
			RR += solver_tools.calculate_inviscid_flux_boundary_integral(
					basis_valR, quad_wts_st, Fq)

		if elemL == echeck or elemR == echeck:
			if elemL == echeck: print("Left!")
			else: print("Right!")
			code.interact(local=locals())

		return RL, RR

	def get_boundary_face_residual(self, BFG, ibface, U, R):

		mesh = self.mesh
		dim = mesh.dim
		physics = self.physics
		ns = physics.NUM_STATE_VARS
		ibfgrp = BFG.number
		boundary_face = BFG.boundary_faces[ibface]
		elem = boundary_face.elem_id
		face = boundary_face.face_id

		bface_ops = self.bface_operators
		bface_ops_st = self.bface_operators_st
		quad_wts_st = bface_ops_st.quad_wts
		faces_to_xref_st = bface_ops_st.faces_to_xref

		faces_to_basis = bface_ops.faces_to_basis
		faces_to_basis_st = bface_ops_st.faces_to_basis

		normals_bfgroups = bface_ops.normals_bfgroups
		x_bfgroups = bface_ops.x_bfgroups

		UqI = bface_ops_st.UqI
		UqB = bface_ops_st.UqB
		Fq = bface_ops_st.Fq

		if face == 0:
			face_st = 3
		elif face == 1:
			face_st = 1
		else:
			return IncompatibleError
	
		basis_val = faces_to_basis[face]
		basis_val_st = faces_to_basis_st[face_st]
		xref_st = faces_to_xref_st[face_st]

		nq_st = quad_wts_st.shape[0]

		TimePhiData = None

		t = np.zeros([nq_st,dim])
		t, TimePhiData = solver_tools.ref_to_phys_time(mesh, elem, self.time,
				self.Stepper.dt, TimePhiData, xref_st, t, None)

		# interpolate state and gradient at quad points
		UqI = helpers.evaluate_state(U, basis_val_st)


		normals = normals_bfgroups[ibfgrp][ibface]
		x = x_bfgroups[ibfgrp][ibface]

		# Get boundary state
		BC = physics.BCs[BFG.name]

		if self.Params["ConvFluxSwitch"] == True:
			# loop over time to apply BC at each temporal quadrature point
			for i in range(len(t)):	
				Fq[i,:] = BC.get_boundary_flux(physics, x, t[i], normals, 
						UqI[i,:].reshape([1,ns]))

			R -= solver_tools.calculate_inviscid_flux_boundary_integral(
					basis_val, quad_wts_st, Fq)

		if elem == echeck:
			code.interact(local=locals())

		return R

	def flux_coefficients(self, elem, dt, order, basis, Up):
		'''
		Calculates the polynomial coefficients for the flux functions in 
		ADER-DG

		Inputs:
		-------
			elem: element index
			dt: time step size
			order: solution order
			basis: basis object
			Up: solution array
			
		Outputs:
		--------
			F: polynomical coefficients of the flux function
		'''
		physics = self.physics
		mesh = self.mesh
		ns = physics.NUM_STATE_VARS
		dim = physics.dim
		Params = self.Params

		InterpolateFlux = Params["InterpolateFlux"]

		elem_ops = self.elem_operators
		elem_ops_st = self.elem_operators_st
		djac_elems = elem_ops.djac_elems 
		djac = djac_elems[elem]

		rhs = np.zeros([basis.get_num_basis_coeff(order),ns,dim],
				dtype=Up.dtype)
		F = np.zeros_like(rhs)

		if Params["InterpolateFlux"]:

			Fq = physics.ConvFluxInterior(Up)
			dg_tools.interpolate_to_nodes(Fq, F)
		else:
			ader_ops = self.ader_operators
			basis_val_st = elem_ops_st.basis_val
			quad_wts_st = elem_ops_st.quad_wts
			quad_wts = elem_ops.quad_wts
			quad_pts_st = elem_ops_st.quad_pts
			quad_pts = elem_ops.quad_pts
			nq_st = quad_wts_st.shape[0]
			nq = quad_wts.shape[0]
			iMM = ader_ops.iMM_elems[elem]

			Uq = helpers.evaluate_state(Up, basis_val_st)

			Fq = physics.ConvFluxInterior(Uq)
			solver_tools.L2_projection(mesh, iMM, basis, quad_pts_st, 
					quad_wts_st, np.tile(djac,(nq,1)), Fq[:,:,0], F[:,:,0])

		return F*dt/2.0

	def source_coefficients(self, elem, dt, order, basis, Up):
		'''
		Calculates the polynomial coefficients for the source functions in 
		ADER-DG

		Inputs:
		-------
			elem: element index
			dt: time step size
			order: solution order
			basis: basis object
			Up: solution array
			
		Outputs:
		--------
			S: polynomical coefficients of the flux function
		'''
		mesh = self.mesh
		dim = mesh.dim
		physics = self.physics
		ns = physics.NUM_STATE_VARS
		Params = self.Params

		InterpolateFlux = Params["InterpolateFlux"]

		elem_ops = self.elem_operators
		elem_ops_st = self.elem_operators_st
		djac_elems = elem_ops.djac_elems 
		djac = djac_elems[elem]

		x_elems = elem_ops.x_elems
		x = x_elems[elem]

		ader_ops = self.ader_operators
		x_elems_ader = ader_ops.x_elems
		x_ader = x_elems_ader[elem]

		TimePhiData = None

		if Params["InterpolateFlux"]:
			xnode = basis.get_nodes(order)
			nb = xnode.shape[0]
			t = np.zeros([nb,dim])
			t, TimePhiData = solver_tools.ref_to_phys_time(mesh, elem, 
					self.time, self.Stepper.dt, TimePhiData, xnode, t, None)
			Sq = np.zeros([t.shape[0],ns])
			S = np.zeros_like(Sq)
			Sq = physics.SourceState(nb, x_ader, t, Up, Sq)
			dg_tools.interpolate_to_nodes(Sq, S)
		else:

			ader_ops = self.ader_operators
			basis_val_st = elem_ops_st.basis_val
			nb_st = basis_val_st.shape[1]
			quad_wts_st = elem_ops_st.quad_wts
			quad_wts = elem_ops.quad_wts
			quad_pts_st = elem_ops_st.quad_pts
			nq_st = quad_wts_st.shape[0]
			nq = quad_wts.shape[0]
			iMM = ader_ops.iMM_elems[elem]

			Uq = helpers.evaluate_state(Up, basis_val_st)

			t = np.zeros([nq_st,dim])
			t, TimePhiData = solver_tools.ref_to_phys_time(mesh, elem, 
					self.time, self.Stepper.dt, TimePhiData, 
					quad_pts_st, t, None)
		
			Sq = np.zeros([t.shape[0],ns])
			S = np.zeros([nb_st, ns])
			Sq = physics.SourceState(nq_st, x, t, Uq, Sq) # [nq,sr,dim]		
			solver_tools.L2_projection(mesh, iMM, basis, quad_pts_st, 
					quad_wts_st, np.tile(djac,(nq,1)), Sq, S)

		return S*dt/2.0