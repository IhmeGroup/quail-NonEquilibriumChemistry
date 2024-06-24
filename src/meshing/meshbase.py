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
#       File : src/meshing/meshbase.py
#
#       Contains class definitions for mesh structures.
#
# ------------------------------------------------------------------------ #
from enum import Enum, auto
import numpy as np

import scipy
from scipy import linalg
import pickle

from general import ShapeType
import numerics.basis.basis as basis_defs


class InteriorFace():
	'''
	This class provides information about a given interior face.

	Attributes:
	-----------
	elemL_ID : int
		ID of "left" element
	faceL_ID : int
		local ID of face from perspective of left element
	elemR_ID : int
		ID of "right" element
	faceR_ID : int
		local ID of face from perspective of right element
	'''
	def __init__(self):
		self.elemL_ID = 0
		self.faceL_ID = 0
		self.elemR_ID = 0
		self.faceR_ID = 0


class BoundaryFace():
	'''
	This class provides information about a given boundary face.

	Attributes:
	-----------
	elem_ID : int
		ID of adjacent element
	face_ID : int
		local ID of face from perspective of adjacent element
	'''
	def __init__(self):
		self.elem_ID = 0
		self.face_ID = 0


class BoundaryGroup():
	'''
	This class stores boundary face objects for a given boundary group.

	Attributes:
	-----------
	name : str
		boundary name
	number : int
		boundary number
	num_boundary_faces : int
		number of faces in boundary group
	boundary_faces : list
		list of BoundaryFace objects

	Methods:
	---------
	allocate_boundary_faces
		allocates list of BoundaryFace objects
	'''
	def __init__(self):
		self.name = ""
		self.number = -1
		self.num_boundary_faces = 0
		self.boundary_faces = []

	def allocate_boundary_faces(self):
		'''
		This method allocates the list of boundary_face objects

		Outputs:
		--------
			self.boundary_faces
		'''
		self.boundary_faces = [BoundaryFace() for i in \
				range(self.num_boundary_faces)]


class Element():
	'''
	This class provides information about a given element.

	Attributes:
	-----------
	id: int
		element ID
	node_IDs: numpy array
		global IDs of the element nodes
	node_coords: numpy array
		coordinates of the element nodes [num_nodes, ndims]
	face_to_neighbors: numpy array
		maps local face ID to element ID of
		neighbor across said face [num_faces]
	'''
	def __init__(self, elem_ID=-1):
		self.ID = elem_ID
		self.node_IDs = np.zeros(0, dtype=int)
		self.node_coords = np.zeros(0)
		self.face_to_neighbors = np.zeros(0, dtype=int)


class Mesh():
	'''
	This class stores information about the mesh.

	Attributes:
	-----------
	ndims : int
		number of spatial dimensions
	num_nodes : int
		total number of nodes
	node_coords : numpy array
		coordinates of nodes [num_nodes, ndims]
	num_interior_faces : int
		number of interior faces
	interior_faces : list
		list of interior face objects
	num_boundary_groups : int
		number of boundary face groups
	boundary_groups : dict
		dict whose keys are boundary names and values are BoundaryGroup
		objects
	gbasis : Basis class
		object for geometry basis
	gorder : int
		order of geometry interpolation
	num_elems : int
		total number of elements in mesh
	num_nodes_per_elem : int
		number of nodes per element
	elem_to_node_IDs : numpy array
		maps element ID to global node IDs
		[num_elems, num_nodes_per_elem]
	elements : list
		list of Element objects

	Methods:
	---------
	set_params
		sets certain mesh parameters
	allocate_elem_to_node_IDs_map
		allocates self.elem_to_node_IDs
	allocate_interior_faces
		allocates self.interior_faces
	add_boundary_group
		appends new boundary group to self.boundary_groups
	create_elements
		creates self.elements
	'''
	def __init__(self, ndims=1, num_nodes=1, num_elems=1, gbasis=None,
			gorder=1):
		if gbasis is None:
			gbasis = basis_defs.LagrangeSeg(1)

		self.ndims = ndims
		self.num_nodes = num_nodes
		self.node_coords = None
		self.num_interior_faces = 0
		self.interior_faces = []
		self.num_boundary_groups = 0
		self.boundary_groups = {}
		self.gbasis = gbasis
		self.gorder = gorder
		self.num_elems = num_elems
		self.num_nodes_per_elem = gbasis.get_num_basis_coeff(gorder)
		self.elem_to_node_IDs = np.zeros(0, dtype=int)
		self.elements = []

	def set_params(self, gbasis, gorder=1, num_elems=1):
		'''
		This method sets certain mesh parameters

		Inputs:
		-------
			gbasis: geometry basis object
			gorder: [OPTIONAL] order of geometry interpolation
			num_elems: [OPTIONAL] total number of elements in mesh

		Outputs:
		--------
			self.gbasis: geometry basis object
			self.gorder: order of geometry interpolation
			self.num_elems: total number of elements in mesh
			self.num_nodes_per_elem: number of nodes per element
		'''
		self.gbasis = gbasis
		self.gorder = gorder
		self.num_elems = num_elems
		self.num_nodes_per_elem = gbasis.get_num_basis_coeff(gorder)

	def allocate_elem_to_node_IDs_map(self):
		'''
		This method allocates self.elem_to_node_IDs

		Outputs:
		--------
			self.elem_to_node_IDs: maps element ID to global node IDs
				[num_elems, num_nodes_per_elem]

		Notes:
		------
			elem_to_node_IDs[elem_ID][i] = ith node of elem_ID,
				where i = 1, 2, ..., num_nodes_per_elem
		'''
		self.elem_to_node_IDs = np.zeros([self.num_elems,
				self.num_nodes_per_elem], dtype=int)

	def allocate_interior_faces(self):
		'''
		This method allocates self.interior_faces

		Outputs:
		--------
			self.interior_faces: list of InteriorFace objects
		'''
		self.interior_faces = [InteriorFace() for i in range(
				self.num_interior_faces)]

	def add_boundary_group(self, bname):
		'''
		This method appends a new boundary group to self.boundary_groups

		Inputs:
		-------
			bname: name of boundary

		Outputs:
		--------
			bgroup: new boundary group
			self.boundary groups: updated to contain bgroup
		'''
		if bname in self.boundary_groups:
			raise ValueError("Repeated boundary names")
		bgroup = BoundaryGroup()
		self.boundary_groups[bname] = bgroup
		bgroup.name = bname
		self.num_boundary_groups = len(self.boundary_groups)
		bgroup.number = self.num_boundary_groups - 1

		return bgroup

	def create_elements(self):
		'''
		This method creates self.elements

		Outputs:
		--------
			self.elements: list of Element objects
		'''
		# Allocate
		self.elements = [Element() for i in range(self.num_elems)]

		# Fill in information for each element
		for elem_ID in range(self.num_elems):
			elem = self.elements[elem_ID]

			elem.ID = elem_ID
			elem.node_IDs = self.elem_to_node_IDs[elem_ID]
			elem.node_coords = self.node_coords[elem.node_IDs]
			elem.face_to_neighbors = np.full(self.gbasis.NFACES, -1)

		# Fill in information about neighbors
		for int_face in self.interior_faces:
			elemL_ID = int_face.elemL_ID
			elemR_ID = int_face.elemR_ID
			faceL_ID = int_face.faceL_ID
			faceR_ID = int_face.faceR_ID

			elemL = self.elements[elemL_ID]
			elemR = self.elements[elemR_ID]

			elemL.face_to_neighbors[faceL_ID] = elemR_ID
			elemR.face_to_neighbors[faceR_ID] = elemL_ID

	def create_metric_tensor(self):
		'''
		This method creates a smooth Riemannian metric tensor from mesh elements,
		used for AV computation. Uses the procedure oultined by Pennec, Yano:
			Pennec, X., Fillard, P., and Ayache, N. "A Riemannian framework for 
			tensor computing." Int. J. Comput. Vision, 66(1):41-66, 2006. 

			M. Yano. An optimization framework for adaptive higher-order 
			discretizations of partial differential equations on anisotropic 
			simplex meshes. Ph.D. Thesis, Department of Aeronautics and 
			Astronautics, Massachusetts Institute of Technology, 2012. 

		Outputs:
		--------
			self.metric_tensor: smooth Riemannian metric tensor for the mesh
			self.inv_metric_tensor: the inverse metric tensor
			self.length_scale: element length scales computed from metric tensor
		'''
		# Unpack
		ndims = self.ndims
		num_nodes = self.num_nodes
		num_elems = self.num_elems
		min_err = 1e-3

		print("Creating Metric Tensor...")

		# Check dimensions (only defined for 2D mesh)
		if ndims != 2:
			raise NotImplementedError

		# Load vertex-based metric for the mesh (if available)
		try:
			with open('metric.pkl', 'rb') as f:
				vert_metric_tensor = pickle.load(f)
		# If unavailable, generate from element-implied metric tensor
		except:
			# Only defined for triangular (simplex) mesh
			if self.gbasis.SHAPE_TYPE != basis_defs.TriShape.SHAPE_TYPE:
				raise NotImplementedError

			# Compute element-implied metric tensor
			elem_metric_tensor = np.zeros([num_elems, ndims, ndims])
			for elem_ID in range(num_elems):
				elem = self.elements[elem_ID]
				edges = elem.node_coords - np.roll(elem.node_coords, 1, 0)

				# Generate edge-length metric
				sz = int( ndims*(ndims+1)/2 )
				A = np.zeros([sz, sz])
				for edge_ind in range(edges.shape[0]):
					ex, ey = edges[edge_ind]
					A[edge_ind, :] = [ex**2, ex*ey, ey**2]

				# Solve for element-implied matrix coefficients (symmetric positive semidefinite)
				coeffs = np.linalg.solve(A, np.ones(sz))
				M = [[coeffs[0], coeffs[1]/2], [coeffs[1]/2, coeffs[2]]]
				elem_metric_tensor[elem_ID, :, :] = [[coeffs[0], coeffs[1]/2], [coeffs[1]/2, coeffs[2]]]

			# Instantiate vertex-based metric tensor
			vert_metric_tensor = np.zeros([num_nodes, ndims, ndims])
			nodes = np.arange(num_nodes)
			node_to_elem_IDs = [[] for i in range(num_nodes)]

			# Store adjacent elements for each vertex
			for elem_ID in range(num_elems):
				for vert_ID in self.elem_to_node_IDs[elem_ID]:
					node_to_elem_IDs[vert_ID].append(elem_ID)
					# Initial guess for vertex-based metric is superposition of adjacent element metrics
					vert_metric_tensor[vert_ID] += elem_metric_tensor[elem_ID, :, :]

			# Iteratively step thru element metric-averaging gradient descent
			err = np.ones(num_nodes) * np.inf
			while nodes[err>min_err].size > 0:

				# Iterate over set of unconverged vertices
				for vert_ID in nodes[err>min_err]:
					Mv = vert_metric_tensor[vert_ID]
					Mv_sqrt = scipy.linalg.fractional_matrix_power(Mv, 0.5)
					Mv_invsqrt = scipy.linalg.fractional_matrix_power(Mv, -0.5)
					metric_logsum = np.zeros([ndims, ndims])

					# Add element metric log-distance contribution for each adjacent element
					for elem_ID in node_to_elem_IDs[vert_ID]:
						Mk = elem_metric_tensor[elem_ID]
						metric_logsum += scipy.linalg.logm(Mv_invsqrt@Mk@Mv_invsqrt)

					# Update vertex-based metric
					nv = len(node_to_elem_IDs[vert_ID])
					logsum_exp = scipy.linalg.expm(1/nv * metric_logsum)
					Mv_next = Mv_sqrt@logsum_exp@Mv_sqrt

					# Compute step norm to evaluate convergence
					err[vert_ID] = np.linalg.norm(Mv_next - Mv)
					vert_metric_tensor[vert_ID] = Mv_next
				
				# Write vertex-based metric tensor to file
				with open('metric.pkl', 'wb') as f:
					pickle.dump(vert_metric_tensor, f)

		# Instantiate smooth Riemannian metric tensor for arbitrary element shapes
		smooth_metric_tensor = np.zeros([num_elems, ndims, ndims])
		inv_metric_tensor = np.zeros([num_elems, ndims, ndims])
		length_scale = np.zeros([num_elems])
		elems = np.arange(num_elems)

		# Store vertex IDs composing each element
		for elem_ID in elems:
			for vert_ID in self.elem_to_node_IDs[elem_ID]:
				# Initial guess for smooth metric is superposition of vertex metrics
				smooth_metric_tensor[elem_ID] += vert_metric_tensor[vert_ID, :, :]

		# Iteratively step thru vertex metric-averaging gradient descent
		err = np.ones(num_elems) * np.inf
		while elems[err>min_err].size > 0:

			# Iterate over set of unconverged elements
			for elem_ID in elems[err>min_err]:
				Mk = smooth_metric_tensor[elem_ID]
				Mk_sqrt = scipy.linalg.fractional_matrix_power(Mk, 0.5)
				Mk_invsqrt = scipy.linalg.fractional_matrix_power(Mk, -0.5)
				metric_logsum = np.zeros([ndims, ndims])

				# Add vertex metric log-distance contribution for each element vertex
				for vert_ID in self.elem_to_node_IDs[elem_ID]:
					Mv = vert_metric_tensor[vert_ID]
					metric_logsum += scipy.linalg.logm(Mk_invsqrt@Mv@Mk_invsqrt)

				# Update smooth metric
				nk = len(self.elem_to_node_IDs[elem_ID])
				logsum_exp = scipy.linalg.expm(1/nk * metric_logsum)
				Mk_next = Mk_sqrt@logsum_exp@Mk_sqrt

				# Compute step norm to evaluate convergence
				err[elem_ID] = np.linalg.norm(Mk_next - Mk)
				smooth_metric_tensor[elem_ID] = Mk_next
				h = scipy.linalg.fractional_matrix_power(Mk_next, -0.5)
				inv_metric_tensor[elem_ID] = h
				length_scale[elem_ID] = scipy.linalg.det(h)**(1/ndims)

		print("Metric Tensor Complete")

		self.metric_tensor = smooth_metric_tensor
		self.inv_metric_tensor = inv_metric_tensor
		self.length_scale = length_scale

