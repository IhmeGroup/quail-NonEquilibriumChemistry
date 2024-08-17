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
from os.path import isfile

import scipy
import pickle

from quail.general import ShapeType
import quail.numerics.basis.basis as basis_defs

# Vectorize some scipy.linalg functions
sqrtm = np.vectorize(scipy.linalg.sqrtm, signature='(n,m)->(n,m)')
logm = np.vectorize(scipy.linalg.logm, signature='(n,m)->(n,m)')
expm = np.vectorize(scipy.linalg.expm, signature='(n,m)->(n,m)')
fractional_matrix_power = np.vectorize(scipy.linalg.fractional_matrix_power,
                                       excluded=[1], signature='(n,m)->(n,m)')


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
    
    def get_simplex_implied_metrics(self):
        '''
        Computes the simplex-implied metrics and simplex-to-vertex mapping.
        
        Element-implied metrics are defined for simplex elements. If the mesh
        contains non-simplex elements, they will first be triangulated.

        Outputs:
        --------
        simplex_metric_tensor: ndarray
            Element-implied metrics for a set of simplex elements, with shape
            ``(num_elems*num_simplex_per_elem, ndims, ndims)``.
        node_to_simplex_IDs: list of ndarray
            Set of simplex indices neighboring each node.
        '''
        # Unpack
        ndims = self.ndims
        num_nodes = self.num_nodes
        num_elems = self.num_elems
        num_nodes_per_simplex = ndims+1
        num_edges_per_simplex = ndims*(ndims+1)//2

        # Define simplex elements
        if (self.gbasis.SHAPE_TYPE in [basis_defs.TriShape.SHAPE_TYPE] and
            self.gbasis.order == 1):
            # No triangulation needed if mesh is already a simplex
            num_nodes_per_elem = 3
            num_simplex_per_elem = 1
            simplices = np.array([(0, 1, 2)], dtype=int)
        else:
            # Get node coordinates in reference space
            node_ref = self.gbasis.get_nodes(p=self.gbasis.order)
            num_nodes_per_elem = node_ref.shape[0]

            # Create triangulation from reference element
            tri = scipy.spatial.Delaunay(node_ref)
            simplices = tri.simplices
            num_simplex_per_elem = tri.nsimplex
        
        # Create a lookup from each vertex to each simplex it borders
        vertex_to_simplex = [
            np.array([i for i, s in enumerate(simplices) if vertex in s])
            for vertex in range(num_nodes_per_elem)
        ]

        # Check if simplex metrics are already available
        if isfile('simplex_metric.pkl'):
            with open('simplex_metric.pkl', 'rb') as f:
                simplex_metric_tensor = pickle.load(f)
                node_to_simplex_IDs = pickle.load(f)
            
            # If it seems to match the current mesh, return it
            if (simplex_metric_tensor.shape == (num_elems*num_simplex_per_elem, ndims, ndims) and
                len(node_to_simplex_IDs) == num_nodes):
                print("Using simplex metrics from simplex_metric.pkl")
                return simplex_metric_tensor, node_to_simplex_IDs

        print("Computing simplex element-implied metrics...")

        # Set up mapping
        node_to_simplex_IDs = [np.empty(shape=(0,), dtype=int)
                               for i in range(num_nodes)]
        
        # Define all edges for each simplex
        idx = np.stack(np.triu_indices(num_nodes_per_simplex, k=1), axis=-1)
        edge_indices = simplices[:, idx].reshape((-1, 2))  # [nsimplex*nedges, 2]

        # Find unique edges, and return the inverse which maps edges to the simplices
        edge_indices, edges_to_simplex = np.unique(edge_indices, return_inverse=True, axis=0)

        # Preallocate metric tensor
        simplex_metric_tensor = np.zeros((num_elems, num_simplex_per_elem, ndims, ndims))

        # Preallocate arrays for the linear system of equations
        # A = np.zeros((num_edges_per_simplex, num_edges_per_simplex))
        b = np.ones((num_edges_per_simplex,))

        # Loop over each element
        for elem_ID in range(num_elems):
            # Compute all edge lengths
            elem = self.elements[elem_ID]
            nodes = elem.node_coords
            edges = np.diff(nodes[edge_indices, :], axis=1)

            # Map edges to simplices
            edges = edges[edges_to_simplex, :].reshape((num_simplex_per_elem, num_edges_per_simplex, ndims))

            # Construct and solve linear system for all simplices belonging to this element
            # TODO: Look for way to fully vectorize this
            triu_idx = np.triu_indices(ndims)
            A = edges[:, :, triu_idx[0]] * edges[:, :, triu_idx[1]]
            M_coeffs = np.linalg.solve(A, b)
            simplex_metric_tensor[elem_ID][:, triu_idx[0], triu_idx[1]] = M_coeffs
            simplex_metric_tensor[elem_ID][:, triu_idx[1], triu_idx[0]] += M_coeffs
            simplex_metric_tensor[elem_ID] *= 0.5

            # Map simplices to their corresponding nodes
            node_IDs = elem.node_IDs
            for local_node_ID, global_node_ID in enumerate(node_IDs):
                local_simplex_IDs = vertex_to_simplex[local_node_ID]
                global_simplex_IDs = elem_ID*num_simplex_per_elem + local_simplex_IDs
                node_to_simplex_IDs[global_node_ID] = np.append(
                    node_to_simplex_IDs[global_node_ID], global_simplex_IDs)

        # Flatten first two dimensions and save results
        simplex_metric_tensor = simplex_metric_tensor.reshape((num_elems*num_simplex_per_elem, ndims, ndims))
        with open('simplex_metric.pkl', 'wb') as f:
            pickle.dump(simplex_metric_tensor, f)
            pickle.dump(node_to_simplex_IDs, f)

        # Flatten first two dimensions		
        return simplex_metric_tensor, node_to_simplex_IDs
    
    def get_vertex_metrics(self):
        '''
        Computes the affine-invariant Riemannian metric tensors at each nodes
        in the mesh.
        '''
        # Unpack
        ndims = self.ndims
        num_nodes = self.num_nodes
        num_elems = self.num_elems
        min_err = 1e-3

        # Load vertex-based metric for the mesh (if available)
        if isfile('vertex_metric.pkl'):
            with open('vertex_metric.pkl', 'rb') as f:
                vertex_metric_tensor = pickle.load(f)
                print("Using vertex metrics from vertex_metric.pkl")
                return vertex_metric_tensor
        
        # If unavailable, generate from element-implied metric tensor
        simplex_metric_tensor, node_to_simplex_IDs = self.get_simplex_implied_metrics()

        print("Computing affine-invariant metric tensors at mesh nodes...")

        # Instantiate vertex-based metric tensor
        vertex_metric_tensor = np.zeros([num_nodes, ndims, ndims])
        nodes = np.arange(num_nodes, dtype=int)

        # Initial guess for vertex-based metric is superposition of adjacent element metrics
        for node_ID in nodes:
            vertex_metric_tensor[node_ID] = simplex_metric_tensor[node_to_simplex_IDs[node_ID]].sum(axis=0)

        # # Initial guess for vertex-based metric is identity matrix
        # vertex_metric_tensor[...] = np.eye(ndims)[None, :, :]

        # # Initial guess for vertex-based metric is first bordering simplex-implied metric tensor
        # for node_ID in nodes:
        #     vertex_metric_tensor[node_ID] = simplex_metric_tensor[node_to_simplex_IDs[node_ID][0]]

        # Iteratively step thru element metric-averaging gradient descent
        err = np.ones(num_nodes) * np.inf
        iteration = 1
        while len(nodes) > 0:
            print(f"{iteration=}, {len(nodes)=}")
            # Iterate over set of unconverged vertices
            Mv = vertex_metric_tensor[nodes]
            Mv_old = Mv.copy()
            Mv_sqrt = sqrtm(Mv)
            Mv_invsqrt = np.linalg.inv(Mv_sqrt)
            # Mv_invsqrt = fractional_matrix_power(Mv, -0.5)

            # Sum element metric log-distance contribution for each adjacent simplex
            metric_logsum = np.zeros([len(nodes), ndims, ndims])
            for local_ID, node_ID in enumerate(nodes):
                simplex_IDs = node_to_simplex_IDs[node_ID]
                Mk = simplex_metric_tensor[simplex_IDs]
                temp = logm(np.einsum('jk,ikl,lm->ijm', Mv_invsqrt[local_ID], Mk, Mv_invsqrt[local_ID]))
                metric_logsum[local_ID] = temp.mean(axis=0)
            
            # Update best-guess value for the vertex metric
            Mv = np.einsum('ijk,ikl,ilm->ijm', Mv_sqrt, expm(metric_logsum), Mv_sqrt)

            # Compute step norm to evaluate convergence
            vertex_metric_tensor[nodes] = Mv
            err[nodes] = np.linalg.norm(Mv - Mv_old, axis=(1, 2))
            nodes = np.where(err > min_err)[0]

            iteration += 1
            if iteration > 20:
                raise RuntimeError("Failed to converge the vertex metric calculation.")
        
        with open('vertex_metric.pkl', 'wb') as f:
            pickle.dump(vertex_metric_tensor, f)

        return vertex_metric_tensor

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

        # Get the metrics at the mesh nodes
        vertex_metric_tensor = self.get_vertex_metrics()

        print("Computing affine-invariant metric tensors for elements...")

        # Instantiate smooth Riemannian metric tensor for arbitrary element shapes
        smooth_metric_tensor = np.zeros([num_elems, ndims, ndims])
        elems = np.arange(num_elems, dtype=int)

        # Initial guess for element metric is superposition of adjacent vertex metrics
        for elem_ID in elems:
            smooth_metric_tensor[elem_ID] = vertex_metric_tensor[self.elem_to_node_IDs[elem_ID]].mean(axis=0)

        # Iteratively step thru element metric-averaging gradient descent
        err = np.ones(num_elems) * np.inf
        iteration = 1
        while len(elems) > 0:
            print(f"{iteration=}, {len(elems)=}")
            # Iterate over set of unconverged vertices
            Mk = smooth_metric_tensor[elems]
            Mk_old = Mk.copy()
            Mk_sqrt = sqrtm(Mk)
            Mk_invsqrt = np.linalg.inv(Mk_sqrt)

            # Sum element metric log-distance contribution for each adjacent simplex
            metric_logsum = np.zeros([len(elems), ndims, ndims])
            for local_ID, elem_ID in enumerate(elems):
                node_IDs = self.elem_to_node_IDs[elem_ID]
                Mv = vertex_metric_tensor[node_IDs]
                temp = np.einsum('jk,ikl,lm->ijm', Mk_invsqrt[local_ID], Mv, Mk_invsqrt[local_ID])
                metric_logsum[local_ID] = np.mean(logm(temp), axis=0)
            
            # Update best-guess value for the vertex metric
            Mk = np.einsum('ijk,ikl,ilm->ijm', Mk_sqrt, expm(metric_logsum), Mk_sqrt)

            # Compute step norm to evaluate convergence
            smooth_metric_tensor[elems] = Mk
            err[elems] = np.linalg.norm(Mk - Mk_old, axis=(1, 2))
            elems = np.where(err > min_err)[0]

            iteration += 1
            if iteration > 20:
                raise RuntimeError("Failed to converge the metric tensor calculation.")

        self.metric_tensor = smooth_metric_tensor
        self.inv_metric_tensor = np.linalg.inv(sqrtm(smooth_metric_tensor))
        self.length_scale = np.linalg.det(self.inv_metric_tensor)**(1/ndims)

        print("Metric tensor calculation complete.")
