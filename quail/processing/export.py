# ------------------------------------------------------------------------ #
#
#       File : src/processing/export.py
#
#       Contains functions for exporting VTK files for 1D and 2D.
#
# ------------------------------------------------------------------------ #
import numpy as np
import math

import quail.numerics.helpers.helpers as helpers
import quail.solver.tools as solver_tools
import quail.meshing.tools as mesh_tools
import quail.processing.plot as plot

from matplotlib import tri
from pyevtk.vtk import VtkTriangle,VtkQuad
from pyevtk.hl import unstructuredGridToVTK


# Export cell-centered values
def export_cells(mesh,physics,solver,file_name='export',save_AV=False,csv_export=False):
	'''
	This function exports cell-centered data and mesh coordinates to a VTK file,
	or optionally, a CSV file.

	Inputs:
	-------
	    mesh: mesh object
	    physics: physics object
	    solver: solver object
	    file_name: name of VTK and CSV files that are produced
	    save_AV: if True, calculate and save artificial viscosity values
	    csv_export: if True, export variables and cell centers to CSV
	'''
	if physics.NDIMS != 2:
		raise ValueError

	# Setup mesh and grid
	# The vertices of each element must be sorted counterclockwise
	coords = mesh.node_coords
	elem_points = mesh.elem_to_node_IDs
	node_list = []
	for elem in elem_points:
		idx = sort_points(coords[elem])
		new_elem = elem[idx]
		node_list.append(new_elem)
	    
	# These are the "points" of each "cell" (element)
	# Copy the variable to set C-order flags
	x_vertices = coords[:,0].copy()
	y_vertices = coords[:,1].copy()
	z_vertices = np.zeros_like(x_vertices)
	
	# Connectivity matrix and offset matrix
	cvec = np.array(node_list).reshape(-1).copy()
	ovec = np.arange(1,mesh.num_elems+1)*mesh.num_nodes_per_elem
	
	# Cell types
	# Currently only tri and quads... add more later
	ctypes = np.zeros(mesh.num_elems)
	ctypes[:] = VtkQuad.tid if mesh.num_nodes_per_elem == 4 else VtkTriangle.tid
	
	# Element-averaged solution variables
	elem_helpers = solver.elem_helpers
	basis_val = elem_helpers.basis_val
	quad_wts = elem_helpers.quad_wts
	djacs = elem_helpers.djac_elems
	vols = elem_helpers.vol_elems
	Uc = solver.state_coeffs
	Uq = helpers.evaluate_state(Uc,basis_val)
	Ubar = helpers.get_element_mean(Uq,quad_wts,djacs,vols)

	# Variables of interest
	pressure = physics.compute_variable("Pressure",Ubar).reshape(-1)
	temperature = physics.compute_variable("Temperature",Ubar).reshape(-1)
	xvel = physics.compute_variable("XVelocity",Ubar).reshape(-1)
	yvel = physics.compute_variable("YVelocity",Ubar).reshape(-1)
	rho = physics.compute_variable("Density",Ubar).reshape(-1)

	# Export to CSV
	if csv_export:
		xc = []
		yc = []
		for elem_ID in range(mesh.num_elems):
			centroid = mesh_tools.get_element_centroid(mesh,elem_ID)
			xc.append(centroid[0,0])
			yc.append(centroid[0,1])
		np.savetxt(file_name+'.csv',np.c_[xc,yc,
			pressure,
			temperature,
			xvel,
			yvel,
			rho
		],delimiter=',',header='x,y,Pressure,Temperature,XVelocity,YVelocity,Density')
		
	# Pack up all the data
	solution_data = {
		"Pressure":pressure,
		"Temperature":temperature,
		"XVelocity":xvel,
		"YVelocity":yvel,
		"Density":rho
	}
	
	# If AV output is desired, calculate it
	if save_AV:
		U_bar = helpers.get_element_mean(Uq,quad_wts,djacs,vols)
		a_max = physics.compute_variable("MaxWaveSpeed",U_bar).reshape(-1,1,1)
		try:
			try:
				avc = solver_tools.calculate_artificial_viscosity(solver, solver.mesh)
			# If there is not an existing AV solver, need to initialize
			except AttributeError:
				solver.av_solver = solver_tools.initialize_artificial_viscosity(solver)
				avc = solver_tools.calculate_artificial_viscosity(solver, solver.mesh)
			av = helpers.evaluate_state(avc, solver.basis.basis_val)
			av = helpers.get_element_mean(av,quad_wts,djacs,vols)
			av2 = av/a_max/0.25
		except Exception as e:
			print("Unable to calculate AV. Continuing...")
		
		# Add to the solution data dictionary
		solution_data.update({"AV":av,"NormalizedAV":av2})

	# Export to VTK	
	unstructuredGridToVTK(
	    file_name,
	    x_vertices,
	    y_vertices,
	    z_vertices,
	    connectivity=cvec,
	    offsets=ovec,
	    cell_types=ctypes,
	    cellData = solution_data
	)
	
# Export nodal values and sampled points
def export_points(mesh,physics,solver,file_name='export',save_AV=False, equidistant=False):
	'''
	This function exports triangulated plottable data to a VTK file.

	Inputs:
	-------
	    mesh: mesh object
	    physics: physics object
	    solver: solver object
	    file_name: name of VTK and CSV files that are produced
	'''
	if physics.NDIMS != 2:
		raise ValueError

	# Unpack
	basis = solver.basis
	Uc = solver.state_coeffs

	# If AV output is desired, calculate it
	if save_AV:
		elem_helpers = solver.elem_helpers
		basis_val = elem_helpers.basis_val
		quad_wts = elem_helpers.quad_wts
		djacs = elem_helpers.djac_elems
		vols = elem_helpers.vol_elems

		# Compute AV
		Uq = helpers.evaluate_state(Uc, basis_val)
		U_bar = helpers.get_element_mean(Uq, quad_wts, djacs, vols)
		a_max = physics.compute_variable("MaxWaveSpeed", U_bar).reshape(-1, 1, 1)
		try:
			avc = solver_tools.calculate_artificial_viscosity(solver, mesh)
		except AttributeError:
			solver.av_solver = solver_tools.initialize_artificial_viscosity(solver)
			avc = solver_tools.calculate_artificial_viscosity(solver, mesh)
		except Exception as e:
			print(e)
			print("Unable to calculate AV. Continuing...")

	# Get quadrature points for computation
	# For some reason this must be done AFTER the AV calculation and BEFORE the state evaluation
	# coords = plot.get_sample_points(mesh, solver, physics, solver.basis, equidistant)
	coords = get_sample_points(mesh,solver,physics,solver.basis,equidistant)
		
	# Variables of interest - they are pointwise
	pressure = plot.get_numerical_solution(physics,Uc,coords,basis,"Pressure").reshape(-1)
	temperature = plot.get_numerical_solution(physics,Uc,coords,basis,"Temperature").reshape(-1)
	xvel = plot.get_numerical_solution(physics,Uc,coords,basis,"XVelocity").reshape(-1)
	yvel = plot.get_numerical_solution(physics,Uc,coords,basis,"YVelocity").reshape(-1)
	rho = plot.get_numerical_solution(physics,Uc,coords,basis,"Density").reshape(-1)

	# Pack up all the data
	pData = {
		"Pressure":pressure,
		"Temperature":temperature,
		"XVelocity":xvel,
		"YVelocity":yvel,
		"Density":rho
	}

	# Include the AV data if desired
	if save_AV:
		try:
			av = helpers.evaluate_state(avc, solver.basis.basis_val)
			av2 = av/a_max/0.25
			av = av.reshape(-1)
			av2 = av2.reshape(-1)
			pData.update({"AV":av,"NormalizedAV":av2})
		except Exception as e:
			print(e)
			print("Unable to save AV. Continuing...")
	
	# Setup the mesh and the grid
	# Variables must be sorted counterclockwise
	# After plotter update, need to update result from get_sample_points
	new_coords = coords.reshape(-1,coords.shape[2])
        
	node_list = []
	for elem_ID in range(len(coords)):
		triangles = tri.Triangulation(coords[elem_ID][:,0],coords[elem_ID][:,1]).triangles
		global_offset = elem_ID*len(coords[0])
		for i in triangles:
			i = np.array(i)
			idx = sort_points(new_coords[i])
			new_elem = i[idx]
			new_elem = [x+global_offset for x in new_elem]
			node_list.append(new_elem)
	node_list = np.array(node_list)
	cvec = np.array(node_list).reshape(-1).copy()
	ovec = np.arange(1,len(node_list)+1)*3 #triangles

	ctypes = np.zeros(len(node_list))
	ctypes[:] = VtkTriangle.tid

	# These are the "points" of each "cell" (element)
	# Copy the variable to set C-order flags
	x_vertices = new_coords[:,0].copy()
	y_vertices = new_coords[:,1].copy()
	z_vertices = np.zeros_like(x_vertices)

	# Export to VTK
	unstructuredGridToVTK(
	    	file_name,
	    	x_vertices,
	    	y_vertices,
	    	z_vertices,
	        connectivity=cvec,
	        offsets=ovec,
	        cell_types=ctypes,
	        pointData = pData
	)
	
# Counterclockwise point sorter
def sort_points(point_set,refvec=[1,0]):
	'''
	This function sorts a set of points in counterclockwise order
	around a reference vector.

	Inputs:
	-------
		point_set: A set of coordinates describing an element
	    refvec: Reference vector defining the counterclockwise direction.
				If not provided, defaults to using the x-axis [1,0].

	Outputs:
	-------
		idx: Indices describing the sorting order.
	'''
	origin = point_set.mean(axis=0)
	def clockwiseangle_and_distance(point):
		vector = [point[0]-origin[0], point[1]-origin[1]]
		lenvector = math.hypot(vector[0], vector[1])
		if lenvector == 0:
			return -math.pi, 0
		normalized = [vector[0]/lenvector, vector[1]/lenvector]
		dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]   
		diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]
		angle = math.atan2(diffprod, dotprod)
		if angle < 0:
			return 2*math.pi+angle, lenvector
		return angle, lenvector
	idx = [i for i,x in sorted(enumerate(point_set), key=lambda x: clockwiseangle_and_distance(x[1]),reverse=True)]
	return idx

# Copy paste get_sample_points from plot without triangulation
def get_sample_points(mesh, solver, physics, basis, equidistant=False):
    '''
    This function returns sample points at which to evaluate the solution
    for plotting.

    Inputs:
    -------
        mesh: mesh object
        physics: physics object
        basis: basis object
        equidistant: if True, then sample points will be equidistant
            (within each element); if False, sample points will be based
            on quadrature points

    Outputs:
    -------
        x: sample points [num_elems, num_pts, ndims], where num_pts is the
            number of sample points per element
    '''
    # Extract
    ndims = mesh.ndims
    order = solver.order

    # Get sample points in reference space
    if equidistant:
        xref = basis.equidistant_nodes(max([1, 3*order]))
    else:
        quad_order = basis.get_quadrature_order(mesh,order,physics=physics)
        xref, _ = mesh.gbasis.get_quadrature_data(quad_order)

    # Allocate
    num_pts = xref.shape[0]
    x = np.zeros([mesh.num_elems, num_pts, ndims])

    # Evaluate basis at reference-space points
    basis.get_basis_val_grads(xref, True, False, False, None)

    # Convert reference-space points to physical space
    for elem_ID in range(mesh.num_elems):
        xphys = mesh_tools.ref_to_phys(mesh, elem_ID, xref)
        x[elem_ID, :, :] = xphys

    return x