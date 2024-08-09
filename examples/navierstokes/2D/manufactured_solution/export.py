import quail.processing.export as export
import quail.processing.readwritedatafiles as readwritedatafiles

# Read data file
fname = 'Data_final.pkl'
solver = readwritedatafiles.read_data_file(fname)

# Unpack
mesh = solver.mesh
physics = solver.physics

# Export
export.export_cells(mesh,physics,solver,file_name='cell_data')
export.export_points(mesh,physics,solver,file_name='point_data')
