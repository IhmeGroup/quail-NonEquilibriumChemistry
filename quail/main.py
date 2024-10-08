#!/usr/bin/env python
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
#       File : src/quail.py
#
#       Contains driver and helper functions for Quail.
#
# ------------------------------------------------------------------------ #
import argparse
import importlib
import numpy as np
import os
import sys

import quail.defaultparams as default_deck
from quail import errors
from quail.general import ShapeType, SolverType, PhysicsType, ThermoType, TransportType

import quail.meshing.common as mesh_common
import quail.meshing.gmsh as mesh_gmsh
import quail.meshing.tools as mesh_tools

import quail.numerics.timestepping.tools as stepper_tools

import quail.physics.zerodimensional.zerodimensional as zerod
from quail.physics.euler import euler
from quail.physics.navierstokes import navierstokes
from quail.physics.scalar import scalar
from quail.physics.chemistry import chemistry

import quail.physics.base.thermo as thermo_tools
import quail.physics.base.transport as transport_tools

from quail.processing import readwritedatafiles

from quail.solver import DG
from quail.solver import ADERDG


def set_thermo(ThermoModel='NotNeeded', **kwargs):
    '''
    Given the Thermodynamics parameter, set the get_transport function

    Inputs:
    -------
        transport_type: string to determine transport type

    Outputs:
    --------
        fcn: name of function to be passed
    '''

    if ThermoType[ThermoModel] == ThermoType.CaloricallyPerfectGas:
        return thermo_tools.CaloricallyPerfectGas(**kwargs)
    elif ThermoType[ThermoModel] == ThermoType.Cantera:
        return thermo_tools.CanteraThermo(**kwargs)
    elif ThermoType[ThermoModel] == ThermoType.Mutationpp:
        return thermo_tools.MutationppThermo(**kwargs)
    elif ThermoType[ThermoModel] == ThermoType.NotNeeded:
        return None
    else:
        raise NotImplementedError("Thermodynamics not supported")


def set_transport(TransportModel='NotNeeded', **kwargs):
    '''
    Given the Transport parameter, set the get_transport function

    Inputs:
    -------
        transport_type: string to determine transport type

    Outputs:
    --------
        fcn: name of function to be passed
    '''

    if TransportType[TransportModel] == TransportType.Constant:
        return transport_tools.ConstantTransport(**kwargs)
    elif TransportType[TransportModel] == TransportType.Sutherland:
        return transport_tools.SutherlandTransport(**kwargs)
    elif TransportType[TransportModel] == TransportType.Cantera:
        return transport_tools.CanteraTransport(**kwargs)
    elif TransportType[TransportModel] == TransportType.Mutationpp:
        return transport_tools.MutationppTransport(**kwargs)
    elif TransportType[TransportModel] == TransportType.NotNeeded:
        return None
    else:
        raise NotImplementedError("Transport not supported")


def set_physics(mesh, physics_type, thermo, transport):
    '''
    This function creates the physics object based on the input parameters.

    Inputs:
    -------
        order: order of solution approximation
        basis_type: solution basis type
        mesh: mesh object
        physics_type: desired physics type

    Outputs:
    --------
        physics: physics object
    '''
    ndims = mesh.ndims

    if PhysicsType[physics_type] == PhysicsType.ConstAdvScalar and \
            ndims == 1:
        physics_class = scalar.ConstAdvScalar1D
    elif PhysicsType[physics_type] == PhysicsType.ConstAdvScalar and \
            ndims == 2:
        physics_class = scalar.ConstAdvScalar2D
    elif PhysicsType[physics_type] == PhysicsType.ConstAdvDiffScalar and \
            ndims == 1:
        physics_class = scalar.ConstAdvDiffScalar1D
    elif PhysicsType[physics_type] == PhysicsType.ConstAdvDiffScalar and \
            ndims == 2:
        physics_class = scalar.ConstAdvDiffScalar2D
    elif PhysicsType[physics_type] == PhysicsType.Burgers and ndims == 1:
        physics_class = scalar.Burgers1D
    elif PhysicsType[physics_type] == PhysicsType.ModelProblem:
        physics_class = zerod.ModelProblem
    elif PhysicsType[physics_type] == PhysicsType.ModelPSRScalar:
        physics_class = zerod.ModelPSRScalar
    elif PhysicsType[physics_type] == PhysicsType.MultispeciesPSR:
        physics_class = zerod.MultispeciesPSR
    elif PhysicsType[physics_type] == PhysicsType.Pendulum:
        physics_class = zerod.Pendulum
    elif PhysicsType[physics_type] == PhysicsType.Euler:
        physics_class = euler.Euler
    elif PhysicsType[physics_type] == PhysicsType.NavierStokes:
        physics_class = navierstokes.NavierStokes
    elif PhysicsType[physics_type] == PhysicsType.Chemistry and ndims ==1:
        physics_class = chemistry.Chemistry1D
    else:
        raise NotImplementedError

    physics = physics_class(thermo=thermo, transport=transport, NDIMS=ndims)

    return physics


def overwrite_params(params, params_new, allow_new_keys=False):
    '''
    This function overwrites default parameters in the params dict.

    Inputs:
    -------
        params: dict with values to be overwritten
        params_new: dict with desired values
        allow_new_keys: if True, then new keys may be added to params

    Outputs:
    --------
        params: dict with values to be overwritten (modified)
    '''
    if params_new is None:
        return params

    for key in params_new:
        if not allow_new_keys and key not in params.keys():
            raise KeyError
        params[key] = params_new[key]

    return params


def read_inputs(deck):
    '''
    This function reads in the input deck and overwrites the default
    parameters.

    Inputs:
    -------
        deck: input deck

    Outputs:
    --------
        deck: input deck (modified)
    '''
    # Defaults
    restart_params = default_deck.Restart.copy()
    stepper_params = default_deck.TimeStepping.copy()
    numerics_params = default_deck.Numerics.copy()
    mesh_params = default_deck.Mesh.copy()
    physics_params = default_deck.Physics.copy()
    IC_params = default_deck.InitialCondition.copy()
    exact_params = default_deck.ExactSolution.copy()
    BC_params = default_deck.BoundaryConditions.copy()
    source_params = default_deck.SourceTerms.copy()
    output_params = default_deck.Output.copy()

    # Overwrite
    try:
        restart_params = overwrite_params(restart_params, deck.Restart)
    except AttributeError:
        pass
    try:
        stepper_params = overwrite_params(stepper_params, deck.TimeStepping)
    except AttributeError:
        pass
    try:
        numerics_params = overwrite_params(numerics_params, deck.Numerics)
    except AttributeError:
        pass
    try:
        mesh_params = overwrite_params(mesh_params, deck.Mesh)
    except AttributeError:
        pass
    try:
        physics_params = overwrite_params(physics_params, deck.Physics, True)
    except AttributeError:
        pass
    try:
        IC_params = overwrite_params(IC_params, deck.InitialCondition, True)
    except AttributeError:
        pass
    try:
        exact_params = overwrite_params(exact_params, deck.ExactSolution,
                True)
    except AttributeError:
        pass
    try:
        BC_params = overwrite_params(BC_params, deck.BoundaryConditions,
                True)
    except AttributeError:
        pass
    try:
        source_params = overwrite_params(source_params, deck.SourceTerms,
                True)
    except AttributeError:
        pass
    try:
        output_params = overwrite_params(output_params, deck.Output)
    except AttributeError:
        pass

    return restart_params, stepper_params, numerics_params, mesh_params, \
            physics_params, IC_params, exact_params, BC_params, \
            source_params, output_params


def print_info(restart_params, stepper_params, numerics_params, mesh_params,
        physics_params, IC_params, exact_params, BC_params, source_params,
        output_params):
    print()
    print("=================================================")
    print("||                                             ||")
    print("||  Quail: a discontinuous Galerkin solver in  ||")
    print("||     Python for teaching and prototyping     ||")
    print("||                                             ||")
    print("=================================================")
    print()

    def print_dict(d):
        [print("   ", key, ":", value) for key, value in d.items()]
        print()

    # Print input deck
    if output_params["Verbose"]:
        print("-------------------")
        print("PRINTING INPUT DECK")
        print("-------------------")
        print()
        print("Restart:")
        print("--------")
        print_dict(restart_params)
        print("TimeStepping:")
        print("-------------")
        print_dict(stepper_params)
        print("Numerics:")
        print("---------")
        print_dict(numerics_params)
        print("Mesh:")
        print("-----")
        print_dict(mesh_params)
        print("Physics:")
        print("--------")
        print_dict(physics_params)
        print("InitialCondition:")
        print("-----------------")
        print_dict(IC_params)
        print("ExactSolution:")
        print("--------------")
        print_dict(exact_params)
        print("BoundaryConditions:")
        print("-------------------")
        print_dict(BC_params)
        print("SourceTerms:")
        print("------------")
        print_dict(source_params)
        print("Output:")
        print("-------")
        print_dict(output_params)


def driver(deck):
    '''
    This function processes the input deck and performs the simulation.

    Inputs:
    -------
        deck: input deck

    Outputs:
    --------
        solver: solver object
        physics: physics object
        mesh: mesh object
    '''
    '''
    Input deck
    '''
    restart_params, stepper_params, numerics_params, mesh_params, \
            physics_params, IC_params, exact_params, BC_params, \
            source_params, output_params = read_inputs(deck)
    # Print info
    print_info(restart_params, stepper_params, numerics_params, mesh_params,
            physics_params, IC_params, exact_params, BC_params,
            source_params, output_params)

    '''
    Mesh
    '''
    if mesh_params["File"] is not None:
        # Gmsh file
        mesh = mesh_gmsh.import_gmsh_mesh(mesh_params["File"])
    elif mesh_params["Mesh"] is not None:
        mesh = mesh_params["Mesh"]
    else:
        # Create our own mesh

        # Unpack
        shape = ShapeType[mesh_params["ElementShape"]]
        xmin = mesh_params["xmin"]
        xmax = mesh_params["xmax"]
        num_elems_x = mesh_params["NumElemsX"]
        num_elems_y = mesh_params["NumElemsY"]
        ymin = mesh_params["ymin"]
        ymax = mesh_params["ymax"]

        # Create mesh
        if shape is ShapeType.Segment:
            # 1D - segments
            mesh = mesh_common.mesh_1D(num_elems=num_elems_x,
                    xmin=xmin, xmax=xmax)
        else:
            # 2D - quads or tris

            # First start with quads
            mesh = mesh_common.mesh_2D(num_elems_x=num_elems_x,
                    num_elems_y=num_elems_y, xmin=xmin, xmax=xmax,
                    ymin=ymin, ymax=ymax)
            # Split into tris if required
            if shape is ShapeType.Triangle:
                mesh = mesh_common.split_quadrils_into_tris(mesh)

    ''' Impose periodicity if requested '''
    pb_x = mesh_params["PeriodicBoundariesX"]
    pb_y = mesh_params["PeriodicBoundariesY"]

    # Store periodic boundaries in pb
    pb = [None]*4
    if pb_x != []:
        pb[:2] = pb_x
    if pb_y != []:
        pb[2:] = pb_y

    # Make periodic
    if pb != [None]*4:
        mesh_tools.make_periodic_translational(mesh, x1=pb[0], x2=pb[1],
                y1=pb[2], y2=pb[3])

    if numerics_params["ArtificialViscosity"]:
        # Need to compute mesh metrics for AV
        mesh.create_metric_tensor()


    '''
    Physics
    '''
    # Get order and basis type
    order = numerics_params["SolutionOrder"]
    basis_type = numerics_params["SolutionBasis"]

    # Initialize equation of state
    thermo = set_thermo(**physics_params)

    # Initialize transport properties
    transport = set_transport(**physics_params)

    # Create physics object
    physics = set_physics(mesh, physics_params["Type"], thermo, transport)

    # Set parameters
    conv_flux_type = physics_params["ConvFluxNumerical"]
    diff_flux_type = physics_params["DiffFluxNumerical"]

    physics.set_conv_num_flux(conv_flux_type)
    physics.set_diff_num_flux(diff_flux_type)
    physics.set_physical_params(**physics_params)

    # Initial condition
    iparams = IC_params.copy()
    IC_type = iparams.pop("Function")
    physics.set_IC(IC_type=IC_type, **iparams)

    # Exact solution
    if bool(exact_params): # checks if dictionary is not empty
        eparams = exact_params.copy()
        exact_type = eparams.pop("Function")
        physics.set_exact(exact_type=exact_type, **eparams)

    # Boundary conditions
    physics.BCs = dict.fromkeys(mesh.boundary_groups.keys())

    for bname, bparams in BC_params.items():
        bparams = bparams.copy()
        BC_type = bparams.pop("BCType")

        try:
            # Function required for StateAll
            fcn_type = bparams.pop("Function")
            physics.set_BC(bname, BC_type, fcn_type, **bparams)
        except KeyError:
            physics.set_BC(bname, BC_type, **bparams)

    # Source terms
    for sparams in source_params.values():
        sname = sparams["Function"]
        sparams.pop("Function")
        physics.set_source(source_type=sname, **sparams)

    '''
    Solver
    '''
    # Merge solver-related params
    solver_params = {**stepper_params, **numerics_params, **output_params}
    solver_params["RestartFile"] = restart_params["File"]
    solver_type = solver_params.pop("Solver")
    if SolverType[solver_type] is SolverType.DG:
        solver = DG.DG(solver_params, physics, mesh)
    elif SolverType[solver_type] is SolverType.ADERDG:
        solver = ADERDG.ADERDG(solver_params, physics, mesh)
    else:
        raise NotImplementedError

    '''
    Restart file
    '''
    if restart_params["File"] is not None:
        # Old solver
        solver_old = readwritedatafiles.read_data_file(solver_params[
                "RestartFile"])
        # Project if different basis and/or order
        if order != solver_old.order or solver.basis.BASIS_TYPE != \
                solver_old.basis.BASIS_TYPE:
            print("Projecting to a different solution basis and/or order")
            solver.project_state_to_new_basis(solver_old.state_coeffs,
                    solver_old.basis, solver_old.order)
        else:
            solver.state_coeffs = solver_old.state_coeffs
        # Start from the same time and iteration count
        if restart_params["StartFromFileTime"]:
            solver.time = solver_old.time
            solver.itime = solver_old.itime
            solver.itime_initial = solver_old.itime
            solver.stepper.dt = 0.
            stepper_tools.set_time_stepping_approach(solver.stepper,
                    solver.params)
            solver.stepper.num_time_steps += solver_old.stepper.num_time_steps


    '''
    Run simulation
    '''
    solver.solve()

    return solver, physics, mesh


def process_post_file(post_file, auto_process):
    '''
    This function processes and potentially runs the post-processing file.

    Inputs:
    -------
        post_file: name of post-processing file
        auto_process: if True, will automatically run the post-processing
            script at the end of the simulation
    '''
    if post_file is None:
        if not auto_process:
            return
        post_file = "post_process"

    post_file = post_file.replace(".py","")

    try:
        print("\nRunning post-processing script")
        postprocess = importlib.import_module(post_file)
    except ModuleNotFoundError:
        raise errors.FileReadError(f"{post_file}.py not found")


def main():
    '''
    This is the top-level main function of quail.
    '''
    ''' Parser '''
    my_parser = argparse.ArgumentParser(conflict_handler="resolve",
            description="This script is the driver for Stanford\'s Quail " +
            "solver")

    ''' Command-line arguments '''
    # Input file
    my_parser.add_argument("inputdeck", type=str,
            help="this file contains all requested parameters for " +
            "the solver", nargs='?')
    # Post-processing script (optional)
    my_parser.add_argument("-p", "--post", type=str,
            help="post-processing script to execute")

    ''' Process arguments '''
    args = my_parser.parse_args()

    input_deck = args.inputdeck
    post_file = args.post

    if input_deck is None and post_file is None:
        raise Exception("At least one of the input deck and the " +
                "post-processing script is required")

    ''' Set current directory '''
    if input_deck is not None:
        file = input_deck
    else:
        file = post_file

    current_dir = os.path.dirname(os.path.abspath(file)) + "/"
    sys.path.append(current_dir)

    ''' Run	'''
    if input_deck is not None:
        # Run solver

        # Process deck
        input_deck = input_deck.replace(".py", "")
        deck = importlib.import_module(input_deck)
        # Run
        solver, physics, mesh = driver(deck)
        # Process post-processing script
        auto_process = solver.params["AutoPostProcess"]
        process_post_file(post_file, auto_process)
    else:
        # Post-process only
        process_post_file(post_file, True)

    print()
    print("----------")
    print("Quail done")
    print("----------")


if __name__ == "__main__":
    main()






