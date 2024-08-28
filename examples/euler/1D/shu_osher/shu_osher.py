import numpy as np
import copy

FinalTime = 1.8
NumTimeSteps = 200

TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : FinalTime,
    "CFL" : 0.2,
    "TimeStepper" : "SSPRK3",
}

# Output = {
#     "Prefix": "ShuOsher",
#     "WriteInterval" : 10,
#     "WriteInitialSolution": True,
#     "WriteFinalSolution": True
#             }

Numerics = {
    "SolutionOrder" : 3,
    "SolutionBasis" : "LagrangeSeg",
    # "SolutionBasis" : "LegendreSeg",
    "Solver" : "DG",
    "ApplyLimiters" : "EntropyPreserving",
    # "ApplyLimiters" : "PositivityPreserving",
    # "ApplyLimiters" : "WENO",
}

Output = {
    "AutoPostProcess" : True,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : 200,
    "xmin" : -5.,
    "xmax" : 5.,
}


Physics = {
    "Type" : "Euler",
    "ConvFluxNumerical" : "Roe",
    "GasConstant" : 1.,
    "SpecificHeatRatio" : 1.4,
}

xshock = -4.0
InitialCondition = {
    "Function": "ShuOsherProblem",
    "xshock": xshock,
}

# ExactSolution = InitialCondition.copy()
# ExactSolution = None
BoundaryConditions = {
    "x1" : {
        "BCType" : "Extrapolate"
        },
    "x2" : {
        "BCType" : "Extrapolate"
        }
}
