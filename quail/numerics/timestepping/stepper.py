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
#       File : src/numerics/timestepping/stepper.py
#
#       Contains class definitions for timestepping methods.
#
# ------------------------------------------------------------------------ #
from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import fsolve, root

from quail.general import StepperType, SourceStepperType

import quail.numerics.basis.tools as basis_tools
from quail.numerics.helpers import helpers
import quail.numerics.timestepping.tools as stepper_tools
from quail.numerics.timestepping import source_stepper

import quail.solver.tools as solver_tools


class StepperBase(ABC):
    '''
    This is an abstract base class used to represent time stepping schemes.
    The current build supports the following time schemes:

        Explicit Schemes
        ----------------
        - Forward Euler (FE)
        - 4th-order Runge Kutta (RK4)
        - Low storage 4th-order Runge Kutta (LSRK4)
        - 3-stage strong-stability preserving 3rd-order Runge Kutta (SSPRK3)
        - 4-stage strong-stability preserving 3rd-order Runge Kutta (SSPRK3_4S)
        - 5-stage low-storage strong-stability preserving 3rd-order Runge Kutta (LSSSPRK3)
        - Arbitrary DERivatives in space and time (ADER)
            -> used in tandem with ADERDG solver

        Operator Splitting Type Schemes
        -------------------------------
        - Strang Splitting (Strang)
        - Simpler Splitting (Simpler)

        Source Solvers for Splitting Schemes
        ------------------------------------
        - Backward Difference (BDF1)
        - Trapezoidal Scheme (Trapezoidal)

    Abstract Constants
    ------------------
    STEPPER_TYPE
        Defines an enum from StepperType to identify the time scheme.

    Attributes:
    -----------
    res: ndarray
        Residual array of shape ``(num_elems, nb, ns)``.
    dt: float
        Time-step for the solution.
    num_time_steps: int
        Number of time steps for the given solution's FinalTime.
    get_time_step: method
        Method to obtain dt given input decks logic (CFL-based vs # of
        timesteps, etc...).
    balance_const: ndarray
        Balancing constant array of shape ``(num_elems, nb, ns)``, used only
        with the Simpler splitting scheme.

    Abstract Methods
    ----------------
    take_time_step
        Method that takes a given time step for the solver depending on the
        selected time-stepping scheme.
    '''
    @property
    @abstractmethod
    def STEPPER_TYPE(self):
        '''
        Stores the StepperType enum to define the element's timestepping
        scheme.
        '''
        pass

    def __init__(self, U):
        self.res = np.zeros_like(U)
        self.dt = 0.
        self.num_time_steps = 0
        self.get_time_step = None
        self.balance_const = None # kept as None unless set by Simpler scheme

    def __repr__(self):
        return '{self.__class__.__name__}(TimeStep={self.dt})'.format( \
            self=self)

    def __eq__(self, other):
        if not isinstance(other, StepperBase):
            # don't attempt to compare against unrelated types
            return NotImplementedError
        return self.STEPPER_TYPE == other.STEPPER_TYPE

    @abstractmethod
    def take_time_step(self, solver):
        '''
        Takes a time step using the specified time-stepping scheme for the
        solution.

        Parameters
        ----------
        solver: SolverBase
            Solver object (e.g., DG, ADERDG, etc...).

        Returns
        -------
        res: ndarray
            Updated residual vector of shape ``(num_elems, nb, ns)``.
        U: ndarray
            Updated solution vector of shape ``(num_elems, nb, ns)``.
        '''
        pass


class RungeKuttaBase(StepperBase):
    '''
    Base class for explicit Runge-Kutta schemes.
    '''
    def __init__(self, U):
        super().__init__(U)
        '''
        Additional Attributes
        ---------------------
        nstages: int
            Number of stages in scheme.
        rkcoeff: ndarray
            Coefficients for each Runge-Kutta stage, with shape
            ``(nstages, 4)``.
        dU: ndarray
            Change in solution array in each stage, with shape
            ``(num_elems, nb, ns)``.
        '''
        # Runge-Kutta coefficients on U0, U, dU, and dt:
        self.rkcoeff = None
        self.nstages = 0
        self.dU = np.zeros_like(U)

    def take_time_step(self, solver):
        mesh = solver.mesh
        physics = solver.physics
        U = solver.state_coeffs
        U0 = U.copy()

        res = self.res

        Time = solver.time + 0.0
        for istage in range(self.nstages):
            a, b, c, d = self.rkcoeff[istage]
            dt = self.dt

            res = solver.get_residual(U, res)
            dU = solver_tools.mult_inv_mass_matrix(mesh, solver, dt, res)
            solver.time = Time + d*dt

            # U = a*U0 + b*U + c*dU

            press1 = physics.compute_variable("Pressure",U)
            rho_n = physics.compute_variable("Density",U)

            # In-place update:
            U *= b
            U += a*U0 + c*dU
            solver.apply_limiter(U,rho_n,press1)

        return res # [num_elems, nb, ns]


class FE(RungeKuttaBase):
    '''
    Forward Euler (FE) method inherits attributes from RungeKuttaBase. See
    RungeKuttaBase for detailed comments of methods and attributes.
    '''
    STEPPER_TYPE = StepperType.FE

    def __init__(self, U):
        super().__init__(U)
        # Runge-Kutta coefficients on U0, U, dU, and dt:
        self.rkcoeff = np.array([[0.0, 1.0, 1.0, 1.0]])
        self.nstages = 1


class SSPRK3(RungeKuttaBase):
    '''
    3rd-order strong stability preserving Runge Kutta (SSPRK3). This scheme is
    stable for CFL <= 1.

    References
    ----------
    Dale E. Durran, “Numerical Methods for Fluid Dynamics”, Springer.
    Second Edition.
    '''
    STEPPER_TYPE = StepperType.SSPRK3

    def __init__(self, U):
        super().__init__(U)
        # Runge-Kutta coefficients on U0, U, dU, and dt:
        self.rkcoeff = np.array([[0.0, 1.0, 1.0, 1.0],
                                 [0.75, 0.25, 0.25, 0.5],
                                 [1.0/3.0, 2.0/3.0, 2.0/3.0, 1.0]])
        self.nstages = 3


class SSPRK3_4S(RungeKuttaBase):
    '''
    Four-stage 3rd-order strong stability preserving Runge Kutta (4s-SSP-RK3).
    This scheme is stable for CFL <= 2.

    References
    ----------
    Dale E. Durran, “Numerical Methods for Fluid Dynamics”, Springer.
    Second Edition.
    '''
    STEPPER_TYPE = StepperType.SSPRK3_4S

    def __init__(self, U):
        super().__init__(U)
        # Runge-Kutta coefficients on U0, U, dU, and dt:
        self.rkcoeff = np.array([[0.5, 0.5, 0.5, 0.5],
                                 [0.0, 1.0, 0.5, 1.0],
                                 [2.0/3.0, 1.0/3.0, 1.0/6.0, 0.5],
                                 [0.0, 1.0, 0.5, 1.0]])
        self.nstages = 4


class RK4(StepperBase):
    '''
    4th-order Runge Kutta (RK4) method inherits attributes from StepperBase.
    See StepperBase for detailed comments of methods and attributes.
    '''
    STEPPER_TYPE = StepperType.RK4

    def take_time_step(self, solver):
        physics = solver.physics
        mesh = solver.mesh
        U = solver.state_coeffs

        res = self.res

        # First stage
        press1 = physics.compute_variable("Pressure",U)
        rho_n = physics.compute_variable("Density",U)
        res = solver.get_residual(U, res)
        dU1 = solver_tools.mult_inv_mass_matrix(mesh, solver, self.dt, res)
        Utemp = U + 0.5*dU1
        solver.apply_limiter(Utemp,rho_n,press1)

        # Second stage
        solver.time += self.dt/2.
        press1 = physics.compute_variable("Pressure",Utemp)
        rho_n = physics.compute_variable("Density",Utemp)
        res = solver.get_residual(Utemp, res)
        dU2 = solver_tools.mult_inv_mass_matrix(mesh, solver, self.dt, res)
        Utemp = U + 0.5*dU2
        solver.apply_limiter(Utemp,rho_n,press1)

        # Third stage
        press1 = physics.compute_variable("Pressure",Utemp)
        rho_n = physics.compute_variable("Density",Utemp)
        res = solver.get_residual(Utemp, res)
        dU3 = solver_tools.mult_inv_mass_matrix(mesh, solver, self.dt, res)
        Utemp = U + dU3
        solver.apply_limiter(Utemp,rho_n,press1)

        # Fourth stage
        solver.time += self.dt/2.
        press1 = physics.compute_variable("Pressure",Utemp)
        rho_n = physics.compute_variable("Density",Utemp)
        res = solver.get_residual(Utemp, res)
        dU4 = solver_tools.mult_inv_mass_matrix(mesh, solver, self.dt, res)
        dU = 1./6.*(dU1 + 2.*dU2 + 2.*dU3 + dU4)
        U += dU
        solver.apply_limiter(U,rho_n,press1)

        return res # [num_elems, nb, ns]


class LSRK4(StepperBase):
    '''
    Low storage 4th-order Runge Kutta (RK4) method inherits attributes from
    StepperBase. See StepperBase for detailed comments of methods and
    attributes.

    References
    ----------
    M. H. Carpenter, C. Kennedy, "Fourth-order 2N-storage Runge-Kutta
    schemes," NASA Report TM 109112, NASA Langley Research Center, 1994.
    '''
    STEPPER_TYPE = StepperType.LSRK4

    def __init__(self, U):
        super().__init__(U)
        '''
        Additional Attributes
        ---------------------
        rk4a: ndarray
            Coefficients for LSRK4 scheme.
        rk4b: ndarray
            Coefficients for LSRK4 scheme.
        rk4c: ndarray
            Coefficients for LSRK4 scheme.
        nstages: int
            Number of stages in scheme.
        dU: ndarray
            Change in solution array in each stage, with shape
            ``(num_elems, nb, ns)``.
        '''
        self.rk4a = np.array([0.0, -567301805773.0/1357537059087.0,
            -2404267990393.0/2016746695238.0,
            -3550918686646.0/2091501179385.0,
            -1275806237668.0/842570457699.0])
        self.rk4b = np.array([1432997174477.0/9575080441755.0,
            5161836677717.0/13612068292357.0,
            1720146321549.0/2090206949498.0,
            3134564353537.0/4481467310338.0,
            2277821191437.0/14882151754819.0])
        self.rk4c = np.array([0.0, 1432997174477.0/9575080441755.0,
            2526269341429.0/6820363962896.0,
            2006345519317.0/3224310063776.0,
            2802321613138.0/2924317926251.0])
        self.nstages = 5
        self.dU = np.zeros_like(U)

    def take_time_step(self, solver):
        mesh = solver.mesh
        U = solver.state_coeffs

        res = self.res
        dU = self.dU

        Time = solver.time
        for istage in range(self.nstages):
            dt = self.dt

            res = solver.get_residual(U, res)
            dUtemp = solver_tools.mult_inv_mass_matrix(mesh, solver, dt, res)
            solver.time = Time + self.rk4c[istage]*dt

            dU *= self.rk4a[istage]
            dU += dUtemp

            U += self.rk4b[istage]*dU
            solver.apply_limiter(U)

        return res # [num_elems, nb, ns]


class LSSSPRK3(StepperBase):
    '''
    Low storage 3rd-order strong stability preserving Runge Kutta (SSPRK3)
    method inherits attributes from StepperBase. This scheme is stable for CFL
    <= 1. See StepperBase for detailed comments of methods and attributes.

    References
    ----------
    Spiteri, R.J. and Ruuth, S.J. "A new class of optimal high-order
    strong-stability-preserving time discretization methods". SIAM Journal on
    Numerical Analysis. Vol. 40, Num. 2, pp. 469-491. 2002
    '''
    STEPPER_TYPE = StepperType.LSSSPRK3

    def __init__(self, U):
        super().__init__(U)
        '''
        Additional Attributes
        ---------------------
        ssprk3a: ndarray
            Coefficients for SSPRK3 scheme.
        ssprk3b: ndarray
            Coefficients for SSPRK3 scheme.
        nstages: int
            Number of stages in scheme.
        dU: ndarray
            Change in solution array in each stage, with shape
            ``(num_elems, nb, ns)``.
        '''
        self.ssprk3a = np.array([0.0, -2.60810978953486, -0.08977353434746,
                -0.60081019321053, -0.72939715170280])
        self.ssprk3b = np.array([0.67892607116139, 0.20654657933371,
                0.27959340290485, 0.31738259840613, 0.30319904778284])
        self.nstages = 5
        self.dU = np.zeros_like(U)

    def take_time_step(self, solver):
        mesh = solver.mesh
        U = solver.state_coeffs

        res = self.res
        dU = self.dU

        Time = solver.time
        for istage in range(self.nstages):
            dt = self.dt

            res = solver.get_residual(U, res)
            dUtemp = solver_tools.mult_inv_mass_matrix(mesh, solver, dt, res)
            solver.time = Time + dt

            dU *= self.ssprk3a[istage]
            dU += dUtemp
            U += self.ssprk3b[istage]*dU
            solver.apply_limiter(U)

        return res # [num_elems, nb, ns]


class ADER(StepperBase):
    '''
    Arbitrary DERivatives in space and time (ADER) scheme inherits
    attributes from StepperBase. See StepperBase for detailed comments of
    methods and attributes.

    References
    ----------
    Dumbser, M., Enaux, C., and Toro, E.F., "Finite volume schemes of very
    high order of accuracy for stiff hyperbolic balance laws". Journal of
    Computational Physics. Vol. 227, Num. 8, pp. 3971 - 4001, 2008.
    '''
    STEPPER_TYPE = StepperType.ADER

    def take_time_step(self, solver):
        physics = solver.physics
        mesh = solver.mesh
        W = solver.state_coeffs
        Up = solver.state_coeffs_pred

        res = self.res

        # Prediction step
        Up = solver.calculate_predictor_step(solver, self.dt, W, Up)
        # Correction step
        res = solver.get_residual(Up, res)

        dU = solver_tools.mult_inv_mass_matrix(mesh, solver, self.dt/2., res)

        W += dU
        solver.apply_limiter(W)

        solver.state_coeffs_pred = Up

        return res # [num_elems, nb, ns]


class Strang(StepperBase, source_stepper.SourceSolvers):
    '''
    The Strang operator splitting scheme inherits attributes from
    StepperBase and SourceSolvers (in source_stepper.py). See StepperBase
    and SourceSolvers for detailed comments of methods and attributes.

    Additional Attributes
    ---------------------
    explicit: StepperBase
        Stepper object instantiation for explicit scheme.
    implicit: SourceStepperBase
        Stepper object instantiation for ODE solver.

    References
    ----------
    Strang, G. "On the Construction and Comparison of Difference Schemes".
    SIAM Journal of Numerical Analysis. Vol. 5, Num. 3, 1968.
    '''
    STEPPER_TYPE = StepperType.Strang

    def set_split_schemes(self, explicit, implicit, U):
        '''
        Specifies the explicit and implicit schemes to be used in the
        operator splitting technique.

        Parameters
        ----------
        explicit: str
            Name of chosen explicit scheme from params.
        implicit: str
            Name of chosen implicit (ODE) solver from params.
        U: ndarray
            Solution state vector used to initialize solver, with shape
            ``(num_elems, nb, ns)``.
        '''
        param = {"TimeStepper": explicit}
        # call set_stepper from stepper tools for the explicit scheme
        self.explicit = stepper_tools.set_stepper(param, U)

        if SourceStepperType[implicit] == SourceStepperType.BDF1:
            self.implicit = source_stepper.SourceSolvers.BDF1(U)
        elif SourceStepperType[implicit] == SourceStepperType.Trapezoidal:
            self.implicit = source_stepper.SourceSolvers.Trapezoidal(U)
        elif SourceStepperType[implicit] == SourceStepperType.LSODA:
            self.implicit = source_stepper.SourceSolvers.LSODA(U)
        else:
            raise NotImplementedError("Time scheme '%s' not supported."
                                      % implicit)

    def take_time_step(self, solver):
        physics = solver.physics
        mesh  = solver.mesh
        U = solver.state_coeffs

        # Set the appropriate time steps for each operation
        explicit = self.explicit
        explicit.dt = self.dt/2.
        implicit = self.implicit
        implicit.dt = self.dt

        # Force SourceSwitch ON for splitting schemes
        solver.params["SourceSwitch"] = True

        # First: take the half-step for the inviscid flux only
        solver.params["ConvFluxSwitch"] = True
        physics.source_terms = physics.explicit_sources.copy()
        explicit.take_time_step(solver)

        # Second: take the implicit full step for the source term.
        solver.params["ConvFluxSwitch"] = False
        physics.source_terms = physics.implicit_sources.copy()
        implicit.take_time_step(solver)

        # Third: take the second half-step for the inviscid flux only.
        physics.source_terms = physics.explicit_sources.copy()

        solver.params["ConvFluxSwitch"] = True
        R = explicit.take_time_step(solver)

        return R # [num_elems, nb, ns]


class Simpler(Strang):
    '''
    The Simpler balanced operator splitting scheme inherits attributes from
    Strang. See Strang for detailed comments of methods and attributes.

    References
    ----------
    Wu, H., Ma, P., and Ihme, M. "Efficient time-stepping techniques for
    simulating turbulent reactive flows with stiff chemistry". Computer
    Physics Communications. Vol. 243, pp. 81 - 96, 2019.
    '''
    STEPPER_TYPE = StepperType.Simpler

    def take_time_step(self, solver):
        physics = solver.physics
        mesh  = solver.mesh
        U = solver.state_coeffs

        # Set the appropriate time steps for each operation
        explicit = self.explicit
        explicit.dt = self.dt/2.
        implicit = self.implicit
        implicit.dt = self.dt

        # Force SourceSwitch ON for splitting schemes
        solver.params["SourceSwitch"] = True
        res = self.res

        # First: calculate the balance constant
        # Note: we skip the first explicit step as it is in equilibrium by
        # definition
        physics.source_terms = physics.explicit_sources.copy()

        self.balance_const = None
        balance_const = -1.*solver.get_residual(U, res)
        self.balance_const = -1.*balance_const

        # Second: take the implicit full step for the source term.
        solver.params["ConvFluxSwitch"] = False
        physics.source_terms = physics.implicit_sources.copy()
        implicit.take_time_step(solver)

        # Third: take the second half-step for the inviscid flux only.
        solver.params["ConvFluxSwitch"] = True
        physics.source_terms = physics.explicit_sources.copy()
        self.balance_const = balance_const
        R3 = explicit.take_time_step(solver)

        return R3 # [num_elems, nb, ns]


class ODEIntegrator(StepperBase, source_stepper.SourceSolvers):
    '''
    ODEIntegrator method inherits attributes from StepperBase and
    source_stepper.SourceSolvers. It constructs an interface for users
    to utilize the various time integration schemes in Quail directly
    for ODEs and systems of ODEs.

    Additional Attributes
    ---------------------
    ode_integrator: StepperBase or SourceStepperBase
        Object stored in self that contains the ode time integration scheme.
    '''
    STEPPER_TYPE = StepperType.ODEIntegrator

    def set_ode_integrator(self, ode_scheme, U):
        '''
        Sets the ode integrator from the list of available time integration
        schemes.

        Parameters
        ----------
        ode_scheme: str
            Name of chosen scheme from params.
        U: ndarray
            Solution state vector used to initialize solver, with shape
            ``(num_elems, nb, ns)``.
        '''
        try:
            stepper = StepperType[ode_scheme]
        except:
            pass
            try:
                stepper = SourceStepperType[ode_scheme]
            except:
                raise NotImplementedError("ODE time scheme '" + ode_scheme +
                                          "' is not supported.")

        if stepper == StepperType.FE:
            ode_integrator = FE(U)
        elif stepper == StepperType.RK4:
            ode_integrator = RK4(U)
        elif stepper == StepperType.LSRK4:
            ode_integrator = LSRK4(U)
        elif stepper == StepperType.SSPRK3:
            ode_integrator = SSPRK3(U)
        elif stepper == StepperType.SSPRK3_4S:
            ode_integrator = SSPRK3_4S(U)
        elif stepper == StepperType.LSSSPRK3:
            ode_integrator = LSSSPRK3(U)
        elif stepper == StepperType.ADER:
            ode_integrator = StepperType.ADER(U)
        elif stepper == SourceStepperType.BDF1:
            ode_integrator = source_stepper.SourceSolvers.BDF1(U)
        elif stepper == SourceStepperType.Trapezoidal:
            ode_integrator = source_stepper.SourceSolvers.Trapezoidal(U)
        elif stepper == SourceStepperType.LSODA:
            ode_integrator = source_stepper.SourceSolvers.LSODA(U)

        self.ode_integrator = ode_integrator

    def take_time_step(self, solver):
        self.ode_integrator.dt = self.dt
        R = self.ode_integrator.take_time_step(solver)

        return R
