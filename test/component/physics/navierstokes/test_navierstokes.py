import numpy as np

import quail.physics.navierstokes.navierstokes as navierstokes
import quail.physics.base.thermo as thermo
import quail.physics.base.transport as transport

rtol = 1e-15
atol = 1e-15


def test_diffusion_flux_2D():
	'''
	This tests the diffusive flux for a 2D case.
	'''
	physics = navierstokes.NavierStokes(thermo=thermo.CaloricallyPerfectGas(),
									    transport=transport.SutherlandTransport(beta=1.5),
										NDIMS=2)
	physics.set_physical_params()

	ns = physics.NUM_STATE_VARS

	P = 101325.
	rho = 1.1
	u = 2.5
	v = 3.5
	gamma = physics.thermo.gamma
	R = physics.thermo.R
	rhoE = P / (gamma - 1.) + 0.5 * rho * (u * u + v * v)

	Uq = np.zeros([1, 1, ns])

	srho, srhou, srhoE = physics.get_state_slices()

	Uq[:, :, srho] = rho
	Uq[:, :, srhou] = [rho * u, rho * v]
	Uq[:, :, srhoE] = rhoE

	# Calculate viscosity and thermal conductivity
	e = (Uq[..., srhoE] - 0.5*rho*(u**2+v**2))/Uq[..., srho]
	physics.thermo.set_state_from_rhoi_e(Uq[..., srho], e)
	mu = physics.transport.get_viscosity(physics.thermo)
	kappa = physics.transport.get_thermal_conductivity(physics.thermo)
	nu = mu / rho

	# Get temperature
	T = physics.compute_variable("Temperature", Uq, 
		flag_non_physical=True)[0, 0, 0]

	np.random.seed(10)
	drdx = np.random.rand()
	drdy = np.random.rand()
	dudx = np.random.rand()
	dudy = np.random.rand()
	dvdx = np.random.rand() 
	dvdy = np.random.rand()
	divu = dudx + dvdy
	dTdx = np.random.rand()
	dTdy = np.random.rand()

	tauxx = mu * (dudx + dudx - 2. / 3. * divu)
	tauxy = mu * (dudy + dvdx)
	tauyy = mu * (dvdy + dvdy - 2. / 3. * divu)

	gUq = np.zeros([1, 1, ns, 2])

	gUq[:, :, srho, 0] = drdx
	gUq[:, :, srhou, 0] = np.array([drdx * u + rho * dudx, drdx * v + rho * dvdx]).reshape((1,1,2))
	gUq[:, :, srhoE, 0] = (
		(R * T / (gamma - 1.) + 0.5 * (u * u + v * v)) * drdx
		+ rho * u * dudx
		+ rho * v * dvdx
		+ rho * R / (gamma - 1.) * dTdx
	)

	gUq[:, :, srho, 1] = drdy
	gUq[:, :, srhou, 1] = np.array([drdy * u + rho * dudy, drdy * v + rho * dvdy]).reshape((1,1,2))
	gUq[:, :, srhoE, 1] = (
		(R * T / (gamma - 1.) + 0.5 * (u * u + v * v)) * drdy
		+ rho * u * dudy
		+ rho * v * dvdy
		+ rho * R / (gamma - 1.) * dTdy
	)

	Fref = np.zeros([1, 1, ns, 2])
	Fref[:, :, srhou, 0] = np.array([tauxx, tauxy]).reshape((1,1,2))
	Fref[:, :, srhoE, 0] = tauxx * u + tauxy * v + kappa * dTdx

	Fref[:, :, srhou, 1] = np.array([tauxy, tauyy]).reshape((1,1,2))
	Fref[:, :, srhoE, 1] = tauxy * u + tauyy * v + kappa * dTdy

	F = physics.get_diff_flux_interior(Uq, gUq)

	kappa = kappa[0, 0, 0]
	np.testing.assert_allclose(F, Fref, kappa*rtol, kappa*atol)


def test_diffusion_flux_2D_zero_velocity():
	'''
	This tests the diffusive flux for a 2D case with zero vel
	'''
	physics = navierstokes.NavierStokes(thermo=thermo.CaloricallyPerfectGas(),
									    transport=transport.SutherlandTransport(),
										NDIMS=2)
	physics.set_physical_params()

	ns = physics.NUM_STATE_VARS

	P = 101325.
	rho = 1.1

	gamma = physics.thermo.gamma
	R = physics.thermo.R
	rhoE = P / (gamma - 1.)

	Uq = np.zeros([1, 1, ns])

	srho, srhou, srhoE = physics.get_state_slices()

	Uq[:, :, srho] = rho
	Uq[:, :, srhou] = [0., 0.]
	Uq[:, :, srhoE] = rhoE

	# Calculate viscosity
	e = Uq[..., srhoE] / Uq[..., srho] # kinetic energy is zero
	physics.thermo.set_state_from_rhoi_e(Uq[..., srho], e)
	mu = physics.transport.get_viscosity(physics.thermo)
	kappa = physics.transport.get_thermal_conductivity(physics.thermo)
	mu = mu[0,0,0]
	kappa = kappa[0,0,0]

	nu = mu / rho

	# Get temperature
	T = physics.compute_variable("Temperature", Uq, 
		flag_non_physical=True)[0, 0, 0]

	np.random.seed(10)
	drdx = np.random.rand()
	drdy = np.random.rand()

	dTdx = np.random.rand()
	dTdy = np.random.rand()

	gUq = np.zeros([1, 1, ns, 2])
	gUq[:, :, srho, 0] = drdx
	gUq[:, :, srhoE, 0] = (R * T / (gamma - 1.)) * drdx \
		+ rho * R / (gamma - 1.) * dTdx

	gUq[:, :, srho, 1] = drdy
	gUq[:, :, srhoE, 1] = (R * T / (gamma - 1.)) * drdy \
		+ rho * R / (gamma - 1.) * dTdy

	Fref = np.zeros([1, 1, ns, 2])
	Fref[:, :, srhoE, 0] = kappa * dTdx
	Fref[:, :, srhoE, 1] = kappa * dTdy

	F = physics.get_diff_flux_interior(Uq, gUq)
	np.testing.assert_allclose(F, Fref, kappa*rtol, kappa*atol)
