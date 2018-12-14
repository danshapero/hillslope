import numpy as np
import firedrake
from firedrake import inner, grad, dx, ln, \
        assemble, replace, action, derivative

k = 0.0035  # m^2/yr
S = 1.25    # dimensionless
U = 2.5e-4  # m/yr


def _newton_solve(z, E, scale, tolerance=1e-6, armijo=1e-4, max_iterations=50):
    F = derivative(E, z)
    H = derivative(F, z)

    Q = z.function_space()
    bc = firedrake.DirichletBC(Q, 0, 'on_boundary')
    p = firedrake.Function(Q)
    for iteration in range(max_iterations):
        firedrake.solve(H == -F, p, bc,
            solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

        dE_dp = assemble(action(F, p))
        α = 1.0
        E0 = assemble(E)
        Ez = assemble(replace(E, {z: z + firedrake.Constant(α) * p}))
        while (Ez > E0 + armijo * α * dE_dp) or np.isnan(Ez):
            α /= 2
            Ez = assemble(replace(E, {z: z + firedrake.Constant(α) * p}))

        z.assign(z + firedrake.Constant(α) * p)
        if abs(dE_dp) < tolerance * assemble(scale):
            return z

    raise ValueError("Newton solver failed to converge after {0} iterations"
                     .format(max_iterations))

def solve(dt, z0, u, **kwargs):
    z = z0.copy(deepcopy=True)
    J_diffusive = -k * S**2 / 2 * ln(1 - inner(grad(z), grad(z)) / S**2) * dx
    J_uplift = u * z * dx
    J = J_diffusive - J_uplift

    E = 0.5 * (z - z0)**2 * dx + firedrake.Constant(dt) * J
    scale = 0.5 * z**2 * dx + firedrake.Constant(dt) * J_diffusive
    return _newton_solve(z, E, scale, **kwargs)


def steady_state(u, **kwargs):
    z = firedrake.Function(u.function_space())
    J_diffusive = -k * S**2 / 2 * ln(1 - inner(grad(z), grad(z)) / S**2) * dx
    J_uplift = u * z * dx
    J = J_diffusive - J_uplift

    return _newton_solve(z, J, J_diffusive, **kwargs)
