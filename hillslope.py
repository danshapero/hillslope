import numpy as np
import firedrake
from firedrake import inner, grad, dx, \
        sqrt, exp, ln, sin, cos, tanh, \
        assemble, replace, action, derivative
import matplotlib.pyplot as plt

k = 0.0035  # m^2/yr
γ = 1.25    # dimensionless
U = 2.5e-4    # m/yr

def steady_state(u, bc=None, armijo=1e-4, tolerance=1e-6, max_iterations=50):
    Q = u.ufl_function_space()
    if bc is None:
        bc = firedrake.DirichletBC(Q, 0, "on_boundary")

    z = firedrake.Function(Q)
    E_diffusive = -k * γ**2 / 2 * ln(1 - inner(grad(z), grad(z))/γ**2) * dx
    E_uplift = u * z * dx
    E = E_diffusive - E_uplift
    F = derivative(E, z)
    H = derivative(F, z)

    p = firedrake.Function(Q)
    for iteration in range(max_iterations):
        firedrake.solve(H == -F, p, bc,
                solver_parameters={'ksp_type': 'preonly',
                                   'pc_type': 'lu'})

        dE_dp = assemble(action(F, p))
        α = 1.0
        E0 = assemble(E)
        Ez = assemble(replace(E, {z: z + α * p}))
        while (Ez > E0 + armijo * α * dE_dp) or np.isnan(Ez):
            α /= 2
            Ez = assemble(replace(E, {z: z + α * p}))

        z.assign(z + α * p)

        scale = assemble(E_diffusive)
        if abs(dE_dp) < tolerance * scale:
            print("{0}: {1}, {2}".format(iteration, scale, dE_dp))
            return z

    raise ValueError("Newton solver failed to converge after {0} iterations"
                     .format(max_iterations))
