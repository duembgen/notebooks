"""
Simple CVXPY formulations for the SOS/Moment, Kernel/Image SDP problems

This is not the most viable when it comes to calculating solve times.
"""

import time

import cvxpy as cp

SOLVER = cp.SCS


def start_buffer():
    start_time = time.time()
    return start_time, None


def end_buffer(start_time, not_used):
    ttot = time.time()
    return ttot - start_time


def solve_sos_image(C, A_0, U_basis):
    c = cp.Variable()
    beta = cp.Variable(len(U_basis))
    objective = cp.Maximize(c)
    constraints = [
        C - c * A_0 + cp.sum([beta[i] * Ui for i, Ui in enumerate(U_basis)]) >> 0
    ]
    problem = cp.Problem(objective, constraints)
    a, b = start_buffer()
    problem.solve(solver=SOLVER, verbose=True)
    ttot = end_buffer(a, b)

    assert problem.status == "optimal"
    H = C - c.value * A_0 + cp.sum([beta[i].value * Ui for i, Ui in enumerate(U_basis)])
    info = {
        "X": constraints[0].dual_value,
        "H": H,
        "c": c.value,
        "beta": beta.value,
        "time": ttot,
    }
    return info


def solve_sos_kernel(C, A_0, B_basis):
    H = cp.Variable((4, 4), symmetric=True)
    c = cp.Variable()
    objective = cp.Maximize(c)
    constraints = [H >> 0]
    constraints += [cp.trace(Bi @ (-C + c * A_0 + H)) == 0 for Bi in B_basis]
    problem = cp.Problem(objective, constraints)  # type: ignore
    a, b = start_buffer()
    problem.solve(solver=SOLVER)
    ttot = end_buffer(a, b)
    assert problem.status == "optimal"
    c = problem.value
    alpha = [c.dual_value for c in constraints[1:]]
    info = {
        "X": constraints[0].dual_value,
        "H": H.value,
        "c": c,
        "alpha": alpha,
        "time": ttot,
    }
    return info


def solve_moment_image(C, A_0, B_basis):
    alpha = cp.Variable(len(B_basis))
    objective = cp.Minimize(
        cp.sum([alpha[i] * cp.trace(C @ Bi) for i, Bi in enumerate(B_basis)])
    )
    constraints = [
        cp.sum([alpha[i] * Bi for i, Bi in enumerate(B_basis)]) >> 0,
        cp.sum([alpha[i] * cp.trace(A_0 @ Bi) for i, Bi in enumerate(B_basis)]) == 1,
    ]
    problem = cp.Problem(objective, constraints)
    t1 = time.time()
    a, b = start_buffer()
    problem.solve(solver=SOLVER)
    ttot = end_buffer(a, b)
    assert problem.status == "optimal"
    H = constraints[0].dual_value
    c = problem.value
    X = cp.sum([alpha[i] * Bi for i, Bi in enumerate(B_basis)])
    info = {"X": X.value, "H": H, "c": c, "alpha": alpha.value, "time": ttot}  # type: ignore
    return info


def solve_moment_kernel(C, A_0, U_basis):
    X = cp.Variable((4, 4), symmetric=True)
    objective = cp.Minimize(cp.trace(C @ X))
    constraints = [X >> 0, cp.trace(A_0 @ X) == 1]
    constraints += [cp.trace(Ui @ X) == 0 for Ui in U_basis]
    problem = cp.Problem(objective, constraints)
    a, b = start_buffer()
    problem.solve(solver=SOLVER)
    ttot = end_buffer(a, b)

    assert problem.status == "optimal"
    beta = [c.dual_value for c in constraints[2:]]
    info = {
        "X": X.value,
        "H": constraints[0].dual_value,
        "c": problem.value,
        "beta": beta,
        "time": ttot,
    }
    return info
