import numpy as np

from sdp_formulations_cvxpy import (
    solve_moment_image,
    solve_moment_kernel,
    solve_sos_image,
    solve_sos_kernel,
)


def hvec(M):
    """Half-vectorization of a symmetric matrix M."""
    n = M.shape[0]
    idx = np.tril_indices(n)
    factors = np.ones_like(idx[0], dtype=float)
    factors[idx[0] != idx[1]] = np.sqrt(2.0)
    return M[idx] * factors


def hmat(v):
    """Inverse half-vectorization to get a symmetric matrix from vector v."""
    # given half-vectorization vector v, what is size n?
    n = int((np.sqrt(8 * len(v) + 1) - 1) / 2)
    M = np.zeros((n, n))
    idx = np.tril_indices(n)
    diag = np.diag_indices(n)
    M[idx] = v / np.sqrt(2.0)
    M = M + M.T
    M[diag] /= np.sqrt(2.0)
    return M


phi = lambda x: np.array([1, x, x**2, x**3])

# 1. Define (span) basis, manually
B1 = phi(1.0)[:, None] @ phi(1)[None, :]
B2 = phi(-1.0)[:, None] @ phi(-1)[None, :]

np.testing.assert_allclose(hmat(hvec(B1)), B1)
np.testing.assert_allclose(hvec(B1).T @ hvec(B1), np.trace(B1 @ B1))

# 2. Define span and nullspace bases numerically.

# a. sample feasible points
x_samples = [-1.0, 1.0]

# b. construct moment matrix.
X_samples = [phi(x).reshape(-1, 1) @ phi(x).reshape(1, -1) for x in x_samples]

# Step 3: Stack the vectorized matrices into a single matrix L
L = np.array([hvec(Xi) for Xi in X_samples]).T

print(f"\nShape of the measurement matrix L: {L.shape}")

# Step 4: Use SVD to find the orthonormal bases
# U will contain the orthonormal basis for the column space (range)
# S will contain the singular values
# Vt is the conjugate transpose of V
U, S, Vt = np.linalg.svd(L, full_matrices=True)

# The basis for V is the set of columns in U corresponding to non-zero singular values
tol = 1e-8
rank = np.sum(S > tol)

b_vectors = U[:, :rank]

# The basis for V_perp is the set of columns in U corresponding to zero singular values
u_vectors = U[:, rank:]

print(f"Dimension of ambient space: {L.shape[0]}")
print(f"Dimension of V (span): {b_vectors.shape[1]}")
print(f"Dimension of VT (nullspace): {u_vectors.shape[1]}")

# Reshape the basis vectors back into matrices
B_basis = [hmat(v) for v in b_vectors.T]
U_basis = [hmat(v) for v in u_vectors.T]

# 3. create functions for the four different forms.

# We will solve a simple problem: min p(x) = x, subject to x in {-1, 1}.
# The true minimum is -1.

# First, we need the cost matrix C such that <C, phi(x)phi(x)^T> = x
# <C, M> = tr(C^T M).
# M_12 = x. So we can choose C to have 1 at position (1,0).
# For a symmetric C, we can use C = 0.5 * (e_0 e_1^T + e_1 e_0^T)
C = np.zeros((4, 4))
C[0, 1] = 0.5
C[1, 0] = 0.5

print("Cost of feasible point x=1:", np.trace(C @ B1), phi(1).T @ C @ phi(1))
print("Cost of feasible point x=-1:", np.trace(C @ B2), phi(-1).T @ C @ phi(-1))

# Normalization matrix
A0 = np.zeros((4, 4))
A0[0, 0] = 1.0


def print_sol(title, info):
    x = info["X"][0, 1]  # pick the solution value
    print(f"\n{title}")
    print(f"  optimal value: {info['c']:.4f}")
    print(f"  x: {x.round(3)}")
    print(f"  time: {info['time']*1000:.0f}ms")
    return
    print(f"  H:\n{info['H'].round(3)}")
    if info.get("alpha", None) is not None:
        print(f"  alpha: {np.array(info['alpha']).round(3)}")
    if info.get("beta", None) is not None:
        print(f"  beta: {np.array(info['beta']).round(3)}")


info_sos_image = solve_sos_image(C, A0, U_basis)
print_sol("sos image solution:", info_sos_image)
info_sos_kernel = solve_sos_kernel(C, A0, B_basis)
print_sol("sos kernel solution:", info_sos_kernel)
info_moment_image = solve_moment_image(C, A0, B_basis)
print_sol("moment image solution:", info_moment_image)
info_moment_kernel = solve_moment_kernel(C, A0, U_basis)
print_sol("moment kernel solution:", info_moment_kernel)

print("done")
