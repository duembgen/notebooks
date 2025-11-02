import numpy as np


def svec(A):
    if not np.allclose(A, A.T):
        raise ValueError("Input matrix must be symmetric.")

    n = A.shape[0]
    svec_A = np.zeros(n * (n + 1) // 2)
    k = 0
    for j in range(n):
        for i in range(j, n):
            if i == j:
                svec_A[k] = A[i, j]
            else:
                svec_A[k] = A[i, j] * np.sqrt(2)
            k += 1
    return svec_A


def smat(v):
    n = int((-1 + np.sqrt(1 + 8 * len(v))) / 2)
    A = np.zeros((n, n))
    k = 0
    for j in range(n):
        for i in range(j, n):
            if i == j:
                A[i, j] = v[k]
            else:
                A[i, j] = v[k] / np.sqrt(2)
                A[j, i] = A[i, j]
            k += 1
    return A


if __name__ == "__main__":
    A = np.array([[1, 2], [2, 3]])
    sv = svec(A)
    A_reconstructed = smat(sv)
    assert np.allclose(
        A, A_reconstructed
    ), "svec and smat functions are not consistent."
    print("all tests passed")
