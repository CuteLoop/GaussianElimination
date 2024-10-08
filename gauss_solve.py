
import ctypes

gauss_library_path = './libgauss.so'

def unpack(A):
    """ Extract L and U parts from A, fill with 0's and 1's """
    n = len(A)
    L = [[A[i][j] for j in range(i)] + [1] + [0 for j in range(i+1, n)]
         for i in range(n)]

    U = [[0 for j in range(i)] + [A[i][j] for j in range(i, n)]
         for i in range(n)]

    return L, U

def lu_c(A):
    """ Accepts a list of lists A of floats and
    it returns (L, U) - the LU-decomposition as a tuple.
    """
    # Load the shared library
    lib = ctypes.CDLL(gauss_library_path)

    # Create a 2D array in Python and flatten it
    n = len(A)
    flat_array_2d = [item for row in A for item in row]

    # Convert to a ctypes array
    c_array_2d = (ctypes.c_double * len(flat_array_2d))(*flat_array_2d)

    # Define the function signature
    lib.lu_in_place.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double))

    # Modify the array in C (e.g., add 10 to each element)
    lib.lu_in_place(n, c_array_2d)

    # Convert back to a 2D Python list of lists
    modified_array_2d = [
        [c_array_2d[i * n + j] for j in range(n)]
        for i in range(n)
    ]

    # Extract L and U parts from A, fill with 0's and 1's
    return unpack(modified_array_2d)



def plu_c(A):
    """
    Accepts a list of lists A (matrix) of floats and returns (perm, L, U) - 
    the PLU-decomposition as a tuple.
    """
    # Load the shared library
    lib = ctypes.CDLL(gauss_library_path)

   
    # Size of the matrix (assuming it's square)
    n = len(A)
    
    # Flatten the 2D Python list (row-major order)
    flat_array_2d = [item for row in A for item in row]
    
    # Convert the flattened list to a ctypes array of double
    c_array_2d = (ctypes.c_double * len(flat_array_2d))(*flat_array_2d)
    
    # Create an array for the permutation vector (not matrix)
    perm = (ctypes.c_int * n)()  # Initializes a zeroed permutation array

    # Define the argument types for the C function
    lib.plu.argtypes = (
        ctypes.c_int,                         # size of the matrix
        ctypes.POINTER(ctypes.c_double),      # matrix A (as a 1D array)
        ctypes.POINTER(ctypes.c_int)          # permutation array (perm)
    )
    
    # Call the C function (in-place modification of the matrix and perm)
    lib.plu(n, c_array_2d, perm)
    
    # Convert the modified ctypes array back to a 2D Python list
    modified_array_2d = [
        [c_array_2d[i * n + j] for j in range(n)]
        for i in range(n)
    ]
    
    # Extract L and U from the modified array
    L, U = unpack(modified_array_2d)
    
    # Return the permutation vector, L, and U
    return list(perm), L, U


'''
def create_permutation_matrix(perm, n):
    """
    Converts a permutation vector into a permutation matrix.
    """
    P = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        P[i][perm[i]] = 1.0  # Permutation matrix has 1's in permuted positions
    return P

def unpack_plu(A):
    """
    Extracts the lower (L) and upper (U) triangular matrices from 
    the modified matrix A after PLU decomposition.
    """
    n = len(A)
    L = [[0.0 if i != j else 1.0 for j in range(n)] for i in range(n)]  # Identity for L
    U = [[0.0 for _ in range(n)] for _ in range(n)]

    # Populate L and U from the matrix A
    for i in range(n):
        for j in range(n):
            if i > j:
                L[i][j] = A[i][j]  # Lower triangular part
            else:
                U[i][j] = A[i][j]  # Upper triangular part

    return L, U
'''

def lu_python(A):
    n = len(A)
    for k in range(n):
        for i in range(k,n):
            for j in range(k):
                A[k][i] -= A[k][j] * A[j][i]
        for i in range(k+1, n):
            for j in range(k):
                A[i][k] -= A[i][j] * A[j][k]
            A[i][k] /= A[k][k]

    return unpack(A)



def plu_python(A):
    """
    Perform PLU decomposition of matrix A using Python.
    Returns the permutation vector, L, and U.
    """
    n = len(A)
    perm = list(range(n))  # Initialize permutation vector

    for k in range(n):
        # Find the pivot row
        pivot = k
        for i in range(k+1, n):
            if abs(A[i][k]) > abs(A[pivot][k]):
                pivot = i

        # Swap rows in A and update the permutation vector
        if pivot != k:  # Ensure we only swap when needed
            #print(f"Swapping rows: {k} and {pivot}")

            A[k], A[pivot] = A[pivot], A[k]
            perm[k], perm[pivot] = perm[pivot], perm[k]

        # Perform the decomposition
        for i in range(k+1, n):
            A[i][k] /= A[k][k]  # Compute the L value (modify A in place)
            for j in range(k+1, n):
                #print(f"Index values: i={i}, j={j}")

                A[i][j] -= A[i][k] * A[k][j]

    # Return permutation vector and the decomposed L and U matrices
    return perm, *unpack(A)






def lu(A, use_c=False):
    if use_c:
        return lu_c(A)
    else:
        return lu_python(A)

def plu(A, use_c=False):
    if use_c:
        return plu_c(A)
    else:
        return plu_python(A)




if __name__ == "__main__":

    def print_matrix(name, matrix):
        print(f"{name}:")
        for row in matrix:
            print("  ", " ".join(f"{val:.4f}" for val in row))
        print()

    def print_vector(name, vector):
        print(f"{name}: {vector}")
        print()

    def get_A():
        """ Make a test matrix """
        A = [[2.0, 3.0, -1.0],
             [4.0, 1.0, 2.0],
             [-2.0, 7.0, 2.0]]
        return A

    def hilbert_matrix(n):
        """ Returns the nxn Hilbert matrix. """
        return [[1.0 / (i + j + 1) for j in range(n)] for i in range(n)]

    # Test Python LU decomposition
    A = get_A()
    L, U = lu(A, use_c=False)
    print_matrix("Python LU L matrix", L)
    print_matrix("Python LU U matrix", U)

    # Must re-initialize A as it was destroyed by the previous operation
    A = get_A()

    # Test C LU decomposition
    L, U = lu(A, use_c=True)
    print_matrix("C LU L matrix", L)
    print_matrix("C LU U matrix", U)

    # Test Python PLU decomposition
    A = get_A()
    P, L, U = plu(A, use_c=False)
    print_vector("Python PLU Permutation vector", P)
    print_matrix("Python PLU L matrix", L)
    print_matrix("Python PLU U matrix", U)

    # Test C PLU decomposition
    A = get_A()  # Reset the matrix A
    P, L, U = plu(A, use_c=True)
    print_vector("C PLU Permutation vector", P)
    print_matrix("C PLU L matrix", L)
    print_matrix("C PLU U matrix", U)

    # Test Python PLU decomposition on a 20x20 Hilbert matrix
    A = hilbert_matrix(20)
    P, L, U = plu(A, use_c=False)
    print_vector("Python PLU Permutation vector (20x20 Hilbert)", P)
    print_matrix("Python PLU L matrix (20x20 Hilbert)", L)
    print_matrix("Python PLU U matrix (20x20 Hilbert)", U)

    # Test C PLU decomposition on a 20x20 Hilbert matrix
    A = hilbert_matrix(20)  # Reset the Hilbert matrix
    P, L, U = plu(A, use_c=True)
    print_vector("C PLU Permutation vector (20x20 Hilbert)", P)
    print_matrix("C PLU L matrix (20x20 Hilbert)", L)
    print_matrix("C PLU U matrix (20x20 Hilbert)", U)
