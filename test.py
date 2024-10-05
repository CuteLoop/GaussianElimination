import unittest
import random
from helpers import print_matrix, hilbert_matrix, random_matrix
from gauss_solve import lu, plu

class TestLUDecomposition(unittest.TestCase):

    def test_lu_3x3_python(self):
        """ Test LU decomposition for a simple 3x3 matrix using Python implementation. """
        A = [[2.0, 3.0, -1.0],
             [4.0, 1.0, 2.0],
             [-2.0, 7.0, 2.0]]

        L, U = lu(A, use_c=False)  # Python implementation
        print("Python: LU decomposition for 3x3 matrix")
        print("L matrix:")
        print_matrix(L)
        print("U matrix:")
        print_matrix(U)

    def test_lu_3x3_c(self):
        """ Test LU decomposition for a simple 3x3 matrix using C implementation. """
        A = [[2.0, 3.0, -1.0],
             [4.0, 1.0, 2.0],
             [-2.0, 7.0, 2.0]]

        L, U = lu(A, use_c=True)  # C implementation
        print("C: LU decomposition for 3x3 matrix")
        print("L matrix:")
        print_matrix(L)
        print("U matrix:")
        print_matrix(U)

    def test_lu_hilbert_python(self):
        """ Test LU decomposition for a 20x20 Hilbert matrix using Python implementation. """
        A = hilbert_matrix(20)

        L, U = lu(A, use_c=False)  # Python implementation
        print("Python: LU decomposition for 20x20 Hilbert matrix")
        print("L matrix:")
        print_matrix(L)
        print("U matrix:")
        print_matrix(U)

    def test_lu_hilbert_c(self):
        """ Test LU decomposition for a 20x20 Hilbert matrix using C implementation. """
        A = hilbert_matrix(20)

        L, U = lu(A, use_c=True)  # C implementation
        print("C: LU decomposition for 20x20 Hilbert matrix")
        print("L matrix:")
        print_matrix(L)
        print("U matrix:")
        print_matrix(U)

    def test_lu_singular_python(self):
        """ Test LU decomposition for a singular matrix using Python implementation. """
        A = [[1.0, 2.0, 3.0],
             [0.0, 0.0, 0.0],  # Singular matrix with zero row
             [4.0, 5.0, 6.0]]

        try:
            L, U = lu(A, use_c=False)  # Python implementation
            print("Python: LU decomposition for singular matrix")
            print("L matrix:")
            print_matrix(L)
            print("U matrix:")
            print_matrix(U)
        except Exception as e:
            print(f"Python: LU decomposition failed for singular matrix: {e}")

    def test_lu_singular_c(self):
        """ Test LU decomposition for a singular matrix using C implementation. """
        A = [[1.0, 2.0, 3.0],
             [0.0, 0.0, 0.0],  # Singular matrix with zero row
             [4.0, 5.0, 6.0]]

        try:
            L, U = lu(A, use_c=True)  # C implementation
            print("C: LU decomposition for singular matrix")
            print("L matrix:")
            print_matrix(L)
            print("U matrix:")
            print_matrix(U)
        except Exception as e:
            print(f"C: LU decomposition failed for singular matrix: {e}")

    def test_lu_random_python(self):
        """ Test LU decomposition for a 10x10 random dense matrix using Python implementation. """
        A = random_matrix(10)

        L, U = lu(A, use_c=False)  # Python implementation
        print("Python: LU decomposition for 10x10 random dense matrix")
        print("L matrix:")
        print_matrix(L)
        print("U matrix:")
        print_matrix(U)

    def test_lu_random_c(self):
        """ Test LU decomposition for a 10x10 random dense matrix using C implementation. """
        A = random_matrix(10)

        L, U = lu(A, use_c=True)  # C implementation
        print("C: LU decomposition for 10x10 random dense matrix")
        print("L matrix:")
        print_matrix(L)
        print("U matrix:")
        print_matrix(U)

class TestPLUDecomposition(unittest.TestCase):

    def test_plu_3x3_python(self):
        """ Test PLU decomposition for a simple 3x3 matrix using Python implementation. """
        A = [[2.0, 3.0, -1.0],
             [4.0, 1.0, 2.0],
             [-2.0, 7.0, 2.0]]

        P, L, U = plu(A, use_c=False)  # Python implementation
        print("Python: PLU decomposition for 3x3 matrix")
        print("P matrix:")
        print_matrix(P)
        print("L matrix:")
        print_matrix(L)
        print("U matrix:")
        print_matrix(U)

    def test_plu_3x3_c(self):
        """ Test PLU decomposition for a simple 3x3 matrix using C implementation. """
        A = [[2.0, 3.0, -1.0],
             [4.0, 1.0, 2.0],
             [-2.0, 7.0, 2.0]]

        P, L, U = plu(A, use_c=True)  # C implementation
        print("C: PLU decomposition for 3x3 matrix")
        print("P matrix:")
        print_matrix(P)
        print("L matrix:")
        print_matrix(L)
        print("U matrix:")
        print_matrix(U)

    def test_plu_hilbert_python(self):
        """ Test PLU decomposition for a 20x20 Hilbert matrix using Python implementation. """
        A = hilbert_matrix(20)

        P, L, U = plu(A, use_c=False)  # Python implementation
        print("Python: PLU decomposition for 20x20 Hilbert matrix")
        print("P matrix:")
        print_matrix(P)
        print("L matrix:")
        print_matrix(L)
        print("U matrix:")
        print_matrix(U)

    def test_plu_hilbert_c(self):
        """ Test PLU decomposition for a 20x20 Hilbert matrix using C implementation. """
        A = hilbert_matrix(20)

        P, L, U = plu(A, use_c=True)  # C implementation
        print("C: PLU decomposition for 20x20 Hilbert matrix")
        print("P matrix:")
        print_matrix(P)
        print("L matrix:")
        print_matrix(L)
        print("U matrix:")
        print_matrix(U)




if __name__ == '__main__':
    unittest.main()
