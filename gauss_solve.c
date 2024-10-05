/*----------------------------------------------------------------
* File:     gauss_solve.c
*----------------------------------------------------------------
*
* Author:   Marek Rychlik (rychlik@arizona.edu)
* Date:     Sun Sep 22 15:40:29 2024
* Copying:  (C) Marek Rychlik, 2020. All rights reserved.
*
*----------------------------------------------------------------*/
#include "gauss_solve.h"
#include <math.h>  // For fabs
#include <stdio.h> // For printf

void gauss_solve_in_place(const int n, double A[n][n], double b[n])
{
  for(int k = 0; k < n; ++k) {
    for(int i = k+1; i < n; ++i) {
      /* Store the multiplier into A[i][k] as it would become 0 and be
	 useless */
      A[i][k] /= A[k][k];
      for( int j = k+1; j < n; ++j) {
	A[i][j] -= A[i][k] * A[k][j];
      }
      b[i] -= A[i][k] * b[k];
    }
  } /* End of Gaussian elimination, start back-substitution. */
  for(int i = n-1; i >= 0; --i) {
    for(int j = i+1; j<n; ++j) {
      b[i] -= A[i][j] * b[j];
    }
    b[i] /= A[i][i];
  } /* End of back-substitution. */
}

void lu_in_place(const int n, double A[n][n])
{
  for(int k = 0; k < n; ++k) {
    for(int i = k; i < n; ++i) {
      for(int j=0; j<k; ++j) {
	/* U[k][i] -= L[k][j] * U[j][i] */
	A[k][i] -=  A[k][j] * A[j][i]; 
      }
    }
    for(int i = k+1; i<n; ++i) {
      for(int j=0; j<k; ++j) {
	/* L[i][k] -= A[i][k] * U[j][k] */
	A[i][k] -= A[i][j]*A[j][k]; 
      }
      /* L[k][k] /= U[k][k] */
      A[i][k] /= A[k][k];	
    }
  }
}

void lu_in_place_reconstruct(int n, double A[n][n])
{
  for(int k = n-1; k >= 0; --k) {
    for(int i = k+1; i<n; ++i) {
      A[i][k] *= A[k][k];
      for(int j=0; j<k; ++j) {
	A[i][k] += A[i][j]*A[j][k];
      }
    }
    for(int i = k; i < n; ++i) {
      for(int j=0; j<k; ++j) {
	A[k][i] +=  A[k][j] * A[j][i];
      }
    }
  }
}




void plu(int n, double A[n][n], int P[n]) {
  // Initialize the permutation array
  for(int i = 0; i < n; ++i) {
    P[i] = i;
  }

  // PLU Decomposition with Partial Pivoting
  for(int k = 0; k < n; ++k) {
    // Find the pivot (maximum absolute value in the column)
    double max = 0;
    int imax = k;
    for(int i = k; i < n; ++i) {
      if(fabs(A[i][k]) > max) {  // Fix: Use fabs to find the largest absolute value
        max = fabs(A[i][k]);
        imax = i;
      }
    }

    // Swap rows in the permutation vector and in matrix A
    SWAP(P[k], P[imax], int);
    for(int j = 0; j < n; ++j) {
      SWAP(A[k][j], A[imax][j], double);
    }

    // Check for singularity (zero pivot)
    if(fabs(A[k][k]) < 1e-10) {
      printf("Matrix is singular.\n");
      return;
    }

    // Perform the elimination below the pivot (for rows below the pivot)
    for(int i = k + 1; i < n; ++i) {
      // Compute the multiplier and store it in the lower triangular part of A (matrix L)
      A[i][k] /= A[k][k];

      // Update the remaining elements in the row
      for(int j = k + 1; j < n; ++j) {
        A[i][j] -= A[i][k] * A[k][j];
      }
    }
  }
}
