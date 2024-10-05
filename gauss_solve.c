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



void plu(int n, double *A, int *perm) {
    int i, j, k;
    //printf("Starting PLU decomposition for matrix of size %d\n", n);

    // Initialize the permutation array
    for (i = 0; i < n; i++) {
        perm[i] = i;  // Initially, perm[i] = i for identity
    }

    //printf("Permutation array initialized\n");

    // PLU Decomposition logic
    for (i = 0; i < n; i++) {
        // Find the pivot (largest element in the current column)
        int pivot = i;
        for (j = i + 1; j < n; j++) {
            //printf("Comparing A[%d][%d] with A[%d][%d]\n", j, i, pivot, i);
            if (fabs(A[j * n + i]) > fabs(A[pivot * n + i])) {
                pivot = j;
            }
        }

        //printf("Pivot found at row %d for column %d\n", pivot, i);

        // Swap rows if necessary
        if (pivot != i) {
            //printf("Swapping rows %d and %d\n", i, pivot);
            for (k = 0; k < n; k++) {
                double temp = A[i * n + k];
                A[i * n + k] = A[pivot * n + k];
                A[pivot * n + k] = temp;
            }

            // Swap corresponding entries in permutation array
            int temp_perm = perm[i];
            perm[i] = perm[pivot];
            perm[pivot] = temp_perm;
        }

        //printf("Row swapping complete\n");

        // Check for a singular matrix (zero pivot element)
        if (fabs(A[i * n + i]) < 1e-9) {  // Adjust tolerance as needed
            printf("Matrix is singular or nearly singular\n");
            return;
        }

        // Perform Gaussian elimination on rows below pivot
        for (j = i + 1; j < n; j++) {
            //printf("Eliminating row %d below pivot row %d\n", j, i);
            double factor = A[j * n + i] / A[i * n + i];  // Compute the L factor
            A[j * n + i] = factor;  // Store L factor
            for (k = i + 1; k < n; k++) {
                A[j * n + k] -= factor * A[i * n + k];  // Update U part
            }
        }
    }

    //printf("PLU decomposition complete\n");
}
