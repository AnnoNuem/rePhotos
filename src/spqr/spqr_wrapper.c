// C wrapper to SparseSuiteQR library et al. for Python

// We pass in the sparse matrix data in a COO sparse matrix format. Cholmod
// refers to this as a "cholmod_triplet" format. This is then converted to its
// "cholmod_sparse" format, which is a CSC matrix.

#include <stdio.h>
#include <stdlib.h>
#include "SuiteSparseQR_C.h"

size_t qr_solve(double const *A_data, long const *A_row, long const *A_col, size_t A_nnz, size_t A_m, size_t A_n, double const *b_data, double *C_data, long *C_i, long *C_j, double *r_data, long *r_i, long *r_j) {
    // Solves the matrix equation Ax=b where A is a sparse matrix and x and b
    // are dense column vectors. A and b are inputs, x is solved for in the
    // least squares sense using a rank-revealing QR factorization.
    //
    // Inputs
    //
    // A_data, A_row, A_col: the COO data
    // A_nnz: number of non-zero entries, ie the length of the arrays A_data, etc
    // A_m: number of rows in A
    // A_n: number of cols in A
    // b_data: the data in b. It is A_m entries long.
    //
    // Outputs
    //
    // x_data: the data in x. It is A_n entries long
    //
    // MAKE SURE x_data is allocated to the right size before calling this function
    //
    cholmod_common Common, *cc;
    cholmod_sparse *A_csc;
    cholmod_triplet *A_coo;
    cholmod_dense *b;
    size_t k;
    // Helper pointers
    long *Ai, *Aj;
    double *Ax, *bx;

    /* start CHOLMOD */
    cc = &Common ;
    cholmod_l_start (cc) ;

    // Create A, first as a COO matrix, then convert to CSC
    A_coo = cholmod_l_allocate_triplet(A_m, A_n, A_nnz, 0, CHOLMOD_REAL, cc);
    if (A_coo == NULL) {
        fprintf(stderr, "ERROR: cannot allocate triplet");
        return;
    }
    // Copy in data
    Ai = A_coo->i;
    Aj = A_coo->j;
    Ax = A_coo->x;
    for (k=0; k<A_nnz; k++) {
        Ai[k] = A_row[k];
        Aj[k] = A_col[k];
        Ax[k] = A_data[k];
    }
    A_coo->nnz = A_nnz;
    // Make sure the matrix is valid
    if (cholmod_l_check_triplet(A_coo, cc) != 1) {
        fprintf(stderr, "ERROR: triplet matrix is not valid");
        return;
    }
    // Convert to CSC
    A_csc = cholmod_l_triplet_to_sparse(A_coo, A_nnz, cc);

    // Create b as a dense matrix
    b = cholmod_l_allocate_dense(A_m, 1, A_m, CHOLMOD_REAL, cc);
    bx = b->x;
    for (k=0; k<A_m; k++) {
        bx[k] = b_data[k];
    }
    // Make sure the matrix is valid
    if (cholmod_l_check_dense(b, cc) != 1) {
        fprintf(stderr, "ERROR: b vector is not valid");
        return;
    }

    // Solve for x
    //x = SuiteSparseQR_C_backslash_default(A_csc, b, cc);
    

    // OWN STUFF
    int rank;
    long int *Qfill;
    cholmod_sparse *R, *Csparse;
    int econ = A_csc->nrow;
    double tol = SPQR_DEFAULT_TOL;
    int ordering = SPQR_ORDERING_FIXED;//VERY IMPORTANT TO GET SAME RESULTS AS MATLAB
    rank = SuiteSparseQR_C (ordering, tol, econ, 0, A_csc,
                    NULL, b, &Csparse, NULL, &R, &Qfill, NULL, NULL, NULL, cc) ;

    
    // Return values
    // C
    cholmod_triplet *C_coo = cholmod_l_allocate_triplet(Csparse->nrow, Csparse->ncol, A_m, 0, CHOLMOD_REAL, cc);
    C_coo = cholmod_l_sparse_to_triplet(Csparse, cc);
    if (C_coo == NULL) {
        fprintf(stderr, "ERROR: cannot allocate triplet");
        return;
    }

    double *Ccoo_x = C_coo->x;
    long *Ccoo_i = C_coo->i;
    long *Ccoo_j = C_coo->j;
    for (k=0; k < A_m; k++) {
        C_data[k] = Ccoo_x[k];
        C_i[k] = Ccoo_i[k];
        C_j[k] = Ccoo_j[k];
    }
    
    // r
    cholmod_triplet *R_coo = cholmod_l_allocate_triplet(R->nrow, R->ncol, A_n, 0, CHOLMOD_REAL, cc);
    R_coo = cholmod_l_sparse_to_triplet(R, cc);
    if (R_coo == NULL) {
        fprintf(stderr, "ERROR: cannot allocate triplet");
        return;
    }

    double *R_x = R_coo->x;
    long *R_i = R_coo->i;
    long *R_j = R_coo->j;
    size_t R_nnz = R_coo->nnz;
    for (k=0; k < R_nnz; k++) {
        r_data[k] = R_x[k];
        r_i[k] = R_i[k];
        r_j[k] = R_j[k];
    }



    //  FREE OWN STUFF
    cholmod_l_free_sparse(&R, cc);
    cholmod_l_free_sparse(&Csparse, cc);
    cholmod_l_free_triplet(&R_coo, cc);
    cholmod_l_free_triplet(&C_coo, cc);


    // Return values of x
    //xx = x->x;
    //for (k=0; k<A_n; k++) {
    //    x_data[k] = xx[k];
    //}
    
    /* free everything and finish CHOLMOD */
    cholmod_l_free_triplet(&A_coo, cc);
    cholmod_l_free_sparse(&A_csc, cc);
    cholmod_l_free_dense(&b, cc);
    cholmod_l_finish(cc);
    return R_nnz;
}

