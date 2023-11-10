// Installation: Install openblas-dev in linux, Accelerate framework in Mac to get BLAS and LAPACK functions
// Compilation: g++ -O2 cholesky.cpp -o cholesky -fopenmp -lblas -llapack
// Execution: OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ./cholesky
// For more info regarding BLAS/LAPACK functions, c.f. Intel MKL documentation
// Pour plus d'information concernant les fonctions dans BLAS/LAPACK, c.f. documentation d'Intel MKL
//   https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-fortran/2023-0
// Information on blocked Cholesky algorithm
// Information sur l'algorithme de Cholesky par blocs
//   https://www.netlib.org/utk/papers/factor/node9.html
// Complete list of BLAS routines
// Liste complete des routines BLAS
//   https://www.netlib.org/blas/blasqr.pdf

#include <iomanip>
#include <iostream>
#include <vector>

void printMatrix(const std::vector<double>& mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          std::cout << std::setw(12) << mat[j * rows + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// 2-norm of a vector (BLAS1)
// 2-norme pour un vecteur (BLAS1)
extern "C" double dnrm2_(
  int *n,
  double *x,
  int *incx);

// Compute y = y + a x, where x and y are vectors and a is a scalar (BLAS1)
// Calculer y = y + a x, x et y etant des vecteurs et a etant un scalaire (BLAS1)
extern "C" void daxpy_(
  int *n,
  double *a,
  double *x,
  int *incx,
  double *y,
  int *incy);

// Matrix-vector multiplication where alpha and beta are scalars (BLAS2)
// Produit matrice-vecteur, alpha et beta etant des scalaires (BLAS2)
// y = alpha A x + beta y
extern "C" void dgemv_(
  char *trans,
  int *m,
  int *n,
  double *alpha,
  double *a,
  int *lda,
  double *x,
  int *incx,
  double *beta,
  double *y,
  int *incy);

// Lower/upper triangular system solve for a single column vector. x is modified in-place (overwritten by x_new)
// Resoudre un systeme triangulaire superieure/inferieure pour un vecteur de colonne. x est modifie en place (surecrit
// par x_new)
// L x_new = x, U x_new = x (BLAS2)
extern "C" void dtrsv_(
  char *uplo,
  char *trans,
  char *diag,
  int *n,
  double *A,
  int *lda,
  double *x,
  int *incx);

// Matrix-matrix multiplication, alpha and beta are scalars and op(.) is an optional matrix transposition operator.
// Produit matrice-vecteur, alpha et beta sont des scalaires et op(.) est une transposition matricielle optionnelle.
// C = alpha opA(A) op(B) + beta op(C) (BLAS3)
extern "C" void dgemm_(
  char *transA,
  char *transB,
  int *m,
  int *n,
  int *k,
  double *alpha,
  double *A,
  int *lda,
  double *B,
  int *ldb,
  double *beta,
  double *C,
  int *ldc);

// Symmetric matrix-matrix multiplication. A itself does not have to be symmetric
// Produit matrice-matrice symmetrique. A elle-meme n'a pas a etre symmetrique
// C = alpha A A^T + beta C, C = alpha A^T A + beta C
extern "C" void dsyrk_(
  char *uplo,
  char *trans,
  int *n,
  int *k,
  double *alpha,
  double *A,
  int *lda,
  double *beta,
  double *C,
  int *ldc);

// Lower/upper triangular system solve for a block of column vectors
// Resolution d'un systeme triangulaire inferieure/superieure pour un bloc de vecteur de colonne 
// L B_new = B, U B_new = B (BLAS3)
extern "C" void dtrsm_(
  char *side,
  char *uplo,
  char *transA,
  char *diag,
  int *m,
  int *n,
  double *alpha,
  double *A,
  int *lda,
  double *B,
  int *ldb);

// Cholesky factorization L L^T = A or U UˆT = A (LAPACK). A is overwritten by L or L^T
// Once computed, you can solve A x = b as follows:
// Factorisation Cholesky L L^T = A or U UˆT = A (LAPACK). A est surecrit par L ou L^T
// Une fois calculee, on peut resoudre A x = B comme suit:
// A x = b 
// L L^T x = b         -> dpotrf
// L L^T x = L y = b   -> dtrsv
// L^T x = y           -> dtrsv
extern "C" void dpotrf_(
  char *uplo,
  int *n,
  double *A,
  int *lda,
  int *info);

// Cholesky factorization of a small block of matrix; use only when n is small
// Factorisation Cholesky pour un petit bloc de matrice; a utiliser seulement si n est petit
extern "C" void dpotf2_(
    char *uplo,
    int *n,
    double *A,
    int *lda,
    int *info);

int main()
{
  // Dimension of matrices / Dimensions des matrices
  int N = 32; 
  // Block size for the task-parallel blocked potrf code / Taille de bloc pour potrf parallele par bloc a base de taches
  int BS = 8;  
  // Matrices
  std::vector<double> L(N * N), A(N * N), B(N * N);
  // Vectors
  std::vector<double> x(N), b(N), b2(N);

  // Initialize random number generator
  std::srand(std::time(nullptr));

  // Generate a lower-triangular N x N matrix L with random values between 0.0 and 1.0
	// Optional: Use LAPACK random matrix generator dlatmr
  // Generer une matrice triangulaire inferieure L de taille N x N avec valeurs aleatoires entre 0.0 et 1.0
  // Optionnel: Utiliser la fonction generatrice de matrice aleatoire dlatmr dans LAPACK
  // TODO / A FAIRE
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {
      if (i >= j) {
        L[j * N + i] = static_cast<double>(std::rand()) / RAND_MAX; // Column-major
      }
    }
  }
  for (int i = 0; i < N; ++i) {
    b[i] = static_cast<double>(std::rand()) / RAND_MAX;
  }
  std::cout << "Matrix L:" << std::endl;
  printMatrix(L, N, N);
  std::cout << "Vector b:" << std::endl;
  printMatrix(b, N, 1);

  // Generate a symmetric positive definite matrix A = L * L^T using the dgemm function (BLAS3)
  // Generer une matrice symmetrique positive definite A = L * L^T avec la fonction dgemm (BLAS3)
  // TODO / A FAIRE
  char trans = 'N';
  char transT = 'T'; // Transpose for the second matrix
  double alpha = 1.0;
  double beta = 0.0;
  dgemm_(&trans, &transT, &N, &N, &N, &alpha, &L[0], &N, &L[0], &N, &beta, &A[0], &N);
  std::cout << "Matrix A (after L * L^T):" << std::endl;
  printMatrix(A, N, N);

  // Perform a Cholesky factorization on the matrix A, A = L LˆT using the potrf function (LAPACK)
  // Effecture une factorisation Cholesky sur la matrice A, A = L L^T avec la fonction potrf (LAPACK)
  // TODO / A FAIRE
  char uplo = 'L';
  int info;
  // we copy A before doing the cholesky factorization so we can use B for the verification step
  B = A;
  dpotrf_(&uplo, &N, &A[0], &N, &info);
  if (info != 0) {
    std::cerr << "dpotrf failed with info = " << info << std::endl;
    return 1;
  }
  std::cout << "Matrix A (after Cholesky factorization):" << std::endl;
  printMatrix(A, N, N);

  // Solve the linear system A x = L L^T x = b by first solving L y = b, then solving LˆT x = y, with two successive
  // calls to dtrsv
  // Resoudre le systeme lineaire A x = L L^T x = b, d'abort en resolvant L y = b, ensuite LˆT x = y avec deux appels
  // successifs au dtrsv.
  // TODO / A FAIRE
  // Solve L y = b
  std::copy(b.begin(), b.end(), b2.begin());
  char diag = 'N';
  int incx = 1;
  dtrsv_(&uplo, &trans, &diag, &N, &A[0], &N, &b2[0], &incx);
  std::cout << "Solution vector of L y = b:" << std::endl;
  printMatrix(b2, N, 1);

  // Solve L^T x = y
  dtrsv_(&uplo, &transT, &diag, &N, &A[0], &N, &b2[0], &incx);
  // std::cout << "Solution vector of L^T x = y:" << std::endl;
  // printMatrix(b2, N, 1);

  std::copy(b2.begin(), b2.end(), x.begin());

  // Verify the solution x by computing b2 = A x using dgemv, then compare it to the initial right hand side vector by
  // computing (b - b2) using daxpy, and computing the norm of this vector~(which is the error) by dnrm2
  // Verifier la solution x en calculant b2 = A x avec dgemv, puis en le comparant au second membre initial en calculant
  // (b - b2) avec daxpy d'abord, en suite en calculant la norme de ce vecteur~(ce qui est l'erreur) avec dnrm2.
  // TODO / A FAIRE
  // Verify the solution by computing b2 = A x
  alpha = 1.0;
  beta = 0.0;
  dgemv_(&trans, &N, &N, &alpha, &B[0], &N, &x[0], &incx, &beta, &b2[0], &incx);
  std::cout << "Vector b2 (after verification):" << std::endl;
  printMatrix(b2, N, 1);

  alpha = -1.0;
  daxpy_(&N, &alpha, &b[0], &incx, &b2[0], &incx);
  std::cout << "Vector b2 - b:" << std::endl;
  printMatrix(b2, N, 1);

  double error = dnrm2_(&N, &b2[0], &incx);
  std::cout << "Error norm: " << error << std::endl;

  // Now implement a blocked version of the potrf yourself using dpotf2, dtrsm, dsyrk, and dgemm routines, using block
  // size BS.
  // Use OpenMP Tasks to parallelize the computation, specifying dependencies between operations.
  // Use the priority clause in your task generation to prioritize the critical path in the dependency graph.
  // Maintenant, implanter une version tuilee de potrf avec fonctions dpotf2, dtrsm, dsyrk, et dgemm, en utilisant une
  // taille de tuile BS.
  // Utiliser OpenMP Tasks afin de paralleliser le calcul, en precisant les dependences entre les operations.
  // Employer la clause priority dans la generation des taches afin de prioriser les taches sur le chemin critique.
  // TODO / A FAIRE
  // #pragma omp parallel
  // {
  //   #pragma omp single
  //   {
  //     for (int k = 0; k < N; k += BS) {
  //       int blockSize = std::min(BS, N - k);

  //       // Task 1: Factorize the diagonal block
  //       #pragma omp task depend(inout:A[k*N+k]) priority(1)
  //       dpotf2_(&uplo, &blockSize, &A[k * N + k], &N, &info);

  //       for (int i = k + BS; i < N; i += BS) {
  //         int height = std::min(BS, N - i);

  //         // Task 2: Solve L21 = A21 * L11'
  //         #pragma omp task depend(in:A[k*N+k]) depend(inout:A[i*N+k]) priority(2)
  //         dtrsm_("Right", "Lower", "Transpose", "Non-unit", &height, &blockSize, &alpha, &A[k * N + k], &N, &A[i * N + k], &N);

  //         for (int j = k + BS; j <= i; j += BS) {
  //           int width = std::min(BS, N - j);

  //           // Task 3: Update A22 = A22 - L21 * L21'
  //           #pragma omp task depend(in:A[i*N+k]) depend(in:A[j*N+k]) depend(inout:A[i*N+j]) priority(3)
  //           dgemm_(&trans, &transT, &height, &width, &blockSize, &alpha, &A[i * N + k], &N, &A[j * N + k], &N, &beta, &A[i * N + j], &N);
  //         }
  //       }
  //     }
  //   }
  // }

  // dgemv_(&trans, &N, &N, &alpha, &B[0], &N, &x[0], &1, &beta, &b2[0], &incx);

  // // Compare b2 with the original b
  // double error_norm = 0.0;
  // for (int i = 0; i < N; ++i) {
  //     b2[i] -= b[i];
  //     error_norm += b2[i] * b2[i];
  // }

  // std::cout << "Error norm: " << error_norm << std::endl;

  return 0;
}
