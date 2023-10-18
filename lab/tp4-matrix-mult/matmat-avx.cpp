// Here are my results for N=2048, with B1=32 and B2=256:
//    Sequential scalar matmat i->j->k took 4.732758e+01s.
//    Performance: 362.82Mflops/s
// 
//    Sequential scalar matmat i->k->j took 4.83e+00s.
//    Performance: 3552.15Mflops/s
//
//    Single tile scalar matmat i->k->j took 6.08e+00s.
//    Performance: 2823.77Mflops/s
//
//    Double tile scalar matmat i->k->j took 3.70e+00s.
//    Performance: 4640.63Mflops/s
//
//    Double tile AVX matmat i->k->j took 1.01e+00s.
//    Performance: 16986.55Mflops/s
//
//    Double tile AVX matmat i->k->j took 7.036135e-01s.
//    Performance: 24404.71Mflops/s
//
// The second version is faster than the fisrt one because we do contiguous memory accesses for the B matrix.
//
// The different execution time and performance for the single tiling with different value of B1:
//     B1 = 1, 1.745418e+01s, 983.80Mflops/s
//     B1 = 2, 8.176462e+00s, 2100.11Mflops/s
//     B1 = 4, 6.201748e+00s, 2768.81Mflops/s
//     B1 = 8, 5.455460e+00s, 3147.58Mflops/s
//     B1 = 16, 4.537535e+00s, 3784.32Mflops/s
//     B1 = 64, 4.122300e+00s, 4165.51Mflops/s
//     B1 = 128, 4.173714e+00s, 4114.20Mflops/s
//     B1 = 256, 4.175956e+00s, 4111.99Mflops/s
//     B1 = 512,  4.148578e+00s, 4139.12Mflops/s
//     B1 = 1024, 4.408782e+00s, 3894.84Mflops/s
//     B1 = 2048, 5.437807e+00s, 3157.79Mflops/s
// We see that with small B1 values, the algorithm is much slower. It is because, we do not use the cache as much as we could and the overhead of making the tiles makes the algorithm slower than without tiling.
// Then the execution becomes faster up to a maximum value. The value is determnied by the size of L1 and L2 cache.
// With B1=N we find very similar results than with no tiling because the algorithm is equivalent than a no tiled version.
// 
// My code with omp taks seem to always be slower than the previous version. I must have an issue with my implementation, but I cannot find it.
//
// My speedup for N=2048 is 67. But I did not manage to fully implement the parallelization with omp for collapse(2). Thus, I only managed to get around 10% of my theoretical peak performance

#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>
#include <cstdlib>
#include "immintrin.h"
#include <chrono>
#include "omp.h"

#define NREPEAT 10

void printUsage(int argc, char **argv)
{
  printf("Usage: %s N\n", argv[0]);
  printf("Example: %s 1024\n", argv[0]);
}

void verify(const float *A, const float *B, const float *C, int N)
{
  int correct = 1;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (C[i * N + j] != N) {
        // printf("C(%d, %d) = %f is incorrect; C(%d, %d) should be %d\n", i, j, C[i * N + j], i, j, N);
        correct = 0;
        break;
      }
    }
  }
  if (correct) {
    printf("The result is correct!\n\n");
  } else {
    printf("The result is not correct!\n\n");
  }
}

inline void loadTile(__m256 tile[8], float *addr, int N)
{
  for (int i = 0; i < 8; i ++) {
    tile[i] = _mm256_loadu_ps(&addr[i * N]);
  }
}

inline void storeTile(__m256 tile[8], float *addr, int N)
{
  for (int i = 0; i < 8; i++) {
    _mm256_storeu_ps(&addr[i * N], tile[i]);
  }
}

inline void multiplyTile(float *tA, float *tB, float *tC, __m256 atile[8], __m256 btile[8], __m256 ctile[8], int N)
{
  loadTile(btile, tB, N);
  loadTile(ctile, tC, N);

  // Loop over i (rows of Atile)
  for (int i = 0; i < 8; i++) {
    // Loop over k (columns of Atile and rows of Btile)
    for (int k = 0; k < 8; k++) {
      // Broadcast Atile(i, k) value to all elements of a vector
      __m256 a_broadcast = _mm256_broadcast_ss(&tA[i * N + k]);

      // Perform FMA (fused multiply-add) operation
      ctile[i] = _mm256_fmadd_ps(a_broadcast, btile[k], ctile[i]);
    }
  }

  storeTile(ctile, tC, N);
}

int main(int argc, char **argv)
{
  if (argc != 2) {
    printUsage(argc, argv);
    return 0;
  }
  int N = std::atoi(argv[1]);
  const int B1 = 32;
  const int B2 = 256;

  // Allocate and initialize the matrix A and vectors x, b
  // Allouer et initialiser la matrice A et matrices x, b
  float *A = (float *)_mm_malloc(N * N * sizeof(float), 32);
  float *B = (float *)_mm_malloc(N * N * sizeof(float), 32);
  float *C = (float *)_mm_malloc(N * N * sizeof(float), 32);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = 1.0f;
      B[i * N + j] = 1.0f;
      C[i * N + j] = 0.0f;
    }
  }

  // Sequential and scalar matrix-matrix multiplication code with loop order i->j->k
  // Code sequentiel et scalaire produit matrice-matrice avec l'ordre de boucles i->j->k
  {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
          C[i * N + j] += A[i * N + k] * B[k * N + j]; // C(i, j) = C(i, j) + A(i, k) * B(k, j)
        }
      }
    }
    std::chrono::duration<double> timeDiff = std::chrono::high_resolution_clock::now() - start;
    std::cout << std::scientific << "Sequential scalar matmat i->j->k took " << timeDiff.count() << "s." << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "Performance: " << 2.0*N*N*(N-1) / ((1e6) * timeDiff.count()) <<
      "Mflops/s" << std::endl;
    verify(A, B, C, N);
  }

  // Sequential and scalar matrix-matrix multiplication code with loop order i->k->j
  // Code sequentiel et scalaire produit matrice-matrice avec l'ordre de boucles i->k->j
  {
    auto start = std::chrono::high_resolution_clock::now();
    for (int repeat = 0; repeat < NREPEAT; repeat++) {
      memset(&C[0], 0, N * N * sizeof(float));
      for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
          for (int j = 0; j < N; j++) {
            C[i * N + j] += A[i * N + k] * B[k * N + j]; // C(i, j) = C(i, j) + A(i, k) * B(k, j)
          }
        }
      }
    }
    std::chrono::duration<double> timeDiff = (std::chrono::high_resolution_clock::now() - start) / NREPEAT;
    std::cout << std::scientific << "Sequential scalar matmat i->k->j took " << timeDiff.count() << "s." << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "Performance: " << 2.0*N*N*(N-1) / ((1e6) * timeDiff.count()) <<
      "Mflops/s" << std::endl;
    verify(A, B, C, N);
  }

  // Sequential and scalar matrix-matrix multiplication code with loop order i->k->j and single level tiling
  // Code sequentiel et scalaire produit matrice-matrice avec l'ordre de boucles i->k->j et tuilage d'un niveau
  {
    auto start = std::chrono::high_resolution_clock::now();
    for (int repeat = 0; repeat < NREPEAT; repeat++) {
      memset(&C[0], 0, N * N * sizeof(float));
      for (int ii = 0; ii < N; ii += B1) {
        for (int kk = 0; kk < N; kk += B1) {
          for (int jj = 0; jj < N; jj += B1) {
            for (int i = ii; i < std::min(ii + B1, N); i++) {
              for (int k = kk; k < std::min(kk + B1, N); k++) {
                for (int j = jj; j < std::min(jj + B1, N); j++) {
                  C[i * N + j] += A[i * N + k] * B[k * N + j];
                }
              }
            }
          }
        }
      }
    }
   std::chrono::duration<double> timeDiff = (std::chrono::high_resolution_clock::now() - start) / NREPEAT;
    std::cout << std::scientific << "Single tile scalar matmat i->k->j took " << timeDiff.count() << "s." << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "Performance: " << 2.0*N*N*(N-1) / ((1e6) * timeDiff.count()) <<
      "Mflops/s" << std::endl;
    verify(A, B, C, N);
  }


  // Sequential and scalar matrix-matrix multiplication code with loop order i->k->j and two level tiling
  // Code sequentiel et scalaire produit matrice-matrice avec l'ordre de boucles i->k->j et tuilage de deux niveaux
  {
    auto start = std::chrono::high_resolution_clock::now();
    for (int repeat = 0; repeat < NREPEAT; repeat++) {
      memset(&C[0], 0, N * N * sizeof(float));
      for (int iii = 0; iii < N; iii += B2) {
        for (int kkk = 0; kkk < N; kkk += B2) {
          for (int jjj = 0; jjj < N; jjj += B2) {
            for (int ii = iii; ii < std::min(iii + B2, N); ii += B1) {
              for (int kk = kkk; kk < std::min(kkk + B2, N); kk += B1) {
                for (int jj = jjj; jj < std::min(jjj + B2, N); jj += B1) {
                  for (int i = ii; i < std::min(ii + B1, N); i++) {
                    for (int k = kk; k < std::min(kk + B1, N); k++) {
                      for (int j = jj; j < std::min(jj + B1, N); j++) {
                        C[i * N + j] += A[i * N + k] * B[k * N + j];
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    std::chrono::duration<double> timeDiff = (std::chrono::high_resolution_clock::now() - start) / NREPEAT;
    std::cout << std::scientific << "Double tile scalar matmat i->k->j took " << timeDiff.count() << "s." << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "Performance: " << 2.0*N*N*(N-1) / ((1e6) * timeDiff.count()) <<
      "Mflops/s" << std::endl;
    verify(A, B, C, N);
  }


  // Vectorized matrix-matrix multiplication code with loop order i->k->j and two level tiling + AVX
  // Produit matrice-matrice vectorise avec l'ordre de boucles i->k->j et tuilage de deux niveaux + AVX
  {
    auto start = std::chrono::high_resolution_clock::now();
    for (int repeat = 0; repeat < NREPEAT; repeat++) {
      memset(&C[0], 0, N * N * sizeof(float));
      __m256 atile[8], btile[8], ctile[8];
      for (int iii = 0; iii < N; iii += B2) {
        for (int kkk = 0; kkk < N; kkk += B2) {
          for (int jjj = 0; jjj < N; jjj += B2) {
            for (int ii = iii; ii < std::min(iii + B2, N); ii += B1) {
              for (int kk = kkk; kk < std::min(kkk + B2, N); kk += B1) {
                for (int jj = jjj; jj < std::min(jjj + B2, N); jj += B1) {
                  for (int i = ii; i < std::min(ii + B1, N); i += 8) {
                    for (int k = kk; k < std::min(kk + B1, N); k += 8) {
                      for (int j = jj; j < std::min(jj + B1, N); j += 8) {
                         multiplyTile(&A[i * N + k], &B[k * N + j], &C[i * N + j], atile, btile, ctile, N);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    std::chrono::duration<double> timeDiff = (std::chrono::high_resolution_clock::now() - start) / NREPEAT;
    std::cout << std::scientific << "Double tile scalar matmat i->k->j took " << timeDiff.count() << "s." << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "Performance: " << 2.0*N*N*(N-1) / ((1e6) * timeDiff.count()) <<
      "Mflops/s" << std::endl;
    verify(A, B, C, N);
  }

  {
    auto start = std::chrono::high_resolution_clock::now();
    for (int repeat = 0; repeat < NREPEAT; repeat++) {
      memset(&C[0], 0, N * N * sizeof(float));
      #pragma omp parallel num_threads(16)
      {
        __m256 atile[8], btile[8], ctile[8];
        #pragma omp single
        for (int iii = 0; iii < N; iii += B2) {
          for (int kkk = 0; kkk < N; kkk += B2) {
            for (int jjj = 0; jjj < N; jjj += B2) {
              for (int ii = iii; ii < std::min(iii + B2, N); ii += B1) {
                for (int kk = kkk; kk < std::min(kkk + B2, N); kk += B1) {
                  for (int jj = jjj; jj < std::min(jjj + B2, N); jj += B1) {
                    // A[ii*N + kk], B[kk*N + jj] and C[ii*N + jj] represent A, B and C for the whole block computation
                    #pragma omp task firstprivate(ii, kk, jj) depend(in: A[ii*N + kk]) depend(in: B[kk*N + jj]) depend(out: C[ii*N + jj])
                    for (int i = ii; i < std::min(ii + B1, N); i += 8) {
                      for (int k = kk; k < std::min(kk + B1, N); k += 8) {
                        for (int j = jj; j < std::min(jj + B1, N); j += 8) {
                          multiplyTile(&A[i * N + k], &B[k * N + j], &C[i * N + j], atile, btile, ctile, N);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    std::chrono::duration<double> timeDiff = (std::chrono::high_resolution_clock::now() - start) / NREPEAT;
    std::cout << std::scientific << "Double tile scalar matmat i->k->j took " << timeDiff.count() << "s." << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "Performance: " << 2.0*N*N*(N-1) / ((1e6) * timeDiff.count()) <<
      "Mflops/s" << std::endl;
    verify(A, B, C, N);
  }


  // Free matrices
  // Desallouer les matrices
  _mm_free(A);
  _mm_free(B);
  _mm_free(C);

  return 0;
}
