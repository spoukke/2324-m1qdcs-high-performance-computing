/**
  * Results for N = 1024:
  *   Copier sans AVX: 6.559890e-07s
  *   Copier avec AVX: 1.370410e-07s
  *   AVX acceleration: 4.786808e+00
  *   AVX efficiency: 5.983510e-01
  *   Copier avec AVX et deroulement: 3.426600e-08s
  *   AVX Unrolled acceleration: 1.914402e+01
  *   AVX Unrolled efficiency: 2.393003e+00
  * The execution is faster with Unrolling. This is probably explainable by the fact that unrolling allows us to avoid the latency of the CPU.
  * Unfortunately I was not able to find the information of the latency for my CPU (AMD® Ryzen 7 pro 5850u) on the _mm256_load_ps action.
  */

#include <immintrin.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>

#define NREPET 1000

void afficherUsage()
{ 
  printf("Usage: ./copier-vecteur-avx [taille-du-tableau]\n");
}

void copyScalar(float* A, float* B, int N)
{
  for (int i = 0; i < N; i++) {
    B[i] = A[i];
  }
}

void copyAVX(float* A, float* B, int N)
{
  int i;
  // Perform the copy in 256-bit chunks (= 8 floats at a time)
  for (i = 0; i < N - 7; i += 8) {
    __m256 vec = _mm256_load_ps(&A[i]);
    _mm256_store_ps(&B[i], vec);
  }
  // Copy any remaining elements (less than 8)
  for (; i < N; i++) {
    B[i] = A[i];
  }
}

void copyAVXUnrolled(float* A, float* B, int N)
{
  int i;
  // Perform the copy in 256-bits chunks with unrolling
  for (i = 0; i < N - 31; i += 32) {
    __m256 vec0 = _mm256_load_ps(&A[i]);
    __m256 vec1 = _mm256_load_ps(&A[i + 8]);
    __m256 vec2 = _mm256_load_ps(&A[i + 16]);
    __m256 vec3 = _mm256_load_ps(&A[i + 24]);

    _mm256_store_ps(&B[i], vec0);
    _mm256_store_ps(&B[i + 8], vec1);
    _mm256_store_ps(&B[i + 16], vec2);
    _mm256_store_ps(&B[i + 24], vec3);
  }

  // Copy any remaining elements (less than 32)
  for (; i < N; i++) {
    B[i] = A[i];
  }
}

int main(int argc, char **argv)
{
  if (argc < 2) { 
    afficherUsage();
    return 1;
  }
  int dim = std::atoi(argv[1]);

  // Allouer et initialiser deux tableaux de flottants de taille dim alignes par 32 octets
  float* tab0 = (float*)_mm_malloc(dim * sizeof(float), 32);
  float* tab1 = (float*)_mm_malloc(dim * sizeof(float), 32);
  
  for (int i = 0; i < dim; i++) {
    tab0[i] = static_cast<float>(i);
  }
  
  // Copier tab0 dans tab1 de manière scalaire~(code non-vectorise).
  // On repete NREPET fois pour mieux mesurer le temps d'execution
  auto start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    copyScalar(tab0, tab1, dim);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diffSeq = end-start;
  double seqTime = diffSeq.count() / NREPET;
  std::cout << std::scientific << "Copier sans AVX: " << seqTime << "s" << std::endl;

  // Copier tab0 dans tab1 de maniere vectorisee avec AVX
  start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    copyAVX(tab0, tab1, dim);
  }
  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diffPar = end-start;
  double avxTime = diffPar.count() / NREPET;
  std::cout << std::scientific << "Copier avec AVX: " << diffPar.count() / NREPET << "s" << std::endl;
  // Afficher l'acceleration et l'efficacite
  double accAVX = seqTime / avxTime;
  double effAVX = accAVX / 8; // Proc is AMD® Ryzen 7 pro 5850u. It has 8 cores.
  std::cout << "AVX acceleration: " << accAVX << std::endl;
  std::cout << "AVX efficiency: " << effAVX << std::endl;

  // Copier tab0 dans tab1 de maniere vectorisee avec AVX et deroulement de facteur 4
  start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    copyAVXUnrolled(tab0, tab1, dim);
  }
  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diffParDeroule = end-start;
  double avxUnrolledTime = diffParDeroule.count() / NREPET;
  std::cout << std::scientific << "Copier avec AVX et deroulement: " << avxUnrolledTime << "s" << std::endl;
  // Afficher l'acceleration et l'efficacite
  double accAVXUnrolled = seqTime / avxUnrolledTime;
  double effAVXUnrolled = accAVXUnrolled / 8; // Proc is AMD® Ryzen 7 pro 5850u. It has 8 cores.
  std::cout << "AVX Unrolled acceleration: " << accAVXUnrolled << std::endl;
  std::cout << "AVX Unrolled efficiency: " << effAVXUnrolled << std::endl;

  // Desallouer les tableaux tab0 et tab1
  _mm_free(tab0);
  _mm_free(tab1);

  return 0;
}
