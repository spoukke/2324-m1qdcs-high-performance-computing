/**
  * Copie d'un tableau dans un autre avec les intrinseques AVX.
  * A compiler avec les drapeaux -O2 -mavx2.
  */

#include <immintrin.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>

#define NREPET 1024

void afficherUsage()
{ 
  printf("Usage: ./inversion-tableau-avx [taille-du-tableau]\n");
}

void reverseArray(float* arr, int N)
{
  int i = 0;
  int j = N - 1;

  while (i < j)
  {
    std::swap(arr[i], arr[j]);
    i++;
    j--;
  }
}

void reverseArrayAVX(float* arr, int N)
{
  int i = 0;
  int j = N - 8;

  while (i < j)
  {
    __m256i idx = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7); // Reverse permutation index
    __m256 a = _mm256_load_ps(&arr[i]);
    __m256 b = _mm256_load_ps(&arr[j]);

    // Reverse the vectors using permutation
    a = _mm256_permutevar8x32_ps(a, idx);
    b = _mm256_permutevar8x32_ps(b, idx);

    // Store the reversed vectors back into the array
    _mm256_storeu_ps(&arr[i], b);
    _mm256_storeu_ps(&arr[j], a);

    i += 8;
    j -= 8;
  }

  // Handle the remaining elements with scalar operations
  j += 7;
  while (i < j)
  {
    std::swap(arr[i], arr[j]);
    i++;
    j--;
  }
}

int main(int argc, char **argv)
{
  if (argc < 2) { 
    afficherUsage();
    return 1;
  }
  int dim = std::atoi(argv[1]);
  
  // Allouer et initialiser tableau tab de taille dim aligne par 32 octets
  float *tab = (float*)_mm_malloc(dim * sizeof(float), 32);
  for (int i = 0; i < dim; i++) {
    tab[i] = static_cast<float>(i);
  }
  
  // Inverser le tableau en place~(c'est a dire sans utiliser un deuxieme tableau auxiliaire) sans vectorisation
  // On repete NREPET fois pour mieux mesurer le temps d'execution
  auto start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    reverseArray(tab, dim);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diffSeq = end-start;
  std::cout << std::scientific << "Inversion sans AVX: " << diffSeq.count() / NREPET << "s" << std::endl;

  // Inverser le tableau en place avec AVX~(c'est a dire sans utiliser un deuxieme tableau auxiliaire)
  // On repete NREPET fois pour mieux mesurer le temps d'execution
  start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    reverseArrayAVX(tab, dim);
  }
  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diffPar = end-start;
  std::cout << std::scientific << "Inversion avec AVX: " << diffPar.count() / NREPET << "s" << std::endl;

  // Afficher l'acceleration et l'efficacite
  double tempsSeq = diffSeq.count() / NREPET;
  double tempsAVX = diffPar.count() / NREPET;

  double acc = tempsSeq / tempsAVX;
  double eff = acc / 8;

  std::cout << std::scientific << "Acceleration avec AVX: " << acc << "s" << std::endl;
  std::cout << std::scientific << "Efficacite avec AVX: " << eff / NREPET << "s" << std::endl;

  // Desallouer le tableau tab
  _mm_free(tab);

  return 0;
}
