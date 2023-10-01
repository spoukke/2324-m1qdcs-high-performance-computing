/**
  * Execution time results for the different algorithm for a given matrix size:
  *   Produit scalaire sans AVX: 1.763000e-09s
  *   Produit scalaire avec AVX: 5.000000e-11s
  *   Produit scalaire avec AVX Unrolled 2: 3.000000e-11s
  *   Produit scalaire avec AVX Unrolled 4: 3.000000e-11s
  *   Produit scalaire avec AVX FMA: 3.000000e-11s
  *   Produit scalaire avec AVX FMA deroulement de facteur 2: 3.000000e-11s
  *   Produit scalaire avec AVX FMA deroulement de facteur 4: 3.000000e-11s
  * We notice that with unrolling, the algorithgme goes faster. Indeed, we are probably skipping some latency cycles.
  * 
  * With the `-ftree-vectorize` flag, with the same matrix size, the scalar execution takes 7.220000e-10s.
  * It is faster, but not as much as the vectorization we implemented.
  */

#include <immintrin.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>

#define NREPET 1000

float innerProductScalar(const float* x, const float* y, int N)
{
  float result = 0.0f;
  for (int i = 0; i < N; i++) {
    result += x[i] * y[i];
  }
  return result;
}

float innerProductAVX(const float* x, const float* y, int N)
{
  int i;
  __m256 product;
  for (i = 0; i < N - 7; i += 8) {
    __m256 vecX = _mm256_load_ps(&x[i]);
    __m256 vecY = _mm256_load_ps(&y[i]);
    product = _mm256_add_ps(product, _mm256_mul_ps(vecX, vecY));
  }

  float result = 0.0f;
  float resultArr[8];
  _mm256_storeu_ps(resultArr, product);
  for (; i < N; i++) {
    result += x[i] * y[i];
  }

  for (int j = 0; j < 8; j++) {
    result += resultArr[j];
  }

  return result;
}

float innerProductAVXUnrolled2(const float* x, const float* y, int N)
{
  int i;
  __m256 product1;
  __m256 product2;
  for (i = 0; i < N - 15; i += 16) {
    __m256 x_data1 = _mm256_loadu_ps(&x[i]);
    __m256 x_data2 = _mm256_loadu_ps(&x[i + 8]);
    __m256 y_data1 = _mm256_loadu_ps(&y[i]);
    __m256 y_data2 = _mm256_loadu_ps(&y[i + 8]);
    product1 = _mm256_add_ps(product1, _mm256_mul_ps(x_data1, y_data1));
    product2 = _mm256_add_ps(product2, _mm256_mul_ps(x_data2, y_data2));
  }
  float result1 = 0.0f;
  float result2 = 0.0f;
  float resultArr1[8];
  float resultArr2[8];
  _mm256_storeu_ps(resultArr1, product1);
  _mm256_storeu_ps(resultArr2, product2);
  for (; i < N - 7; i += 8) {
    result1 += x[i] * y[i];
    result2 += x[i + 1] * y[i + 1];
    result1 += x[i + 2] * y[i + 2];
    result2 += x[i + 3] * y[i + 3];
    result1 += x[i + 4] * y[i + 4];
    result2 += x[i + 5] * y[i + 5];
    result1 += x[i + 6] * y[i + 6];
    result2 += x[i + 7] * y[i + 7];
  }
  for (int j = 0; j < 8; j++) {
    result1 += resultArr1[j];
    result2 += resultArr2[j];
  }
  float result = result1 + result2;
  for (; i < N; i++) {
    result += x[i] * y[i];
  }
  return result;
}

float innerProductAVXUnrolled4(const float* x, const float* y, int N)
{
  int i;
  __m256 product1;
  __m256 product2;
  __m256 product3;
  __m256 product4;
  for (i = 0; i < N - 31; i += 32) {
    __m256 x_data1 = _mm256_loadu_ps(&x[i]);
    __m256 x_data2 = _mm256_loadu_ps(&x[i + 8]);
    __m256 x_data3 = _mm256_loadu_ps(&x[i + 16]);
    __m256 x_data4 = _mm256_loadu_ps(&x[i + 24]);
    __m256 y_data1 = _mm256_loadu_ps(&y[i]);
    __m256 y_data2 = _mm256_loadu_ps(&y[i + 8]);
    __m256 y_data3 = _mm256_loadu_ps(&y[i + 16]);
    __m256 y_data4 = _mm256_loadu_ps(&y[i + 24]);
    product1 = _mm256_add_ps(product1, _mm256_mul_ps(x_data1, y_data1));
    product2 = _mm256_add_ps(product2, _mm256_mul_ps(x_data2, y_data2));
    product3 = _mm256_add_ps(product3, _mm256_mul_ps(x_data3, y_data3));
    product4 = _mm256_add_ps(product4, _mm256_mul_ps(x_data4, y_data4));
  }
  float result1 = 0.0f;
  float result2 = 0.0f;
  float result3 = 0.0f;
  float result4 = 0.0f;
  float resultArr1[8];
  float resultArr2[8];
  float resultArr3[8];
  float resultArr4[8];
  _mm256_storeu_ps(resultArr1, product1);
  _mm256_storeu_ps(resultArr2, product2);
  _mm256_storeu_ps(resultArr3, product3);
  _mm256_storeu_ps(resultArr4, product4);
  for (; i < N - 7; i += 8) {
    result1 += x[i] * y[i];
    result2 += x[i + 1] * y[i + 1];
    result3 += x[i + 2] * y[i + 2];
    result4 += x[i + 3] * y[i + 3];
    result1 += x[i + 4] * y[i + 4];
    result2 += x[i + 5] * y[i + 5];
    result3 += x[i + 6] * y[i + 6];
    result4 += x[i + 7] * y[i + 7];
  }
  for (int j = 0; j < 8; j++) {
    result1 += resultArr1[j];
    result2 += resultArr2[j];
    result3 += resultArr3[j];
    result4 += resultArr4[j];
  }
  float result = result1 + result2 + result3 + result4;
  for (; i < N; i++) {
    result += x[i] * y[i];
  }
  return result;
}

float innerProductAVXFMA(const float* x, const float* y, int N)
{
  int i;
  __m256 product;
  for (i = 0; i < N- 7; i += 8) {
    __m256 vecX = _mm256_load_ps(&x[i]);
    __m256 vecY = _mm256_load_ps(&y[i]);
    product = _mm256_fmadd_ps(vecX, vecY, product);
  }

  float result = 0.0f;
  float resultArr[8];
  _mm256_storeu_ps(resultArr, product);
  for (; i < N; i++) {
    result += x[i] * y[i];
  }

  for (int j = 0; j < 8; j++) {
    result += resultArr[j];
  }
  
  return result;
}

float innerProductAVXFMAUnrolled2(const float* x, const float* y, int N)
{
  int i;
  __m256 product1;
  __m256 product2;
  for (i = 0; i < N - 15; i += 16) {
    __m256 x_data1 = _mm256_load_ps(&x[i]);
    __m256 x_data2 = _mm256_load_ps(&x[i + 8]);
    __m256 y_data1 = _mm256_load_ps(&y[i]);
    __m256 y_data2 = _mm256_load_ps(&y[i + 8]);
    product1 = _mm256_fmadd_ps(x_data1, y_data1, product1);
    product2 = _mm256_fmadd_ps(x_data2, y_data2, product2);
  }
  float result1 = 0.0f;
  float result2 = 0.0f;
  float resultArr1[8];
  float resultArr2[8];
  _mm256_storeu_ps(resultArr1, product1);
  _mm256_storeu_ps(resultArr2, product2);
  for (; i < N - 7; i += 8) {
    result1 += x[i] * y[i];
    result2 += x[i + 1] * y[i + 1];
    result1 += x[i + 2] * y[i + 2];
    result2 += x[i + 3] * y[i + 3];
    result1 += x[i + 4] * y[i + 4];
    result2 += x[i + 5] * y[i + 5];
    result1 += x[i + 6] * y[i + 6];
    result2 += x[i + 7] * y[i + 7];
  }
  for (int j = 0; j < 8; j++) {
    result1 += resultArr1[j];
    result2 += resultArr2[j];
  }
  float result = result1 + result2;
  for (; i < N; i++) {
    result += x[i] * y[i];
  }
  return result;
}

float innerProductAVXFMAUnrolled4(const float* x, const float* y, int N)
{
  int i;
  __m256 product1;
  __m256 product2;
  __m256 product3;
  __m256 product4;
  for (i = 0; i < N - 31; i += 32) {
    __m256 x_data1 = _mm256_load_ps(&x[i]);
    __m256 x_data2 = _mm256_load_ps(&x[i + 8]);
    __m256 x_data3 = _mm256_load_ps(&x[i + 16]);
    __m256 x_data4 = _mm256_load_ps(&x[i + 24]);
    __m256 y_data1 = _mm256_load_ps(&y[i]);
    __m256 y_data2 = _mm256_load_ps(&y[i + 8]);
    __m256 y_data3 = _mm256_load_ps(&y[i + 16]);
    __m256 y_data4 = _mm256_load_ps(&y[i + 24]);
    product1 = _mm256_fmadd_ps(x_data1, y_data1, product1);
    product2 = _mm256_fmadd_ps(x_data2, y_data2, product2);
    product3 = _mm256_fmadd_ps(x_data3, y_data3, product3);
    product4 = _mm256_fmadd_ps(x_data4, y_data4, product4);
  }
  float result1 = 0.0f;
  float result2 = 0.0f;
  float result3 = 0.0f;
  float result4 = 0.0f;
  float resultArr1[8];
  float resultArr2[8];
  float resultArr3[8];
  float resultArr4[8];
  _mm256_storeu_ps(resultArr1, product1);
  _mm256_storeu_ps(resultArr2, product2);
  _mm256_storeu_ps(resultArr3, product3);
  _mm256_storeu_ps(resultArr4, product4);
  for (; i < N - 7; i += 8) {
    result1 += x[i] * y[i];
    result2 += x[i + 1] * y[i + 1];
    result3 += x[i + 2] * y[i + 2];
    result4 += x[i + 3] * y[i + 3];
    result1 += x[i + 4] * y[i + 4];
    result2 += x[i + 5] * y[i + 5];
    result3 += x[i + 6] * y[i + 6];
    result4 += x[i + 7] * y[i + 7];
  }
  for (int j = 0; j < 8; j++) {
    result1 += resultArr1[j];
    result2 += resultArr2[j];
    result3 += resultArr3[j];
    result4 += resultArr4[j];
  }
  float result = result1 + result2 + result3 + result4;
  for (; i < N; i++) {
    result += x[i] * y[i];
  }
  return result;
}

int main(int argc, char **argv)
{
  if (argc < 2) {
    std::cout << "Utilisation: \n  " << argv[0] << " [taille-de-tableau]\n";
    return 1;
  }
  int dim = std::atoi(argv[1]);
  if (dim % 8) {
    std::cout << "La taille de tableau doit etre un multiple de 8.\n";
    return 1;
  }

  // Allouer les tableaux x et y de flottants de taille dim alignes par 32 octets, puis initialiser x[i]=i et y[i] = 1
  float* x = (float*)_mm_malloc(dim * sizeof(float), 32);;
  float* y = (float*)_mm_malloc(dim * sizeof(float), 32);;
  
  for (int i = 0; i < dim; i++) {
    x[i] = static_cast<float>(i);
    y[i] = 1.0f;
  }

  // Faire le produit scalaire non-vectorise. On repete le calcul NREPET fois pour mieux mesurer le temps d'execution
  auto start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    innerProductScalar(x, y, dim);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> tempsSeq = end-start;
  std::cout << std::scientific << "Produit scalaire sans AVX: " << tempsSeq.count() / NREPET << "s" << std::endl;

  // Faire le produit scalaire vectorise AVX. On repete le calcul NREPET fois pour mieux mesurer le temps d'execution
  start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    innerProductAVX(x, y, dim);
  }
  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> tempsAVX = end-start;
  std::cout << std::scientific << "Produit scalaire avec AVX: " << tempsAVX.count() / NREPET << "s" << std::endl;

  // Faire le produit scalaire vectorise AVX avec unrooling de facteur 2. On repete le calcul NREPET fois pour mieux mesurer le temps d'execution
  start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    innerProductAVXUnrolled2(x, y, dim);
  }
  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> tempsAVXUnrolled2 = end-start;
  std::cout << std::scientific << "Produit scalaire avec AVX Unrolled 2: " << tempsAVXUnrolled2.count() / NREPET << "s" << std::endl;

  // Faire le produit scalaire vectorise AVX avec unrooling de facteur 2. On repete le calcul NREPET fois pour mieux mesurer le temps d'execution
  start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    innerProductAVXUnrolled4(x, y, dim);
  }
  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> tempsAVXUnrolled4 = end-start;
  std::cout << std::scientific << "Produit scalaire avec AVX Unrolled 4: " << tempsAVXUnrolled4.count() / NREPET << "s" << std::endl;

  // Produit scalaire vectorise AVX FMA. On repete le calcul NREPET fois pour mieux mesurer le temps d'execution
  start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    innerProductAVXFMA(x, y, dim);
  }
  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> tempsParAVXFMA = end-start;
  std::cout << std::scientific << "Produit scalaire avec AVX FMA: " << tempsParAVXFMA.count() / NREPET << "s" <<
    std::endl;

  // Produit scalaire vectorise AVX FMA et deroulement de facteur 2. On repete le calcul NREPET fois pour mieux mesurer le temps
  start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    innerProductAVXFMAUnrolled2(x, y, dim);
  }
  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> tempsParAVXFMADeroule2 = end-start;
  std::cout << std::scientific << "Produit scalaire avec AVX FMA deroulement de facteur 2: " << tempsParAVXFMADeroule2.count() /
    NREPET << "s" << std::endl;

  // Produit scalaire vectorise AVX FMA et deroulement de facteur 4. On repete le calcul NREPET fois pour mieux mesurer le temps
  start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    innerProductAVXFMAUnrolled4(x, y, dim);
  }
  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> tempsParAVXFMADeroule4 = end-start;
  std::cout << std::scientific << "Produit scalaire avec AVX FMA deroulement de facteur 4: " << tempsParAVXFMADeroule4.count() /
    NREPET << "s" << std::endl;

  // Desallouer les tableaux tab0 et tab1
  _mm_free(x);
  _mm_free(y);

  return 0;
}
