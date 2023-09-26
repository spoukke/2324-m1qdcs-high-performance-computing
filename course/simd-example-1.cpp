#include <iostream>
#include "immintrin.h"

int main()
{
  float A[8] = {0, 1, 2, 3, 4, 5, 6, 7};
//  A[0] = A[0] + A[0];
//  A[1] = A[1] + A[1];
//  A[2] = A[2] + A[2];
//  A[3] = A[3] + A[3];
//  A[4] = A[4] + A[4];
//  A[5] = A[5] + A[5];
//  A[6] = A[6] + A[6];
//  A[7] = A[7] + A[7];

  __m256 vec, vec2;

  vec = _mm256_loadu_ps(&A[0]);
  vec2 = _mm256_add_ps(vec, vec);
  _mm256_storeu_ps(&A[0], vec2);

//  _mm256_storeu_ps(&A[0], _mm256_add_ps(_mm256_loadu_ps(&A[0]), _mm256_loadu_ps(&A[0])));

  for (int i = 0; i < 8; i++) {
    printf("%f ", A[i]);
  }
  printf("\n");

  return 0;
}
