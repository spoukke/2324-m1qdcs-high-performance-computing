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

  __m256 vec, vec2, vec4; // floats

//  vec = _mm256_set_ps(7., 6., 5., 4., 3., 2., 1., 0.);
//  vec = _mm256_set1_ps(7.);
  vec2 = _mm256_set1_ps(4.);
  vec = _mm256_loadu_ps(&A[0]);
//  vec4 = _mm256_mul_ps(vec, vec2);
//  vec = _mm256_add_ps(vec4, vec);
  vec = _mm256_fmadd_ps(vec2, vec, vec);
//  vec2 = _mm256_mul_ps(vec, vec);
//  vec2 = _mm256_sub_ps(vec, vec);
//  vec2 = _mm256_div_ps(vec, vec);
  _mm256_storeu_ps(&A[0], vec);

//  _mm256_storeu_ps(&A[0], _mm256_add_ps(_mm256_loadu_ps(&A[0]), _mm256_loadu_ps(&A[0])));

  for (int i = 0; i < 8; i++) {
    printf("%f ", A[i]);
  }
  printf("\n");

  double B[4];
  __m256d vec3; // double
  vec3 = _mm256_set_pd(4., 3., 2., 1);

  vec3 = _mm256_add_pd(vec3, vec3);

  _mm256_storeu_pd(&B[0], vec3);

  for (int i = 0; i < 4; i++) {
    printf("%lf ", B[i]);
  }
  printf("\n");

  __m256i idx = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);

  vec = _mm256_permutevar8x32_ps(vec, idx);
  _mm256_storeu_ps(&A[0], vec);

  for (int i = 0; i < 8; i++) {
    printf("%f ", A[i]);
  }
  printf("\n");

  return 0;
}
