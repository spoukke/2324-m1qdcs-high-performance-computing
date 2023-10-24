// |            | 1 thread  | 2 threads | 4 threads | 8 threads |
// | ---------- | --------- | --------- | --------- | --------- |
// | 1 stride   | 0.25788s  | 0.255618s | 0.272964s | 0.249507s |
// | 2 strides  | 0.325684s | 0.261843s | 0.235435s | 0.242224s |
// | 4 strides  | 0.264313s | 0.256894s | 0.238892s | 0.370166s |
// | 8 strides  | 0.260409s | 0.243062s | 0.275562s | 0.244034s |
// | 16 strides | 0.287775s | 0.333961s | 0.390888s | 0.257152s |
// | 32 strides | 0.356716s | 0.480114s | 0.257916s | 0.249738s |
// | 64 strides | 0.326678s | 0.268314s | 0.278913s | 0.255761s |
//
// I cannit dfferentiate any pattern in the time execution by threads or strides.

#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include "omp.h"

#define NREPEAT 128
#define NTHREADMAX 8
#define STRIDE 64

int main()
{
  int N = 10000000;
  float sum[NTHREADMAX * STRIDE] __attribute__((aligned(64))) = {0};
  std::vector<float> vec(N);
  vec[0] = 0;
  for (int i = 1; i < N; i++) { 
    vec[i] = 1;
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int repeat = 0; repeat < NREPEAT; repeat++) {
    for (int i = 0; i < NTHREADMAX; i++) { sum[i * STRIDE] = 0.0; }
#pragma omp parallel
    {
      int thid = omp_get_thread_num();
#pragma omp for
      for (int i = 0; i < N; i++) {
        sum[thid * STRIDE] += vec[i];
      }
    }
  }
  std::chrono::duration<double> time = std::chrono::high_resolution_clock::now() - start;
  std::cout << "Time: " << time.count() << "s\n";

  float sumFinal = 0.0;
  for (int i = 0; i < NTHREADMAX; i++) { sumFinal += sum[i]; }
  printf("sum = %f", sumFinal);

  return 0;
}
