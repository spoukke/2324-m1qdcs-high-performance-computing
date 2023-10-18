// Peak Compute Performance: 0.36 Gflops/s
// Theoretical Peak Performance: 243.60 Gflops/s

#include <iostream>
#include <chrono>
#include <immintrin.h>  // For AVX intrinsics

int main() {
    const int N = 1000000000;
    volatile __m256 a = _mm256_set1_ps(1.0f);
    volatile __m256 b = _mm256_set1_ps(2.0f);
    volatile __m256 c = _mm256_set1_ps(3.0f);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i += 8) {
        a = _mm256_fmadd_ps(a, b, c);
        a = _mm256_fmadd_ps(a, b, c);
        a = _mm256_fmadd_ps(a, b, c);
        a = _mm256_fmadd_ps(a, b, c);
        a = _mm256_fmadd_ps(a, b, c);
        a = _mm256_fmadd_ps(a, b, c);
        a = _mm256_fmadd_ps(a, b, c);
        a = _mm256_fmadd_ps(a, b, c);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    double gflops = ((double)N * 8.0 / (double)duration) * 1e-6; // x8 because of 8 FP32 operations per AVX instruction
    
    std::cout << "Measured performance: " << gflops << " Gflops/s" << std::endl;
    // Here, you would compare this to the theoretical performance based on the formula provided
}
