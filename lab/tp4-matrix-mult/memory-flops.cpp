#include <iostream>
#include <chrono>
#include <immintrin.h>  // For AVX intrinsics

int main() {
    const int N = 100000000;
    float* A = (float*)_mm_malloc(N * sizeof(float), 32);  // 32-byte aligned memory
    float* B = (float*)_mm_malloc(N * sizeof(float), 32); 

    for (int i = 0; i < N; i++) {
        A[i] = (float)i;
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; i += 8) {
        __m256 va = _mm256_load_ps(&A[i]);
        _mm256_store_ps(&B[i], va);
        __m256 va = _mm256_load_ps(&A[i]);
        _mm256_store_ps(&B[i], va);
        __m256 va = _mm256_load_ps(&A[i]);
        _mm256_store_ps(&B[i], va);
        __m256 va = _mm256_load_ps(&A[i]);
        _mm256_store_ps(&B[i], va);
        __m256 va = _mm256_load_ps(&A[i]);
        _mm256_store_ps(&B[i], va);
        __m256 va = _mm256_load_ps(&A[i]);
        _mm256_store_ps(&B[i], va);
        __m256 va = _mm256_load_ps(&A[i]);
        _mm256_store_ps(&B[i], va);
        __m256 va = _mm256_load_ps(&A[i]);
        _mm256_store_ps(&B[i], va);
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    double bandwidth = ((double)N * sizeof(float) / (double)duration) * 1e-6;
    
    std::cout << "Measured bandwidth: " << bandwidth << " GB/s" << std::endl;
    // Here, you would compare this to the theoretical bandwidth based on the formula provided
    
    _mm_free(A);
    _mm_free(B);
}
