#include <iostream>
#include <immintrin.h>
#include <cstdlib> // for atoi

void matrixVectorMultiply(float* A, float* x, float* y, int size) {
#pragma omp parallel for if (dim >= 64)
    for (int i = 0; i < size - 8; i++) {
        __m256 sum;
        for (int j = 0; j < size; j += 8) {
            __m256 a = _mm256_loadu_ps(&A[i * size + j]);
            __m256 b = _mm256_loadu_ps(&x[j]);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
        }
        _mm256_storeu_ps(&y[i], sum);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        return 1;
    }

    int size = std::atoi(argv[1]);
    if (size % 8 != 0) {
        std::cout << "Matrix size must be a multiple of 8." << std::endl;
        return 1;
    }

    float* A = new float[size * size];
    float* x = new float[size];
    float* y = new float[size];

    // Initialize matrix A and vector x
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            A[i * size + j] = i + j;
        }
        x[i] = i;
    }

    // Perform matrix-vector multiplication
    matrixVectorMultiply(A, x, y, size);

    delete[] A;
    delete[] x;
    delete[] y;

    return 0;
}
