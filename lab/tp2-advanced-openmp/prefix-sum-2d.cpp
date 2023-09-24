#include <iostream>
#include <chrono>
#include "omp.h"

#define N 4096
#define K 4
#define NTASKS (N / K)

double A[N][N];
double B[N][N];
bool deps[NTASKS + 1][NTASKS + 1];

void printArray(double tab[N][N])
{
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%4.0lf ", tab[i][j]);
    }
    printf("\n");
  }
  printf("\n");
}

int main(int argc, char **argv)
{
  // Initialize the array A[i][j] = i + j in parallel.
  // Version 1: Using omp for loop. We can collapse two loops since each iteration (i, j) is independent
  // Version 1: Avec boucle omp for. On peut faire collapse pour deux boucles car chaque iteration (i, j) est
  // independante
  // Initialize the array A[i][j] = i + j in parallel.
#pragma omp parallel for collapse(2)
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      A[i][j] = i + j;

    // Compute array B in parallel
  auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for collapse(2)
  for (int x = 0; x < N; x++) {
    for (int y = 0; y < N; y++) {
      if (x == 0) {
        if (y == 0) {
          B[x][y] = A[x][y];
        }
        else {
          B[x][y] = B[x][y - 1] + A[x][y];
        }
      }
      else if (y == 0) {
        B[x][y] = B[x - 1][y] + A[x][y];
      }
      else {
        B[x][y] = B[x - 1][y] + B[x][y - 1] - B[x - 1][y - 1] + A[x][y];
      }
    }
  }
  std::chrono::duration<double> temps = std::chrono::high_resolution_clock::now() - start;
  // Version 2: Using tasks, each task initializes a block of size K x K. There are (N / K) x (N / K) tasks in total
  // Version 2: Avec taches, chaque tache initialise un bloc de taille K x K. Il y a (N / K) x (N / K) taches au total

  start = std::chrono::high_resolution_clock::now();
#pragma omp parallel
  {
#pragma omp for schedule(static) collapse(2)
    for (int x = 0; x < N; x++) {
      for (int y = 0; y < N; y++) {
#pragma omp task default(none) shared(x, y, A, B)
        {
          if (x == 0) {
            if (y == 0) {
                B[x][y] = A[x][y];
            }
            else {
                B[x][y] = B[x][y - 1] + A[x][y];
            }
          }
            else if (y == 0) {
              B[x][y] = B[x - 1][y] + A[x][y];
            }
            else {
              B[x][y] = B[x - 1][y] + B[x][y - 1] - B[x - 1][y - 1] + A[x][y];
            }
        }
      }
    }
  }
  std::chrono::duration<double> tempsTasks = std::chrono::high_resolution_clock::now() - start;

#pragma omp parallel default(none) shared(A, B, deps)
{
#pragma omp single
  {
    // Iterate over blocks (bi, bj)
    for (int bi = 0; bi < NTASKS; bi++) {
      for (int bj = 0; bj < NTASKS; bj++) {
#pragma omp task default(none) firstprivate(bi, bj) shared(A, B) \
  depend(out: deps[bi + 1][bj + 1]) depend(in: deps[bi][bj], deps[bi + 1][bj], deps[bi][bj + 1])
        {
          // Iterate over elements within the block (K x K)
          for (int i = bi * K; i < (bi + 1) * K; i++) {
            for (int j = bj * K; j < (bj + 1) * K; j++) {
              if (i == 0) {
                if (j == 0) {
                    B[i][j] = A[i][j];
                }
                else {
                    B[i][j] = B[i][j - 1] + A[i][j];
                }
              }
              else if (j == 0) {
                B[i][j] = B[i - 1][j] + A[i][j];
              }
              else {
                B[i][j] = B[i - 1][j] + B[i][j - 1] - B[i - 1][j - 1] + A[i][j];
              }
            }
          }
        }
      }
    }
  }
}

  if (N < 20) {
    printf("Array A:\n");
    printArray(A);
  }

  std::cout << "Temps 1: " << temps.count() << "s\n";
  std::cout << "Temps 2: " << tempsTasks.count() << "s\n";

  // 
#pragma omp parallel default(none) shared(A, B, deps)
  {
#pragma omp single
    {
      for (int ti = 0; ti < NTASKS; ti++) {
        for (int tj = 0; tj < NTASKS; tj++) {
#pragma omp task default(none) firstprivate(ti, tj) shared(A, B) \
          depend(out:deps[ti + 1][tj + 1]) depend(in:deps[ti][tj],deps[ti + 1][tj],deps[ti][tj + 1])
          for (int i = ti * K; i < (ti + 1) * K; i++) {
            for (int j = tj * K; j < (tj + 1) * K; j++) {
              if (i == 0) { 
                if (j == 0) {
                  B[i][j] = A[i][j];
                } else {
                  B[i][j] = B[i][j - 1] + A[i][j];
                }
              } else if (j == 0) {
                B[i][j] = B[i - 1][j] + A[i][j];
              } else {
                B[i][j] = B[i - 1][j] + B[i][j - 1] - B[i - 1][j - 1] + A[i][j];
              }
            }
          }
        }
      }
    }
  }
  if (N < 20) {
    printf("Array B:\n");
    printArray(B);
  }
  return 0;
}
