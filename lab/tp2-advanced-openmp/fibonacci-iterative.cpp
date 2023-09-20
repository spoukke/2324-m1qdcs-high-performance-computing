#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include "omp.h"

void printUsage(int argc, char **argv)
{
  printf("Usage: %s N\n", argv[0]);
  printf("Example: %s 13\n", argv[0]);
}

int main(int argc, char **argv)
{
  // Check the validity of command line arguments and print usage if invalid
  if (argc < 2)
  {
    printUsage(argc, argv);
    return 0;
  }

  // Read the index of the Fibonacci number to compute
  const int N = atoi(argv[1]);

  // Allocate and initialize the array containing Fibonacci numbers
  int fib[N];
  fib[0] = 0;
  fib[1] = 1;

  // Set the number of OpenMP threads (you can adjust this as needed)
  omp_set_num_threads(32);

  auto start = std::chrono::high_resolution_clock::now();
  // Parallelize the computation using OpenMP tasks
#pragma omp parallel
  {
    // Only one thread initializes the tasks
#pragma omp single
    for (int i = 2; i < N; i++)
    {
#pragma omp task depend(in: fib[i - 1], fib[i - 2]) depend(out: fib[i])
      {
        fib[i] = fib[i - 1] + fib[i - 2];
      }
    }
  }

  // Print all computed Fibonacci numbers until n
  std::chrono::duration<double> temps = std::chrono::high_resolution_clock::now() - start;
  printf("Fibonacci numbers: ");
  for (int i = 0; i < N; i++)
  {
    printf("%d ", fib[i]);
  }
  printf("\n");
  std::cout << "Temps: " << temps.count() << "s\n";

  return 0;
}
