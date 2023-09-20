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

// Recursive Fibonacci calculation using OpenMP tasks
int fib(int N)
{
  if (N <= 1)
    return N;

  int fib1, fib2;

#pragma omp task shared(fib1)
  fib1 = fib(N - 1);

#pragma omp task shared(fib2)
  fib2 = fib(N - 2);

#pragma omp taskwait

  return fib1 + fib2;
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

  // Set the number of OpenMP threads (you can change this as needed)
  omp_set_num_threads(8);

  int result;

  auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel
  {
#pragma omp single
    result = fib(N);
  }

  std::chrono::duration<double> temps = std::chrono::high_resolution_clock::now() - start;
  printf("Fibonacci(%d) = %d\n", N, result);
  std::cout << "Temps: " << temps.count() << "s\n";
  return 0;
}
