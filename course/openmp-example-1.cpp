#include <iostream>
#include "omp.h"
#include "unistd.h"

int computeTwo()
{
  sleep(2);
  return 2;
}

int computeThree()
{
  sleep(3);
  return 3;
}

int computeFour()
{
  sleep(4);
  return 4;
}

int main()
{
  int two, three, four, nine;
#pragma omp parallel default(none) num_threads(3) shared(two, three, four, nine)
  {
    int thid = omp_get_thread_num();
    int numth = omp_get_num_threads();
    printf("hello from thread %d/%d\n", thid, numth);
    if (thid == 0) {
      two = computeTwo();
    }
    if (thid == 1) {
      three = computeThree();
    }
    if (thid == 2) {
      four = computeFour();
    }

#pragma omp barrier
#pragma omp single // run by a single thread
    {
      printf("thread %d running single\n", thid);
      nine = two + three + four;
    }
    printf("%d+%d+%d = %d\n", two, three, four, nine);
  } // implicit barrier
  return 0;
}
