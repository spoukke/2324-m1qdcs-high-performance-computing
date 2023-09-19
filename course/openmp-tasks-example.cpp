#include <iostream>
#include "omp.h"
#include <vector>
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

int main()
{
#pragma omp parallel num_threads(2) 
  {
    int two, three, five;
#pragma omp single
    {
#pragma omp task depend(out:two) default(none) shared(two)
      two = computeTwo();
#pragma omp task depend(out:three) default(none) shared(three)
      three = computeThree();
#pragma omp task depend(in:two) depend(in:three) default(none) shared(two,three,five)
      five = two + three;
#pragma omp taskwait
      printf("%d\n", five);
    }
  }


  return 0;
}
