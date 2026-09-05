#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <sys/time.h>

inline void randomize_matrix(float *mat, int N) {
  struct timeval time {};
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);
  for (int i = 0; i < N; i++) {
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    mat[i] = tmp;
  }
}
