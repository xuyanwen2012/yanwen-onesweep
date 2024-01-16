#include <iostream>

#include "common/helper_cuda.hpp"
#include "init.cuh"

namespace {

void init_input_keys_sync(unsigned int* u_input_keys, int n, int seed) {
  constexpr auto threads = 768;
  const auto blocks = (n + threads - 1) / threads;
  k_InitRandom<<<blocks, threads>>>(u_input_keys, n, seed);
  checkCudaErrors(cudaDeviceSynchronize());
}

}  // anonymous namespace

int main() {
  constexpr auto n = 1'000'000;
  constexpr auto seed = 114514;

  unsigned int* u_input_keys;
  unsigned int* u_input_keys_alt;
  checkCudaErrors(cudaMallocManaged(&u_input_keys, n * sizeof(unsigned int)));
  checkCudaErrors(
      cudaMallocManaged(&u_input_keys_alt, n * sizeof(unsigned int)));

  init_input_keys_sync(u_input_keys, n, seed);

  // peek the first 10 elements
  for (int i = 0; i < 10; ++i) {
    std::cout << u_input_keys[i] << '\n';
  }

  checkCudaErrors(cudaFree(u_input_keys));
  checkCudaErrors(cudaFree(u_input_keys_alt));
  return 0;
}