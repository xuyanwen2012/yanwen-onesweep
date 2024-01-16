#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <numeric>

#include "OneSweep.cuh"
#include "init.cuh"

const int sizeExponent = 28;
const int size = 1 << sizeExponent;  // 256 MB

const int radix = 256;
const int radixPasses = 4;
const int partitionSize = 7680;
const int globalHistThreadblocks = 2048;
const int binningThreadblocks = size / partitionSize;

const int laneCount = 32;
const int globalHistWarps = 8;
const int digitBinWarps = 16;
dim3 globalHistDim(laneCount, globalHistWarps, 1);
dim3 digitBinDim(laneCount, digitBinWarps, 1);

int main() {
  unsigned int* sort;
  unsigned int* alt;
  unsigned int* index;
  unsigned int* globalHistogram;
  unsigned int* firstPassHistogram;
  unsigned int* secPassHistogram;
  unsigned int* thirdPassHistogram;
  unsigned int* fourthPassHistogram;

  cudaMallocManaged(&sort, size * sizeof(unsigned int));
  cudaMallocManaged(&alt, size * sizeof(unsigned int));
  cudaMallocManaged(&index, radixPasses * sizeof(unsigned int));
  cudaMallocManaged(&globalHistogram,
                    radix * radixPasses * sizeof(unsigned int));
  cudaMallocManaged(&firstPassHistogram,
                    binningThreadblocks * radix * sizeof(unsigned int));
  cudaMallocManaged(&secPassHistogram,
                    binningThreadblocks * radix * sizeof(unsigned int));
  cudaMallocManaged(&thirdPassHistogram,
                    binningThreadblocks * radix * sizeof(unsigned int));
  cudaMallocManaged(&fourthPassHistogram,
                    binningThreadblocks * radix * sizeof(unsigned int));

  {
    constexpr auto seed = 114514;
    constexpr auto threads = 768;
    constexpr auto blocks = size / threads;
    k_InitRandom<<<blocks, threads>>>(sort, size, seed);
    cudaDeviceSynchronize();
  }

  std::cout << "sorting...\n";

  {
    const auto result = std::is_sorted(sort, sort + size);
    std::cout << "before sorting: " << std::boolalpha << result << '\n';
  }

  k_GlobalHistogram<<<globalHistThreadblocks, globalHistDim>>>(
      sort, globalHistogram, size);

  k_DigitBinning<<<binningThreadblocks, digitBinDim>>>(
      globalHistogram, sort, alt, firstPassHistogram, index, size, 0);

  k_DigitBinning<<<binningThreadblocks, digitBinDim>>>(
      globalHistogram, alt, sort, secPassHistogram, index, size, 8);

  k_DigitBinning<<<binningThreadblocks, digitBinDim>>>(
      globalHistogram, sort, alt, thirdPassHistogram, index, size, 16);

  k_DigitBinning<<<binningThreadblocks, digitBinDim>>>(
      globalHistogram, alt, sort, fourthPassHistogram, index, size, 24);

  cudaDeviceSynchronize();

  std::cout << "done sorting, now checking\n";

  // peek 10 sort

  for (auto i = 0; i < 10; ++i) {
    std::cout << sort[i] << ' ';
  }

  // peek 10 alt

  std::cout << '\n';
  for (auto i = 0; i < 10; ++i) {
    std::cout << alt[i] << ' ';
  }
  std::cout << '\n';

  {
    const auto result = std::is_sorted(sort, sort + size);
    std::cout << "after sorting: " << std::boolalpha << result << '\n';
  }

  // free
  cudaFree(sort);
  cudaFree(alt);
  cudaFree(index);
  cudaFree(globalHistogram);
  cudaFree(firstPassHistogram);
  cudaFree(secPassHistogram);
  cudaFree(thirdPassHistogram);
  cudaFree(fourthPassHistogram);
}