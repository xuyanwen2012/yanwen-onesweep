#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cub/cub.cuh>
#include <iostream>
#include <numeric>

#include "OneSweep.cuh"
#include "init.cuh"

// **************************************************************
// * Constants                                                 *
// **************************************************************

// these two are based on "sorting 32-bit uint32_tegers"
constexpr int radix = 256;
constexpr int radixPasses = 4;

// Warp size is 32
constexpr int laneCount = 32;

// How many warps intended to use for each kernel
constexpr int globalHistWarps = 8;
constexpr int digitBinWarps = 16;

// how many blocks for histogram?
constexpr int globalHistThreadblocks = 2048;

// tile size for digit binning, looks like 512 threads was used, 15 items per
// block. 15 x 32 x 16
constexpr int partitionSize = 7680;

// how many blocks for digit binning?
constexpr auto binningThreadblocks(const int size) {
  return (size + partitionSize - 1) / partitionSize;
}

// 8x32 = 256 threads per block
const dim3 globalHistDim(laneCount, globalHistWarps, 1);

// 16x32 = 512 threads per block
const dim3 digitBinDim(laneCount, digitBinWarps, 1);

struct RadixData {
  explicit RadixData(const size_t size) : size(size) {
    cudaMallocManaged(&u_input_keys, size * sizeof(uint32_t));
    cudaMallocManaged(&u_input_keys_alt, size * sizeof(uint32_t));

    cudaMallocManaged(&globalHistogram, radix * radixPasses * sizeof(uint32_t));

    for (auto& histogram : passHistograms) {
      cudaMallocManaged(&histogram,
                        radix * binningThreadblocks(size) * sizeof(uint32_t));
    }
  }

  size_t size;

  // the data (size n)
  uint32_t* u_input_keys;
  uint32_t* u_input_keys_alt;

  // temp storages for global histogram (size radix * radixPasses)
  // radix * radixPasses = 1024
  uint32_t* globalHistogram;

  // temp storages for digit binning (size radix * binningThreadblocks)
  // radix * binningThreadblocks(x) = ~radix * 130 = 33333? each
  // uint32_t* firstPassHistogram;
  // uint32_t* secPassHistogram;
  // uint32_t* thirdPassHistogram;
  // uint32_t* fourthPassHistogram;

  std::array<uint32_t*, radixPasses> passHistograms;

  enum class Pass : uint32_t {
    First = 0,
    Second = 1,
    Third = 2,
    Fourth = 3,
  };

  // get the histogram for the pass
  uint32_t* getHistogram(const Pass pass) const {
    return passHistograms[static_cast<uint32_t>(pass)];
  }
};

void run_demo() {
  uint32_t* sort;
  uint32_t* alt;
  uint32_t* index;
  uint32_t* globalHistogram;
  uint32_t* firstPassHistogram;
  uint32_t* secPassHistogram;
  uint32_t* thirdPassHistogram;
  uint32_t* fourthPassHistogram;

  cudaMallocManaged(&sort, size * sizeof(uint32_t));
  cudaMallocManaged(&alt, size * sizeof(uint32_t));
  cudaMallocManaged(&index, radixPasses * sizeof(uint32_t));
  cudaMallocManaged(&globalHistogram, radix * radixPasses * sizeof(uint32_t));
  cudaMallocManaged(&firstPassHistogram,
                    binningThreadblocks * radix * sizeof(uint32_t));
  cudaMallocManaged(&secPassHistogram,
                    binningThreadblocks * radix * sizeof(uint32_t));
  cudaMallocManaged(&thirdPassHistogram,
                    binningThreadblocks * radix * sizeof(uint32_t));
  cudaMallocManaged(&fourthPassHistogram,
                    binningThreadblocks * radix * sizeof(uint32_t));

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

  k_DigitBinning<<<binningThreadblocks, digitBinDim, 0>>>(
      globalHistogram, sort, alt, firstPassHistogram, index, size, 0);

  k_DigitBinning<<<binningThreadblocks, digitBinDim, 0>>>(
      globalHistogram, alt, sort, secPassHistogram, index, size, 8);

  k_DigitBinning<<<binningThreadblocks, digitBinDim, 0>>>(
      globalHistogram, sort, alt, thirdPassHistogram, index, size, 16);

  k_DigitBinning<<<binningThreadblocks, digitBinDim, 0>>>(
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

void run_cub() {
  uint32_t* sort;
  uint32_t* alt;

  cudaMallocManaged(&sort, size * sizeof(uint32_t));
  cudaMallocManaged(&alt, size * sizeof(uint32_t));

  {
    constexpr auto seed = 114514;
    constexpr auto threads = 768;
    constexpr auto blocks = size / threads;
    k_InitRandom<<<blocks, threads>>>(sort, size, seed);
    cudaDeviceSynchronize();
  }

  void* temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceRadixSort::SortKeys(
      temp_storage, temp_storage_bytes, sort, alt, size);

  cudaMalloc(&temp_storage, temp_storage_bytes);

  cub::DeviceRadixSort::SortKeys(
      temp_storage, temp_storage_bytes, sort, alt, size);

  cudaDeviceSynchronize();

  // no need to check cub

  // free
  cudaFree(sort);
  cudaFree(alt);
  cudaFree(temp_storage);
}

__global__ void empty() {}

int main(const int argc, const char* argv[]) {
  // problem size
  int size = 1'000'000;

  if (argc > 1) {
    size = std::atoi(argv[1]);
  }
  // check size is smaller than 2^24 (16M) and bigger than 2^8 (256)
  if (size < 256 || size > 16'777'216) {
    spdlog::error("size must be between 256 and 16'777'216");
    return 1;
  }

  empty<<<1, 1>>>();
  cudaDeviceSynchronize();

  run_demo();
  run_cub();

  return 0;
}