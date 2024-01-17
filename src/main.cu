#include <array>
#include <iostream>

#include "common/helper_cuda.hpp"
#include "init.cuh"

// **************************************************************
// * Constants                                                  *
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

// **************************************************************
// * Radix Data
// **************************************************************

struct RadixData {
  RadixData() = default;
  // explicit RadixData(const size_t size) : size(size) { allocate(); }

  ~RadixData() {
    checkCudaErrors(cudaFree(u_input_keys));
    checkCudaErrors(cudaFree(u_input_keys_alt));
    checkCudaErrors(cudaFree(globalHistogram));
    for (auto& histogram : passHistograms) {
      checkCudaErrors(cudaFree(histogram));
    }

    // info that the memory is freed
    std::cout << "Memory Freed\n";
  }

  void allocate() {
    checkCudaErrors(cudaMallocManaged(&u_input_keys, size * sizeof(uint32_t)));
    checkCudaErrors(
        cudaMallocManaged(&u_input_keys_alt, size * sizeof(uint32_t)));

    checkCudaErrors(cudaMallocManaged(&globalHistogram,
                                      radix * radixPasses * sizeof(uint32_t)));

    for (auto& histogram : passHistograms) {
      checkCudaErrors(cudaMallocManaged(
          &histogram, radix * binningThreadblocks(size) * sizeof(uint32_t)));
    }
    checkCudaErrors(cudaDeviceSynchronize());
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
  std::array<uint32_t*, radixPasses> passHistograms;

  // get the histogram for the pass
  uint32_t* getHistogram(const int pass) const { return passHistograms[pass]; }

  void dispatchHistogramKernel() {
    k_GlobalHistogram<<<globalHistThreadblocks, globalHistDim>>>(
        sort, globalHistogram, size);
  }

  void dispatchDigitBinKernel(const int pass) {
    k_DigitBin<<<binningThreadblocks(size), digitBinDim>>>(
        sort, alt, getHistogram(pass), index, size, pass);
  }
};

namespace {

void init_input_keys_sync(uint32_t* u_input_keys, int n, int seed) {
  constexpr auto threads = 768;
  const auto blocks = (n + threads - 1) / threads;
  k_InitRandom<<<blocks, threads>>>(u_input_keys, n, seed);
  checkCudaErrors(cudaDeviceSynchronize());
}

}  // anonymous namespace

int main(const int argc, const char** argv) {
  // problem size
  int size = 1'000'000;

  if (argc > 1) {
    size = std::atoi(argv[1]);
  }

  // check size is smaller than 2^24 (16M) and bigger than 2^8 (256)
  if (size < 256 || size > 16'777'216) {
    std::cerr << "size must be between 256 and 16'777'216\n";
    return 1;
  }

  std::array<RadixData, 2> data{};
  data[0].allocate();
  data[1].allocate();

  constexpr std::array<int, 2> seeds{114514, 1919810};

  init_input_keys_sync(data[0].u_input_keys, size, seeds[0]);
  init_input_keys_sync(data[1].u_input_keys, size, seeds[1]);

  data[0].dispatchHistogramKernel();
  data[0].dispatchDigitBinKernel(0);
  data[0].dispatchDigitBinKernel(1);
  data[0].dispatchDigitBinKernel(2);
  data[0].dispatchDigitBinKernel(3);

  checkCudaErrors(cudaDeviceSynchronize());

  auto result = std::is_sorted(data[0].sort, data[0].sort + size);
  std::cout << "is_sorted: " << std::boolalpha << result << '\n';

  return 0;
}