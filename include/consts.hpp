#pragma once

// =================================================================================================
// Notes:
//  1. We are ONLY sorting 32-bit integers
//  2. Cub uses 128 threads, 16 items per thread for histogram
//  3. For prefix sum, used only 4 blocks
//  4. ...
// =================================================================================================

// We are ONLY sorting 32-bit integers

// we consider each 8 bits as a part. => 4 parts (passes)
// radix = 2^8 = 256 possible values for binning.
constexpr auto radixPasses = 4;
constexpr auto radix = 256;            // 1 << Radix_Bits
constexpr auto radixMask = radix - 1;  // 0xFF
constexpr auto radixLog2 = 8;          // log2(radix), radix bits

// shift values
constexpr auto secondRadix = 8;
constexpr auto thirdRadix = 16;
constexpr auto fourthRadix = 24;
constexpr auto firstRadixStart = 0;
constexpr auto secondRadixStart = 256;
constexpr auto thirdRadixStart = 512;
constexpr auto fourthRadixStart = 768;

static_assert(secondRadixStart == 1 * radix, "secondRadixStart != 1 * radix");
static_assert(thirdRadixStart == 2 * radix, "thirdRadixStart != 2 * radix");
static_assert(fourthRadixStart == 3 * radix, "fourthRadixStart != 3 * radix");

// // warp size on NVIDIA GPUs is 32, on AMD GPUs is 64
// // const auto warpSize = 32;
// const auto warpLog2 = 5;  // log2(warpSize)

// // #define LANE threadIdx.x        // Lane of a thread
// // #define WARP_INDEX threadIdx.y  // Warp of a thread
// // #define THREAD_ID (LANE + (WARP_INDEX << LANE_LOG))

// // ----------------- For the upfront global histogram kernel
// -----------------

// // number of subgroups in a thread block/work group for executing histogram
// // kernel
// constexpr auto hist_warpsPerBlock = 4;
// constexpr auto hist_threadsPerBlock = hist_warpsPerBlock * warpSize;  // 128
// constexpr auto hist_itemsPerThread = 16;                              // 16
// constexpr auto hist_tileSize =
//     hist_threadsPerBlock * hist_itemsPerThread;  // 2048
// #define G_HIST_PART_SIZE hist_tileSize
// #define G_HIST_PART_START \
//   (blockIdx.x * G_HIST_PART_SIZE)  // Starting offset of a partition tile
// #define G_HIST_PART_END \
//   (blockIdx.x == gridDim.x - 1 ? size : (blockIdx.x + 1) * G_HIST_PART_SIZE)
