#include "consts.hpp"
#include "histogram.cuh"

// __global__ void k_GlobalHistogram(unsigned int* sort,
//                                   unsigned int* globalHistogram,
//                                   int size) {
//   __shared__ unsigned int s_globalHistFirst[RADIX];
//   __shared__ unsigned int s_globalHistSec[RADIX];
//   __shared__ unsigned int s_globalHistThird[RADIX];
//   __shared__ unsigned int s_globalHistFourth[RADIX];

//   // clear
//   for (auto i = THREAD_ID; i < RADIX; i += hist_threadsPerBlock) {
//     s_globalHistFirst[i] = 0;
//     s_globalHistSec[i] = 0;
//     s_globalHistThird[i] = 0;
//     s_globalHistFourth[i] = 0;
//   }
//   __syncthreads();

//   // Histogram
//   {
//     const int partitionEnd = G_HIST_PART_END;
//     for (auto i = THREAD_ID + G_HIST_PART_START; i < partitionEnd;
//          i += G_HIST_THREADS) {
//       const unsigned int key = sort[i];
//       atomicAdd(&s_globalHistFirst[key & radixMask], 1);
//       atomicAdd(&s_globalHistSec[key >> secondRadix & radixMask], 1);
//       atomicAdd(&s_globalHistThird[key >> thirdRadix & radixMask], 1);
//       atomicAdd(&s_globalHistFourth[key >> fourthRadix], 1);
//     }
//   }
//   __syncthreads();
// }