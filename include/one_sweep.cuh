#pragma once

__global__ void k_DigitBinning(unsigned int* globalHistogram,
                               unsigned int* sort,
                               unsigned int* alt,
                               volatile unsigned int* passHistogram,
                               unsigned int* index,
                               int size,
                               unsigned int radixShift);
