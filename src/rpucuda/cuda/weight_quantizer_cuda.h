/**
 * 
 * @file weight_quantizer_cuda.h
 * @brief Weight quantizer header for CUDA support.
 * 
 */

#pragma once

#include "cuda_util.h"
#include "maximizer.h"
#include "weight_quantizer.h"

namespace RPU {

template <typename T> class WeightQuantizerCuda {

public:
  explicit WeightQuantizerCuda(CudaContextPtr context, int x_size, int d_size);
  WeightQuantizerCuda(){};

  void apply(T *weights, const WeightQuantizerParameter<T> &wqpar);

  void dumpExtra(RPU::state_t &extra, const std::string prefix){};
  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict){};

private:
    CudaContextPtr context_ = nullptr;
    int x_size_ = 0;
    int d_size_ = 0;
    int size_ = 0;

    std::unique_ptr<Maximizer<T>> amaximizer_ = nullptr;
}

} // namespace RPU