/**
 * 
 * @file weight_quantizer_cuda.cu
 * @brief Weight quantizer CUDA kernel implementations.
 * 
 */


#include "cuda_fp16_util.h"
#include "cuda_math_util.h"
#include "io_iterator.h"
#include "rpu_cub.h"
#include "weight_quantizer_cuda.h"


namespace RPU {


#define RPU_WQ_KERNEL_LOOP(STOCH_IF, BODY)                                                         \
  int tid = blockDim.x * blockIdx.x + threadIdx.x;                                                 \
  int total_threads = blockDim.x * gridDim.x;                                                      \
  int size = size_in;                                                                              \
  int size_without_bias = quantize_last_column ? (size - d_size) : size;                               \
  const bool stoch_if = STOCH_IF;                                                                  \
                                                                                                   \
  curandState local_state;                                                                         \
  if (stoch_if && tid < size) {                                                                    \
    local_state = random_states[tid];                                                              \
  }                                                                                                \
                                                                                                   \
  for (int i_stride = 0; i_stride < size; i_stride += total_threads) {                             \
    int i = i_stride + tid;                                                                        \
    if (i < size_without_bias) {                                                                   \
      {                                                                                            \
        BODY;                                                                                      \
      }                                                                                            \
    } else if ((i < size) && (new_weights != weights)) {                                           \
      new_weights[i] = weights[i];                                                                     \
    }                                                                                              \
  }                                                                                                \
                                                                                                   \
  if (stoch_if && tid < size) {                                                                    \
    random_states[tid] = local_state;                                                              \
  }

template <typename T>
__global__ void kernelQuantize(
    int size_in,
    int d_size,
    const bool quantize_last_column,
    T *new_weights,
    T *weights,
    const T res_in,
    const bool sto_round,
    const T zero_point,
    const unsigned int levels,
    const T *wmax,
    curandState_t *random_states) {
  const T res = res_in;
  T amax = (wmax) ? (*wmax) : (T)1.0;
  amax = amax > (T)0.0 ? amax : (T)1.0;

  RPU_WQ_KERNEL_LOOP(
      sto_round,

      T value = weights[i] / amax;
      value /= res;

      if (stoch_if) {
        T stoch_value = curand_uniform(&local_state);
        value += stoch_value - (T)0.5;
      }

      if (levels == 0) {
        weights[i] = amax * res * (round(value + zero_point) - zero_point);
      }
      else {
        T quant_value = round(value + zero_point);
        quant_value = fmin(fmax(quant_value, - ((T)levels - 1)/2.), (((T)levels - 1)/2.));
        weights[i] = amax * res * (quant_value - zero_point);
      });
        
}

template <typename T>
WeightQuantizerCuda<T>::WeightQuantizerCuda(CudaContextPtr context, int x_size, int d_size)
    : context_(context), x_size_(x_size), d_size_(d_size), size_(x_size * d_size) {

}

template <typename T>
void WeightQuantizerCuda<T>::apply(T *weights, const WeightQuantizerParameter<T> &wqpar) {
  
    // int nthreads = context_->getNThreads();
    // int nblocks = context_->getNBlocks(size_, nthreads);
    // auto s = context_->getStream();

    // First, rescale the weights based on the maximum absolute value:
    // 1. Find the maximum absolute value of the weights if required
    T *amax = nullptr;
    if (wmpar.rel_to_actual_wmax) {
        if (!amaximizer_) {
        amaximizer_ = RPU::make_unique<Maximizer<T>>(
            context_, wmpar.quantize_last_column ? (size_ - d_size_) : size_, true);
        }
        amaximizer_->compute(weights, 1, false);
        amax = amaximizer_->getMaxValues();
    }

    // For now, only the implementation for the uniform quantization is provided (no stochastic rounding)
    switch (wqpar.quantizer_type) {
        case WeightQuantizerType::UniformSymmetric: {
            if (wqpar.resolution >0){

                T z = (T).0
                
                // call the kernel
                kernelQuantize<T><<<nblocks, nthreads, 0, s>>>(
                    size_, d_size_, 
                    wqpar.quantize_last_column, weights, weights, wqpar.resolution, wqpar.stochastic_round, 
                    z, wqpar.levels, amax, wmpar.stochastic_round ? context_->getRandomStates(nblocks * nthreads) : nullptr);
                
            }
            break;
        }
        case WeightQuantizerType::UniformAsymmetric: {
            if (wqpar.resolution >0){
                
                // call the kernel
                kernelQuantize<T><<<nblocks, nthreads, 0, s>>>(
                    size_, d_size_, 
                    wqpar.quantize_last_column, weights, weights, wqpar.resolution, wqpar.stochastic_round, 
                    wqpar.z, wqpar.levels, amax,wmpar.stochastic_round ? context_->getRandomStates(nblocks * nthreads) : nullptr);
            }
            break;
        }
        case WeightQuantizerType::Custom: {
            if (wqpar.quant_values.size() == 0){
                RPU_FATAL("Custom quantization requires quant_values to be set.");
            }
            // Quantize the weights
            RPU::math::custom_quantize(context_, weights, size_, wqpar.quant_values);

            break;
        }
        default:
            RPU_FATAL("Weight quantizer type not implemented.");
    }
}

template class WeightQuantizerCuda<float>;
#ifdef RPU_USE_DOUBLE
template class WeightQuantizerCuda<double>;
#endif
#ifdef RPU_USE_HALF
template class WeightQuantizerCuda<half>;
#endif

} // namespace RPU