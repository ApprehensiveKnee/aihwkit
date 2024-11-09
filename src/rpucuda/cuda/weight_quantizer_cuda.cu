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
  int size = x_size * d_size;                                                                      \
  int size_without_bias = quantize_last_column ? size: (size - d_size);                            \
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
      new_weights[i] = weights[i];                                                                 \
    }                                                                                              \
  }                                                                                                \
                                                                                                   \
  if (stoch_if && tid < size) {                                                                    \
    random_states[tid] = local_state;                                                              \
  }


template <typename T>
__global__ void kernelQuantize(
    const int x_size,
    const int d_size,
    const bool amax_channelwise,
    const T * amax_values,
    const bool quantize_last_column,
    T *new_weights,
    T *weights,
    const T res_in,
    const bool sto_round,
    const T zero_point,
    const unsigned int levels,
    const T *wmax,
    curandState_t *random_states) {
  T amax = (wmax) ? (*wmax) : (T)1.0;
  amax = amax > (T)0.0 ? amax : (T)1.0;

  printf("inside kernelQuantize\n");

  RPU_WQ_KERNEL_LOOP(
      sto_round,

      // first determine the resolution value based on the element 
      // being processed
      int row_idx = i / x_size;
      T res = amax_channelwise ? (T)((2./(levels - 1.)) * amax_values[row_idx]) : res_in;
      printf("res: %f for row_idx: %d\n", res, row_idx);

      T value = weights[i] / amax;
      value /= res;

      if (stoch_if) {
        T stoch_value = curand_uniform(&local_state);
        value += stoch_value - (T)0.5;
      }

      if (levels == 0) {
        new_weights[i] = amax * res * (round(value + zero_point) - zero_point);
      }
      else {
        T quant_value = round(value + zero_point);
        quant_value = quant_value > (T)levels/2.0 ? (T)(levels-1.)/2. : quant_value;
        quant_value = quant_value < -(T)levels/2.0 ? -(T)(levels-1.)/2. : quant_value;
        new_weights[i] = amax * res * (quant_value - zero_point);
      });
        
}

template <typename T>
__global__ void kernelCustomQuantize(
    int size_in,
    int d_size,
    const bool quantize_last_column,
    T *new_weights,
    T *weights,
    const T* device_quant_values,
    int quant_values_size) {

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int total_threads = blockDim.x * gridDim.x;
  int size = size_in;
  int size_without_bias = quantize_last_column ? size : (size - d_size);

  for (int i_stride = 0; i_stride < size; i_stride += total_threads) {                             
    int i = i_stride + tid;                                                                        
    if (i < size_without_bias) {                                                                                                                                                         
      T value = weights[i];                                                                      
      T quant_value = (T)device_quant_values[0];                                                          
      for (int j = 0; j < quant_values_size; j++) {                                           
        // check the difference between the value and the quantization value
        if (fabs(value - device_quant_values[j]) < fabs(value - quant_value)) {                         
          quant_value = device_quant_values[j];                                                        
        }                                                                                  
      } 
      new_weights[i] = quant_value;                                                                                           
    } else if ((i < size) && (new_weights != weights)) {                                           
      new_weights[i] = weights[i];                                                                  
    }                                                                                              
  }                                                                                                  
}



template <typename T>
WeightQuantizerCuda<T>::WeightQuantizerCuda(CudaContextPtr context, int x_size, int d_size)
    : context_(context), x_size_(x_size), d_size_(d_size), size_(x_size * d_size) {

}

template <typename T>
void WeightQuantizerCuda<T>::apply(T *weights, const WeightQuantizerParameter<T> &wqpar) {

    std::cout << " inside apply " << std::endl;
    std::cout << " wqpar.resolution: " << wqpar.resolution << std::endl;
    std::cout << " wqpar.amax_channelwise: " << wqpar.amax_channelwise << std::endl;
    std::cout << " wqpar.quantizer_type: " << wqpar.getMethodName() << std::endl;

    if ((wqpar.resolution == 0.0 && wqpar.amax_channelwise == false &&
        (wqpar.quantizer_type == WeightQuantizerType::UniformSymmetric 
        || wqpar.quantizer_type == WeightQuantizerType::UniformAsymmetric))
        || wqpar.quantizer_type == WeightQuantizerType::None
        ){ 
        return;
    }
  
    int nthreads = context_->getNThreads();
    auto s = context_->getStream();
    int nblocks = context_->getNStrideBlocks(size_, nthreads);

    // The following portion of code won't affect the results
    // of the following quantization steps
    // ==================== DEPRECATED ====================
    // First, rescale the weights based on the maximum absolute value:
    // 1. Find the maximum absolute value of the weights if required
    T *amax = nullptr;
    if (wqpar.rel_to_actual_wmax) {
        if (!amaximizer_) {
        amaximizer_ = RPU::make_unique<Maximizer<T>>(
            context_, wqpar.quantize_last_column ? (size_ - d_size_) : size_, true);
        }
        amaximizer_->compute(weights, 1, false);
        amax = amaximizer_->getMaxValues();
    }
    // ==================== DEPRECATED ====================

    T* device_amax_values;

    if (wqpar.amax_channelwise) {
      if (wqpar.amax_values.size() != (size_t)d_size_){
        RPU_FATAL("amax_values size is not equal to d_size");
      }
      // move the amax_values to the device
      cudaMalloc(&device_amax_values, d_size_ * sizeof(T));
      cudaMemcpy(device_amax_values, wqpar.amax_values.data(), d_size_ * sizeof(T), cudaMemcpyHostToDevice);
    }

    std::cout << " before kernel call " << std::endl;

    // For now, only the implementation for the uniform quantization is provided (no stochastic rounding)
    switch (wqpar.quantizer_type) {
        case WeightQuantizerType::UniformSymmetric: {
            if (wqpar.resolution >0){
              T z = (T).0; 
                
              // call the kernel
              kernelQuantize<T><<<nblocks, nthreads, 0, s>>>(
                  x_size_, d_size_, wqpar.amax_channelwise, device_amax_values, wqpar.quantize_last_column, weights, weights, wqpar.resolution, wqpar.stochastic_round, 
                  z, wqpar.levels, amax, wqpar.stochastic_round ? context_->getRandomStates(nblocks * nthreads) : nullptr);
                
            }
            break;
        }
        case WeightQuantizerType::UniformAsymmetric: {
            if (wqpar.resolution >0){
                
              // call the kernel
              kernelQuantize<T><<<nblocks, nthreads, 0, s>>>(
                  x_size_, d_size_, wqpar.amax_channelwise, device_amax_values, wqpar.quantize_last_column, weights, weights, wqpar.resolution, wqpar.stochastic_round, 
                  wqpar.z, wqpar.levels, amax, wqpar.stochastic_round ? context_->getRandomStates(nblocks * nthreads) : nullptr);
            }
            break;
        }
        case WeightQuantizerType::Custom: {
            if (wqpar.quant_values.size() == 0){
                RPU_FATAL("Custom quantization requires quant_values to be set.");
            }
            
            // move the quant_values to the device
            T* device_quant_values;
            cudaMalloc(&device_quant_values, wqpar.quant_values.size() * sizeof(T));
            cudaMemcpy(device_quant_values, wqpar.quant_values.data(), wqpar.quant_values.size() * sizeof(T), cudaMemcpyHostToDevice);


            kernelCustomQuantize<T><<<nblocks, nthreads, 0, s>>>(
                size_, d_size_, wqpar.quantize_last_column, weights, weights, device_quant_values, wqpar.quant_values.size());


            cudaDeviceSynchronize();

            cudaFree(device_quant_values);

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