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

template <typename T>
WeightQuantizerCuda<T>::WeightQuantizerCuda(CudaContextPtr context, int x_size, int d_size)
    : context_(context), x_size_(x_size), d_size_(d_size), size_(x_size * d_size) {

}

template <typename T>
void WeightQuantizerCuda<T>::apply(T *weights, const WeightQuantizerParameter<T> &wqpar) {
  
    // int nthreads = context_->getNThreads();
    // int nblocks = context_->getNBlocks(size_, nthreads);
    // auto s = context_->getStream();

    // For now, only the implementation for the uniform quantization is provided (no stochastic rounding)
    switch (wqpar.quantizer_type) {
        case WeightQuantizerType::Uniform: {
            if (wqpar.resolution >0){
                // First, rescale the weights based on the maximum absolute value:
                // 1. Find the maximum absolute value of the weights
                T bound_value = bound_;
                // 2. Rescale the weights
                RPU::math::elemscale(context_, weights, size_, (T)1.0 / bound_value);

                // Quantize the weights
                RPU::math::uquantize(context_, weights, size_, (T)wqpar.resolution, wqpar.levels);

                // Rescale back the weights
                RPU::math::elemscale(context_, weights, size_, bound_value);

            }
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