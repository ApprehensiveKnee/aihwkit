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

//   T *tmp = nullptr;
//   StdFunctor<T> std_functor((T)x_size_, tmp);
//   RPU_CUB_NS_QUALIFIER TransformInputIterator<T, StdFunctor<T>, T *> std_input(tmp, std_functor);

//   RPU_CUB_NS_QUALIFIER DeviceReduce::Sum(
//       nullptr, temp_storage_bytes_, std_input, tmp, size_, context_->getStream());
//   dev_temp_storage_ = RPU::make_unique<CudaArray<char>>(context, temp_storage_bytes_);
}

template <typename T>
__global__ T WeightQuantizerCuda<T>::fit(const T *weights, const WeightQuantizerParameter<T> &wqpar, const T bound) {

    // The fit function is used to fine tune the redolution of the quantizer, so that up to a minimum
    // of (1 - eps) fraction of the weights are included in the FSR.

    if (wqpar.resolution != 0 || wqpar.eps == 0){
        return wqpar.resolution;
    }

    int total_weights = size_;
    T percentage = (float)wqpar.eps;
    int max_count = (int)(total_weights * percentage/2.);

    std::vector<T> sorted_weights(size_);
    PRAGMA_SIMD
    for (int i = 0; i < size_; i++) {
        sorted_weights[i] =  weights[i];
    }

    std::sort(sorted_weights.begin(), sorted_weights.end(), std::greater<T>());
    T max_bound = sorted_weights[0];
    T min_bound = sorted_weights[total_weights - 1];
    int max_index = 0;
    int min_index = total_weights - 1;


    // Loop thought the sorted weights until we reach the count value
    for (int i = 0; i < max_count; i++) {
        // For each iteration, move to the next element starting from the ends
        // of the sorted weights array
        max_index++;
        min_index--;
        max_bound = sorted_weights[max_index];
        min_bound = sorted_weights[min_index];
    }

    // Check which bound is closer to the zero value
    T limit = (fabs(min_bound) < fabs(max_bound)) ? max_bound : min_bound;
    int limit_index = (fabs(min_bound) < fabs(max_bound)) ? max_index : total_weights - min_index - 1;
    limit = fabs(limit);
    std::cout << "Limit value: " << limit << std::endl;
    std::cout << "Cutout percentage: " << (float)(total_weights - 2*limit_index)/(float)total_weights << std::endl;

    // Set the resolution value, so that the limit value is included in the FSR
    T levels = (T)wqpar.levels;
    return (T) (2/(levels-1))*(limit/bound);
}



template <typename T>
__global__ void WeightQuantizerCuda<T>::apply(T *weights, const WeightQuantizerParameter<T> &wqpar) {
  
    // int nthreads = context_->getNThreads();
    // int nblocks = context_->getNBlocks(size_, nthreads);
    auto s = context_->getStream();

    // For now, only the implementation for the uniform quantization is provided (no stochastic rounding)
    switch (wqpar.quantizer_type) {
        case WeightQuantizerType::Uniform: {
            if (wqpar.resolution > 0){
                // First, rescale the weights based on the maximum absolute value:
                // 1. Find the maximum absolute value of the weights
                if (amaximizer_ == nullptr){
                    amaximizer_ = RPU::make_unique<Maximizer<T>>(context_, size_);
                }
                amaximizer_->compute(weights, 1, true);
                T bound_value_;
                amaximizer_->copyMaxValuesToHost(&bound_value_);
                // 2. Rescale the weights
                RPU::math::elemscale(context_, weights, size_, (T)1.0 / bound_value_);

                (T) resolution = fit(weights, wqpar, bound_value_);
                // Quantize the weights
                RPU::math::uquantize(context_, weights, size_, resolution, wqpar.levels);

                // Rescale back the weights
                RPU::math::elemscale(context_, weights, size_, bound_value_);

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