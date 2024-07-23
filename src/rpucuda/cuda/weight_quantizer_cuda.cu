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
void getBound(T *weights) {
    // Get the maximum absolute value of the weights
    if (amaximizer_ == nullptr){
        amaximizer_ = RPU::make_unique<Maximizer<T>>(context_, size_);
    }
    amaximizer_->compute(dev_weights->getData(), 1, true);
    amaximizer_->copyMaxValuesToHost(&bound_);
}

template <typename T>
T WeightQuantizerCuda<T>::fit(const *CudaArray<T> &dev_weights, const WeightQuantizerParameter<T> &wqpar) {

    // The fit function is used to fine tune the redolution of the quantizer, so that up to a minimum
    // of (1 - eps) fraction of the weights are included in the FSR.

    if (wqpar.resolution != 0 || wqpar.eps == 0){
        return wqpar.resolution;
    }

    T bound = bound_;

    int total_weights = size_;
    T percentage = (float)wqpar.eps;
    int max_count = (int)(total_weights * percentage/2.);

    // Move the weights to the host
    std::vector<T> sorted_weights(total_weights);
    dev_weights->copyTo(sorted_weights.data());
 
    std::cout << "sorted_weights" << std::endl;
    for (int i = 0; i < total_weights; i++){
        std::cout << sorted_weights[i] << " ";
    }
    std::cout << std::endl;

    // Sort the weights
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
    int limit_index_0 = max_index;
    int limit_index_1 = 0;
    if (fabs(min_bound) < fabs(max_bound)){
        for (int i = min_index; i < total_weights; i++){
            limit_index_1 = i;
            if (fabs(sorted_weights[i]) < fabs(limit)){
                break;
            }
        }
        limit_index_1 = total_weights - limit_index_1 - 1;
    } else {
        for (int i = max_index; i >= 0; i--){
            limit_index_1 = i;
            if (fabs(sorted_weights[i]) < fabs(limit)){
                break;
            }
        }
    }
    limit = fabs(limit);
    std::cout << "Limit value: " << limit << std::endl;
    std::cout << "Cutout percentage: " << (float)(total_weights - limit_index_0 - limit_index_1)/(float)total_weights << std::endl;

    // Set the resolution value, so that the limit value is included in the FSR
    T levels = (T)wqpar.levels;
    return (T) (2/(levels-1))*(limit/bound);
}



template <typename T>
void WeightQuantizerCuda<T>::apply(T *weights, const WeightQuantizerParameter<T> &wqpar, const T &resolution) {
  
    // int nthreads = context_->getNThreads();
    // int nblocks = context_->getNBlocks(size_, nthreads);
    //auto s = context_->getStream();

    // For now, only the implementation for the uniform quantization is provided (no stochastic rounding)
    switch (wqpar.quantizer_type) {
        case WeightQuantizerType::Uniform: {
            if (wqpar.resolution >0 || (wqpar.resolution == 0 && wqpar.eps > 0)){
                // First, rescale the weights based on the maximum absolute value:
                // 1. Find the maximum absolute value of the weights
                T bound_value = bound_;
                // 2. Rescale the weights
                RPU::math::elemscale(context_, weights, size_, (T)1.0 / bound_value);

                // Quantize the weights
                RPU::math::uquantize(context_, weights, size_, resolution, wqpar.levels);

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