/**
 * 
 * @file weight_quantizer.cpp
 * @brief Weight quantizer class.
 * 
 */


#include "weight_quantizer.h"
#include "math_util.h"
#include "utility_functions.h"


namespace RPU {

template <typename T>
WeightQuantizer<T>::WeightQuantizer(int x_size, int d_size) 
    : x_size_(x_size), d_size_(d_size), size_(d_size * x_size) {}


template <typename T>
void WeightQuantizer<T>::apply(T *weights, const WeightQuantizerParameter<T> &wqpar,RNG<T> &rng) {

    if ((wqpar.resolution == 0.0 && wqpar.amax_channelwise == false &&
        (wqpar.quantizer_type == WeightQuantizerType::UniformSymmetric 
        || wqpar.quantizer_type == WeightQuantizerType::UniformAsymmetric))
        || wqpar.quantizer_type == WeightQuantizerType::None
        ){ 
        return;
    }

    // If quantization for the bias is disabled, save the bias values
    // in a buffer
    if(wqpar.quantize_last_column == false){
        saved_bias_.resize(d_size_);
        for (int j = 0; j < d_size_; j++) {
            saved_bias_[j] = weights[(j + 1) * x_size_ - 1];
        }
    }

    // this portion of code is not used - deprecated
    // =================================================================

    T amax = (T)wqpar.amax;
    // amax represents int_range/float_range [(2**bits/(beta-alpha)]
    if (wqpar.rel_to_actual_wmax && wqpar.amax_channelwise == false) {
        // compute the max absolute value for the tile
        T amax = 0.0;
        PRAGMA_SIMD
        for (int i = 0; i < size_; i++) {
            if (wqpar.quantize_last_column && (i % x_size_) == x_size_ - 1) {
                continue;
            }
            T a = (T)fabsf(weights[i]);
            amax = a > amax ? a : amax;
        }
        amax = amax > (T)0.0 ? amax : (T)1.0;
    }
    // =================================================================

    const bool stochastic_round = wqpar.stochastic_round;
    const unsigned int levels = wqpar.levels;
    const T z = wqpar.z;
    const T resolution = wqpar.resolution;
    // define resolution as a vector, whose values are taken from the amax_values vector
    std::vector<T> resolutions;

    
    if (wqpar.amax_channelwise){

        resolutions.resize(d_size_);
        // check the size of the amax_values vector
        if (wqpar.amax_values.size() != d_size_){
            RPU_FATAL("amax_values size is not equal to d_size");
        }
        for (int i = 0; i < d_size_; i++) {
            resolutions[i] = (T)((2./(levels-1.)) * wqpar.amax_values[i]);
        }
    }
    
    // Check for the quantizer_type 
    switch (wqpar.quantizer_type){
        case WeightQuantizerType::UniformSymmetric:
            // Run the uniform quantization function from the utility_functions.h file
            // based on the bound value and the stochastic_round flag
            
            if (levels == 0){
                PRAGMA_SIMD
                for (int i = 0; i < size_; i++) {
                    T w = weights[i];
                    T temp = wqpar.amax_channelwise ? resolutions[i / x_size_] : resolution;
                    weights[i]= amax*getDiscretizedValueRound(w/amax, temp, stochastic_round, rng);
                }
            }
            else
            {
                PRAGMA_SIMD
                for (int i = 0; i < size_; i++) {
                    T w = weights[i];
                    T temp = wqpar.amax_channelwise ? resolutions[i / x_size_] : resolution;
                    weights[i] = amax * getDiscretizedValueClip(w/amax, temp, (T)0.0 , stochastic_round, levels, rng);
                }
            }
            break;
        case WeightQuantizerType::UniformAsymmetric:

            if(z == 0.0){
                RPU_FATAL("zero-point value is set to 0.0 for asymmetric quantization");
            }

            if (levels == 0){
                PRAGMA_SIMD
                for (int i = 0; i < size_; i++) {
                    T w = weights[i];
                    T temp = wqpar.amax_channelwise ? resolutions[i / x_size_] : resolution;
                    weights[i]= amax*getDiscretizedValueRound(w/amax, temp, z, stochastic_round, rng);
                }
            }
            else
            {
                PRAGMA_SIMD
                for (int i = 0; i < size_; i++) {
                    T w = weights[i];
                    T temp = wqpar.amax_channelwise ? resolutions[i / x_size_] : resolution;
                    weights[i] = amax * getDiscretizedValueClip(w/amax, temp, z, stochastic_round, levels, rng);
                }
            }
            break;
        case WeightQuantizerType::Custom:
            // Check if the quant_values vector is empty
            if (wqpar.quant_values.size() == 0){
                RPU_FATAL("Quant values are empty");
            }
            else{
                // Run the non uniform quantization function 
                // from the utility_functions.h file, based on the
                // quant_values vector and the bound value
                const std::vector<T> &quant_values = wqpar.quant_values;
                PRAGMA_SIMD
                for (int i = 0; i < size_; i++) {
                    T w = weights[i];
                    weights[i] = amax * getDiscretizedValueNonUniform(w/amax, quant_values, rng);
                }
            }
            break;
        default:
            RPU_FATAL("Unknown quantizer type");
    }

    if (wqpar.quantize_last_column == false){
        for (int j = 0; j < d_size_; j++) {
            weights[(j + 1) * x_size_ - 1] = saved_bias_[j];
        }
    }

};



template <typename T>
void WeightQuantizer<T>::dumpExtra(RPU::state_t &extra, const std::string prefix) {
    RPU::state_t state;
    
    RPU::insert(state, "saved_bias", saved_bias_);
    RPU::insertWithPrefix(extra, state, prefix);
};

template <typename T>
void WeightQuantizer<T>::loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict) {
    
    auto state = RPU::selectWithPrefix(extra, prefix);
    RPU::load(state, "saved_bias", saved_bias_, strict);
};

template class WeightQuantizer<float>;
#ifdef RPU_USE_DOUBLE
template class WeightQuantizer<double>;
#endif
#ifdef RPU_USE_FP16
template class WeightQuantizer<half_t>;
#endif

}; // namespace RPU