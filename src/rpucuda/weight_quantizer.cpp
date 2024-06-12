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
WeightQuantizer<T>::WeightQuantizer(int x_size, int d_size, const WeightQuantizerParameter<T> &wqpar)
    : WeightQuantizer(x_size, d_size) {
    populate(wqpar);
};

template <typename T>
void WeightQuantizer<T>::fit(const T *weights) {
    // The function is used to determine the best value for the relat_bound parameter
    // based on the weights values, to allow for at last quantization levels to be used

    // First we determine the maximum value in the weights array
    T amax = 0.0;
    for (int i = 0; i < size_; i++) {
        T a = (T)fabsf(weights[i]);
        amax = a > amax ? a : amax;
    }
    amax = amax > (T)0.0 ? amax : (T)1.0;

    // We want the last quantization levels to be used by, at least, 3% of the weights

    int total_weights = size_;
    float percentage = 0.03;
    int count = (int)(total_weights * percentage);

    // Order the weights in a descending order
    std::vector<T> sorted_weights(weights, weights + size_);
    std::sort(sorted_weights.begin(), sorted_weights.end(), std::greater<T>());
    T max_bound = sorted_weights[0];
    T min_bound = sorted_weights[0];

    // Loop thought the sorted weights until we reach the count value
    for (int i = 0; i < count; i++) {
        T a = sorted_weights[i];
        max_bound = a > max_bound ? a : max_bound;
        min_bound = a < min_bound ? a : min_bound;
    }

    T bound = (max_bound < fabsf(min_bound)) ? max_bound : fabsf(min_bound);
    T levels = (T)par_.levels;
    T resolution = (T)par_.resolution;

    // Determine the relat_bound value
    T relat_bound = bound / ((levels - 1.0) * resolution);
    par_.relat_bound = relat_bound< 0.9 ? relat_bound : 0.9;

};

template <typename T>
void WeightQuantizer<T>::apply(T *weights, RNG<T> &rng) {
    

    if (par_.resolution == 0.0 && par_.uniform_quant) {
        return;
    }
    if (par_.resolution > par_.bound){
        RPU_FATAL("Quantize value cannot be greater than bound");
    }
    // if (new_weights != weights) {
    // RPU::math::copy<T>(size_, weights, 1, new_weights, 1);
    // }


    // If quantization for the bias is disabled, save the bias values
    // in a buffer
    if(par_.quantize_last_column == false){
        saved_bias_.resize(d_size_);
        for (int j = 0; j < d_size_; j++) {
            saved_bias_[j] = weights[(j + 1) * x_size_ - 1];
        }
    }

    T bound = (T)par_.bound;
    if (par_.rel_to_actual_bound || (par_.relat_bound > 0.0)) {
        T amax = 0.0;
        PRAGMA_SIMD
        for (int i = 0; i < size_; i++) {
            if (par_.quantize_last_column && (i % x_size_) == x_size_ - 1) {
                continue;
            }
            T a = (T)fabsf(weights[i]);
            amax = a > amax ? a : amax;
        }
        amax = amax > (T)0.0 ? amax : (T)1.0;
        bound = amax;
        if (par_.relat_bound > 0.0) {
            bound *= (T)par_.relat_bound;
        }
    }

    // Check for the uniform_quant flag
    if (par_.uniform_quant == false){
        // Check if the quant_values vector is empty
        if (par_.quant_values.size() == 0){
            RPU_FATAL("Quant values are empty");
        }
        else{
            // Run the non uniform quantization function 
            // from the utility_functions.h file, based on the
            // quant_values vector and the bound value
            const std::vector<T> &quant_values = par_.quant_values;
            PRAGMA_SIMD
            for (int i = 0; i < size_; i++) {
                T w = weights[i];
                weights[i] = bound * getDiscretizedValueNonUniform(w/bound, quant_values, rng);
            }
        }
    }
    else {
        const bool stochastic_round = par_.stochastic_round;
        const T resolution = par_.resolution;
        const T levels = (T) par_.levels;
        // Run the uniform quantization function from the utility_functions.h file
        // based on the bound value and the stochastic_round flag
        if (levels == 0){
            PRAGMA_SIMD
            for (int i = 0; i < size_; i++) {
                T w = weights[i];
                weights[i]= bound*getDiscretizedValueRound(w/bound, resolution, stochastic_round, rng);
            }
        }
        else
        {
            PRAGMA_SIMD
            for (int i = 0; i < size_; i++) {
                T w = weights[i];
                weights[i] = bound * getDiscretizedValueCollapse(w/bound, resolution, stochastic_round, levels, rng);
            }
        }
    }

    if (par_.quantize_last_column == false){
        for (int j = 0; j < d_size_; j++) {
            weights[(j + 1) * x_size_ - 1] = saved_bias_[j];
        }
    }

};

template <typename T>
void WeightQuantizer<T>::populate(const WeightQuantizerParameter<T> &wqpar) {
    par_ = wqpar;
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