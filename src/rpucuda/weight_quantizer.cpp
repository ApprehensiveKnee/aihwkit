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
void WeightQuantizer<T>::apply(T *weights, RNG<T> &rng) {
    

    if (par_.quantize == 0.0 && par_.uniform_quant) {
        return;
    }
    if (par_.quantize > par_.bound){
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
    if (par_.rel_to_actual_bound){
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
        const float quantize = par_.quantize;
        // Run the uniform quantization function from the utility_functions.h file
        // based on the bound value and the stochastic_round flag
        PRAGMA_SIMD
        for (int i = 0; i < size_; i++) {
            T w = weights[i];
            weights[i] = bound * getDiscretizedValueRound(w/bound, quantize, stochastic_round, rng);
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