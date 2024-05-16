/**
 * 
 * @file weight_quantizer.h
 * @brief Weight quantizer class.
 * 
 */

#pragma once

#include "rng.h"
#include <memory>

namespace RPU {



template <typename T> struct WeightQuantizerParameter{

  T quantize = 0.0;
  T bound = 1.0;
  bool rel_to_actual_bound = true;
  bool quantize_last_column = false;
  bool uniform_quant = false;
  std::vector<T> quant_values = {};
  bool stochastic_round = false;

  void print() const {
    std::stringstream ss;
    printToStream(ss);
    std::cout << ss.str();
  };

  void printToStream(std::stringstream &ss) const {
    ss << "\t quantize:\t" << quantize << std::endl;
    ss << "\t bound: \t" << bound << std::endl;
    ss << "\t rel_to_actual_bound: \t" << rel_to_actual_bound << std::endl;
    ss << "\t quantize_last_column: \t" << quantize_last_column << std::endl;
    ss << "\t stochastic_round: \t" << stochastic_round << std::endl;
    if(uniform_quant){
      ss << "\t uniform_quant: \t" << uniform_quant << std::endl;
      ss << "\t quant_values: \t[";
      for (int i = 0; i < quant_values.size(); i++) {
        ss << quant_values[i];
        if (i < quant_values.size() - 1) {
          ss << ",";
        }
      }
      ss << "]" << std::endl;
    };
  };

};

template <typename T> 
class WeightQuantizer {

public: 
    explicit WeightQuantizer(int x_size, int d_size);
    WeightQuantizer(){};

    void apply(T* new_weights, const T *weights, const WeightQuantizerParameter<T> &wqpar);

    void dumpExtra(RPU::state_t &extra, const std::string prefix);
    void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict);

private:
    int x_size_ = 0;
    int d_size_ = 0;
    int size_ = 0;
    std::vector<T> saved_bias_;
    RealWorldRNG<T> rw_rng_{0};
};

}; // namespace RPU