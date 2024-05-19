/**
 * 
 * @file weight_quantizer.h
 * @brief Weight quantizer class.
 * 
 */

#pragma once

#include "rng.h"
#include <memory>
#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace RPU {



struct WeightQuantizerParameter{

  float quantize = 0.0;
  float bound = 1.0;
  bool rel_to_actual_bound = true;
  bool quantize_last_column = true;
  bool uniform_quant = true;
  std::vector<float> quant_values = {};
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

  void copy_from(const py::object& obj) {
        quantize = obj.attr("quantize").cast<float>();
        bound = obj.attr("bound").cast<float>();
        rel_to_actual_bound = obj.attr("rel_to_actual_bound").cast<bool>();
        quantize_last_column = obj.attr("quantize_last_column").cast<bool>();
        uniform_quant = obj.attr("uniform_quant").cast<bool>();
        quant_values = obj.attr("quant_values").cast<std::vector<float>>();
        stochastic_round = obj.attr("stochastic_round").cast<bool>();
    }

};

extern const WeightQuantizerParameter default_weight_quantizer_parameter;

template <typename T> 
class WeightQuantizer {

public: 
    explicit WeightQuantizer(int x_size, int d_size);
    WeightQuantizer(){};

    // Apply in-place quantization
    void apply(T *weights, const WeightQuantizerParameter &wqpar);
    void apply(T **weights, const WeightQuantizerParameter &wqppar);

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