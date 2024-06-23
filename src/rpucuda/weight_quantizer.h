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

enum class WeightQuantizerType {
  Uniform,
  FixedValued,
};

template <typename T>
struct WeightQuantizerParameter{

  T resolution = (T)0.0;
  T bound =(T) 1.0;
  T relat_bound = (T)0.0;
  unsigned short levels = 0;
  bool rel_to_actual_bound = true;
  bool quantize_last_column = true;
  WeightQuantizerType quantizer_type = WeightQuantizerType::Uniform;
  std::vector<T> quant_values = {};
  bool stochastic_round = false;

  inline std::string getTypeName() const {
    switch (quantizer_type) {
    case WeightQuantizerType::Uniform:
      return "Uniform";
    case WeightQuantizerType::FixedValued:
      return "FixedValued";
    }
  }

  void print() const {
    std::stringstream ss;
    printToStream(ss);
    std::cout << ss.str();
  };

  void printToStream(std::stringstream &ss) const {
    ss << "\t resolution:\t" << resolution << std::endl;
    ss << "\t levels: \t" << levels << std::endl;
    //ss << "\t bound: \t" << bound << std::endl;
    //ss << "\t relat_bound: \t" << relat_bound << std::endl;
    //ss << "\t rel_to_actual_bound: \t" << rel_to_actual_bound << std::endl;
    ss << "\t quantize_last_column: \t" << quantize_last_column << std::endl;
    ss << "\t stochastic_round: \t" << stochastic_round << std::endl;
    if(quantizer_type == WeightQuantizerType::FixedValued){
      ss << "\t quantizer_type: \t" << getTypeName() << std::endl;
      ss << "\t quant_values: \t[";
      for (size_t i = 0; i < quant_values.size(); i++) {
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

    WeightQuantizer(const WeightQuantizer<T> &) = default;
    WeightQuantizer(WeightQuantizer<T> &&) = default;
    WeightQuantizer<T> &operator=(WeightQuantizer<T> &&other) = default;
    WeightQuantizer<T> &operator=(const WeightQuantizer<T> &other) = default;

    inline int getSize() const { return size_; };

    // Apply in-place quantization
    void apply(T *weights, const WeightQuantizerParameter<T> &wqpar ,RNG<T> &rng);
    void fit(const T *weights, WeightQuantizerParameter<T> &wqpar);

    void dumpExtra(RPU::state_t &extra, const std::string prefix);
    void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict);

private:
    int x_size_ = 0;
    int d_size_ = 0;
    int size_ = 0;
    std::vector<T> saved_bias_;

};

}; // namespace RPU