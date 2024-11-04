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
  UniformSymmetric, // zero-centered
  UniformAsymmetric, // zero-point value is != 0
  Custom // fixed quantization values specified in input
};

template <typename T>
struct WeightQuantizerParameter{

  T resolution = (T)0.0;
  T amax =(T) 1.0;
  T eps = (T) 0.0;
  T z = (T) 0.0;
  std::string method = "percentile";
  unsigned short levels = 0;
  bool quantize_last_column = false;
  bool rel_to_actual_wmax = false;
  WeightQuantizerType quantizer_type = WeightQuantizerType::UniformSymmetric;
  std::vector<T> quant_values = {};
  bool stochastic_round = false;
  bool debug = true;

  inline std::string getTypeName() const {
    switch (quantizer_type) {
    case WeightQuantizerType::UniformSymmetric:
      return "Uniform";
    case WeightQuantizerType::UniformAsymmetric:
      return "FixedValued";
    case WeightQuantizerType::Custom:
    default:
      return "Unknown";
    }
  }

  inline std::string getMethodName() const {
    return method;
  }

  void print() const {
    std::stringstream ss;
    printToStream(ss);
    std::cout << ss.str();
  };

  void printToStream(std::stringstream &ss) const {
    ss << "\t resolution:\t" << resolution << std::endl;
    ss << "\t levels: \t" << levels << std::endl;
    ss << "\t eps: \t" << eps << std::endl;
    //ss << "\t bound: \t" << bound << std::endl;
    ss << "\t z: \t" << z << std::endl;
    ss << "\t method: \t" << getMethodName() << std::endl;
    ss << "\t quantize_last_column: \t" << quantize_last_column << std::endl;
    ss << "\t stochastic_round: \t" << stochastic_round << std::endl;
    ss << "\t quantizer_type: \t" << getTypeName() << std::endl;
    if(quantizer_type == WeightQuantizerType::Custom){
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

    void dumpExtra(RPU::state_t &extra, const std::string prefix);
    void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict);

private:
    int x_size_ = 0;
    int d_size_ = 0;
    int size_ = 0;
    std::vector<T> saved_bias_;

};

}; // namespace RPU