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



template <typename T>
struct WeightQuantizerParameter{

  T resolution = (T)0.0;
  T bound =(T) 1.0;
  T relat_bound = (T)0.0;
  unsigned short levels = 0;
  bool rel_to_actual_bound = true;
  bool quantize_last_column = true;
  bool uniform_quant = true;
  std::vector<T> quant_values = {};
  bool stochastic_round = false;

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
    if(uniform_quant){
      ss << "\t uniform_quant: \t" << uniform_quant << std::endl;
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

  void copyFrom(const py::object& obj) {
        resolution = obj.attr("resolution").cast<T>();
        levels = obj.attr("levels").cast<unsigned short>();
        //bound = obj.attr("bound").cast<T>();
        //relat_bound = obj.attr("relat_bound").cast<T>();
        //rel_to_actual_bound = obj.attr("rel_to_actual_bound").cast<bool>();
        quantize_last_column = obj.attr("quantize_last_column").cast<bool>();
        uniform_quant = obj.attr("uniform_quant").cast<bool>();
        quant_values = obj.attr("quant_values").cast<std::vector<T>>();
        stochastic_round = obj.attr("stochastic_round").cast<bool>();
    }

};

template <typename T> 
class WeightQuantizer {

public: 
    explicit WeightQuantizer(int x_size, int d_size);
    explicit WeightQuantizer(int x_size, int d_size, const WeightQuantizerParameter<T> &wqpar);
    WeightQuantizer(){};

    WeightQuantizer(const WeightQuantizer<T> &) = default;
    WeightQuantizer(WeightQuantizer<T> &&) = default;
    WeightQuantizer<T> &operator=(WeightQuantizer<T> &&other) = default;
    WeightQuantizer<T> &operator=(const WeightQuantizer<T> &other) = default;

    inline const WeightQuantizerParameter<T> &getPar() const { return par_; }
    inline int getSize() const { return size_; };

    // Apply in-place quantization
    void apply(T *weights, RNG<T> &rng);
    void fit(const T *weights);

    void dumpExtra(RPU::state_t &extra, const std::string prefix);
    void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict);

private:
    int x_size_ = 0;
    int d_size_ = 0;
    int size_ = 0;
    std::vector<T> saved_bias_;
    WeightQuantizerParameter<T> par_;


    void populate(const WeightQuantizerParameter<T> &wqpar);
};

}; // namespace RPU