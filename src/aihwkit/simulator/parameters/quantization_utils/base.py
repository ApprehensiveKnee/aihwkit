# ///////////////////////////////////////////////////////////////////////////////////////////
# base class for calibrator to determine the amax element for applying quantization
# Inspo from TensorTR tool pythorch-quantization (Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES)
# ///////////////////////////////////////////////////////////////////////////////////////////



class Calibrator():
    """Abstract base class for calibrators
    
    Args: 
        nums_bits: Integer, number of bits for the quantization
        axis: Tuple, axis to quantize
        unsinged: Boolean, whether to use unsigned quantization

    """


    def __init__(self, num_bits, axis, unsigned):
        self._num_bits = num_bits
        self._axis = axis
        self._unsigned = unsigned


    def collect(self, x):
        """Collects the statistics for the calibrator
        
        Args:
            x: Tensor, input tensor to collect statistics from

        """
        raise NotImplementedError
    
    def reset(self):
        """ Resets the calibrator to initial state"""
        raise NotImplementedError

    def compute_amax(self, *argss, **kwargs):
        """Computes the amax value out of the collected data for the quantization
        
        Returns:
            amax: a tensor
        """

        raise NotImplementedError
    
    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join(
                '{}={}'.format(k, v)
                for k, v in self.__dict__.items()
            )
        )

