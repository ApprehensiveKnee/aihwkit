# -*- coding: utf-8 -*-

# ///////////////////////////////////////////////////////////////////////////////////////////
# histogram based calibrators
# Inspo from TensorTR tool pythorch-quantization (Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES)
# ///////////////////////////////////////////////////////////////////////////////////////////

from collections import Counter
import numpy as np
from scipy.stats import entropy

import torch
from aihwkit.simulator.parameters.quantization_utils.base import Calibrator



class MaxCalibrator(Calibrator):
    """ Max calibrator, tarcks the maximum value globally
    
    
    Args:
        num_bits: An integer. Number of bits for the quantization.
        axis: A tuple, the axis on which to perform quantization.
        unsigned: A boolean, whether to use unsigned quantization.

    Readonly Properties:
        amaxs: a list of amax values
    """

    def __init__(self, num_bits, axis, unsigned, track_amax=False):
        super(MaxCalibrator, self).__init__(num_bits, axis, unsigned)
        self._track_amax = track_amax
        if self._track_amax:
            self._amaxs = []
        self._calib_amax = None

    @property
    def amaxs(self):
        return self._amaxs
    
    def collect(self, x):
        """ Tracks the absolute max for all tensors

        Args:
            x: A tensor
        
        Raises:
            RuntimeError: it amax shape changes
        
        """

        if torch.min(x) < 0.:
            print("Calibrator encountered negative values. It shouldn't happen after ReLU. Make sure this is the right tensor to calibrate.")
            x = x.abs()

        # Swap axis to reduce.
        axis = self._axis if isinstance(self._axis, (list, tuple)) else [self._axis]
        # Handle negative axis.
        axis = [x.dim() + i if isinstance(i, int) and i < 0 else i for i in axis]
        reduce_axis = []
        for i in range(x.dim()):
            if not i in axis:
                reduce_axis.append(i)
        local_amax = reduce_amax(x, axis=reduce_axis).detach()
        if self._calib_amax is None:
            self._calib_amax = local_amax
        else:
            if local_amax.shape != self._calib_amax.shape:
                raise RuntimeError("amax shape changed!")
            self._calib_amax.copy_(torch.max(self._calib_amax, local_amax).data)

        if self._track_amax:
            self._amaxs.append(local_amax.cpu().numpy())


    def reset(self):
        """Reset the collected absolute max"""
        self._calib_amax = None

    def compute_amax(self):
        """Return the absolute max of all tensors collected"""
        return self._calib_amax 
    
    def __str__(self):
        s = "MaxCalibrator("
        s += "track_amax={_track_amax}"
        s += ")"
        return s.format(**self.__dict__)
    

def reduce_amax(input, axis=None, keepdims=True):
    """Compute the absolute maximum value of a tensor.

    Reduces input_tensor along the dimensions given in axis. Unless keepdims is true,
    the rank of the tensor is reduced by 1 for each entry in axis. If keepdims is true,
    the reduced dimensions are retained with length 1.

    .. note::
        Gradient computeation is disabled as this function is never meant learning reduces amax

    Args:
        input: Input tensor
        axis: The dimensions to reduce. None or int or tuple of ints. If None (the default),
            reduces all dimensions. Must be in the range [-rank(input_tensor), rank(input_tensor)).
        keepdims: A boolean. If true, retains reduced dimensions with length 1. Default True

    Returns:
        The reduced tensor.

    Raises:
        ValueError: Any axis which doesn't make sense or is not supported
        ValueError: If unknown granularity is passed in.
    """
    with torch.no_grad():
        output = input.abs()
        if axis is None:
            output = torch.max(output)
        else:
            if isinstance(axis, int):
                output, _ = torch.max(output, dim=axis, keepdim=keepdims)
            else:
                if isinstance(axis, tuple) and len(axis) > input.dim():
                    raise ValueError("Cannot reduce more axes than tensor's dim.")
                for i in axis:
                    output, _ = torch.max(output, dim=i, keepdim=True)
                if not keepdims or output.numel() == 1:
                    output.squeeze_()
        return output


class HistogramCalibrator(Calibrator):
    """Unified histogram calibrator

    Histogram will be only collected once. compute_amax() performs entropy, percentile, or mse
        calibration based on arguments

    Args:
        num_bits: An integer. Number of bits of quantization.
        axis: A tuple. see QuantDescriptor.
        unsigned: A boolean. using unsigned quantization.
        num_bins: An integer. Number of histograms bins. Default 2048.
        skip_zeros: A boolean. If True, skips zeros when collecting data for histogram. Default False.
        torch_hist: A boolean. If True, collect histogram by torch.histc instead of np.histogram. If input tensor
            is on GPU, histc will also be running on GPU. Default True.
    """
    def __init__(self, num_bits, axis, unsigned, num_bins=2048, skip_zeros=False, torch_hist=True):
        super(HistogramCalibrator, self).__init__(num_bits, axis, unsigned)
        self._num_bins = num_bins
        self._skip_zeros = skip_zeros

        self._calib_bin_edges = None
        self._calib_hist = None

        self._torch_hist = torch_hist

        if axis is not None:
            raise NotImplementedError("Calibrator histogram collection only supports per tensor scaling")
        
    def collect(self, x):
        """Collects histogram (minimization happens on ReLU output)"""
        if torch.min(x) < 0.:
            print("Calibrator encountered negative values. It shouldn't happen after ReLU. Make sure this is the right tensor to calibrate.")
            x = x.abs()

        x = x.float()

        if not self._torch_hist:
            x_np = x.cpu().detach().numpy()

            if self._skip_zeros:
                x_np = x_np[np.where(x_np != 0)]

            if self._calib_bin_edges is None and self._calib_hist is None:
                self._calib_hist, self._calib_bin_edges = np.histogram(x_np, bins=self._num_bins)

            else:
                temp_amax = np.max(x_np)
                if temp_amax > self._calib_bin_edges[-1]:
                    # increase the number of bins
                    width = self._calib_bin_edges[1] - self._calib_bin_edges[0]
                    
                    new_bin_edges = np.arange(self._calib_bin_edges[-1] + width, temp_amax + width, width)
                    self._calib_bin_edges = np.hstack((self._calib_bin_edges, new_bin_edges))
                hist, self._cali_bin_edges = np.histogram(x_np, bins= self._calib_bin_edges)
                hist[:len(self._calib_hist)] += self._calib_hist
                self._calib_hist = hist
        else:
            # This branch of code is designed to match numpy version as close as possible
            with torch.no_grad():
                if self._skip_zeros:
                    x = x[torch.where(x != 0)]

                # Because we collect histogram on absolute value, setting min=0 simplifying the rare case where
                # minimum value is not exactly 0 and first batch collected has larger min value than later batches
                x_max = x.max()
                if self._calib_bin_edges is None and self._calib_hist is None:
                    self._calib_hist = torch.histc(x, bins=self._num_bins, min=0, max=x_max)
                    self._calib_bin_edges = torch.linspace(0, x_max, self._num_bins + 1)
                else:
                    if x_max > self._calib_bin_edges[-1]:
                        width = self._calib_bin_edges[1] - self._calib_bin_edges[0]
                        self._num_bins = int((x_max / width).ceil().item())
                        self._calib_bin_edges = torch.arange(0, x_max + width, width, device=x.device)

                    hist = torch.histc(x, bins=self._num_bins, min=0, max=self._calib_bin_edges[-1])
                    hist[:self._calib_hist.numel()] += self._calib_hist
                    self._calib_hist = hist

    def reset(self):
        """Reset the collected histogram"""
        self._calib_hist = None
        self._calib_bin_edges = None

    def compute_amax(self, method: str, *, stride: int = 1, start_bin: int = 128, percentile: float = 99.99):
        """Compute the amax from the collected histogram

        Args:
            method: A string. One of ['entropy', 'mse', 'percentile']

        Keyword Arguments:
            stride: An integer. Default 1
            start_bin: An integer. Default 128
            percentils: A float number between [0, 100]. Default 99.99.

        Returns:
            amax: a tensor
        """
        if isinstance(self._calib_hist, torch.Tensor):
            calib_hist = self._calib_hist.to(torch.int64).cpu().numpy()
            calib_bin_edges = self._calib_bin_edges.cpu().numpy()
        else:
            calib_hist = self._calib_hist
            calib_bin_edges = self._calib_bin_edges

        if method == 'entropy':
            calib_amax = _compute_amax_entropy(calib_hist, calib_bin_edges, self._num_bits, self._unsigned, stride,
                                               start_bin)
        elif method == 'mse':
            calib_amax = _compute_amax_mse(calib_hist, calib_bin_edges, self._num_bits, self._unsigned, stride,
                                           start_bin)
        elif method == 'percentile':
            calib_amax = _compute_amax_percentile(calib_hist, calib_bin_edges, percentile)
        else:
            raise TypeError("Unknown calibration method {}".format(method))

        return calib_amax
    
    def __str__(self):
        s = "HistogramCalibrator("
        if self._calib_bin_edges is None:
            bin_edge_str = "None"
        else:
            bin_edge_str = "[{:3f}, ..., {:3f}]({})".format(self._calib_bin_edges[0], self._calib_bin_edges[-1],len(self._calib_bin_edges))
        s += "calib_bin_edges={}, ".format(bin_edge_str)
        return s
    

def _compute_amax_entropy(calib_hist, calib_bin_edges, num_bits, unsigned, stride=1, start_bin=128):
    """Returns amax that minimizes KL-Divergence of the collected histogram"""

    # If calibrator hasn't collected any data, return none
    if calib_bin_edges is None and calib_hist is None:
        return None

    def _normalize_distr(distr):
        summ = np.sum(distr)
        if summ != 0:
            distr = distr / summ

    bins = calib_hist[:]
    bins[0] = bins[1]

    total_data = np.sum(bins)

    divergences = []
    arguments = []

    # we are quantizing to 128 values + sign if num_bits=8
    nbins = 1 << (num_bits - 1 + int(unsigned))

    starting = start_bin
    stop = len(bins)

    new_density_counts = np.zeros(nbins, dtype=np.float64)

    for i in range(starting, stop + 1, stride):
        new_density_counts.fill(0)
        space = np.linspace(0, i, num=nbins + 1)
        digitized_space = np.digitize(range(i), space) - 1

        digitized_space[bins[:i] == 0] = -1

        for idx, digitized in enumerate(digitized_space):
            if digitized != -1:
                new_density_counts[digitized] += bins[idx]

        counter = Counter(digitized_space)
        for key, val in counter.items():
            if key != -1:
                new_density_counts[key] = new_density_counts[key] / val

        new_density = np.zeros(i, dtype=np.float64)
        for idx, digitized in enumerate(digitized_space):
            if digitized != -1:
                new_density[idx] = new_density_counts[digitized]

        total_counts_new = np.sum(new_density) + np.sum(bins[i:])
        _normalize_distr(new_density)

        reference_density = np.array(bins[:len(digitized_space)])
        reference_density[-1] += np.sum(bins[i:])

        total_counts_old = np.sum(reference_density)
        if round(total_counts_new) != total_data or round(total_counts_old) != total_data:
            raise RuntimeError("Count mismatch! total_counts_new={}, total_counts_old={}, total_data={}".format(
                total_counts_new, total_counts_old, total_data))

        _normalize_distr(reference_density)

        ent = entropy(reference_density, new_density)
        divergences.append(ent)
        arguments.append(i)

    divergences = np.array(divergences)
    print("divergences={}".format(divergences))
    last_argmin = len(divergences) - 1 - np.argmin(divergences[::-1])
    calib_amax = calib_bin_edges[last_argmin * stride + starting]
    calib_amax = torch.tensor(calib_amax.item())  #pylint: disable=not-callable

    return calib_amax


def _compute_amax_mse(calib_hist, calib_bin_edges, levels, stride=1, start_bin=256):
    """Returns amax that minimizes MSE of the collected histogram"""

    # If calibrator hasn't collected any data, return none
    if calib_bin_edges is None and calib_hist is None:
        return None

    counts = torch.from_numpy(calib_hist[:]).float()
    counts = counts.cuda() if torch.cuda.is_available() else counts.cpu()
    edges = torch.from_numpy(calib_bin_edges[:]).float()
    edges = edges.cuda() if torch.cuda.is_available() else edges.cpu()
    centers = (edges[1:] + edges[:-1]) / 2

    mses = []
    arguments = []

    for i in range(start_bin, len(centers), stride):

        amax = centers[i]
        if isinstance(levels, int) and levels >= 0:
            scale = (2/(levels-1))*amax
            if levels == 0:
                print("num_bits is 0. This will result in the tensor being quantized without checking the quantization range.")
                quant_centers = torch.round(centers/scale)*scale
            else:
                quant_centers = torch.clamp(torch.round(centers/scale), -levels//2, levels//2)*scale
        else:
            raise TypeError("Invalid levels. levels must be a positive integer.")

        mse = ((quant_centers - centers)**2 * counts).mean()

        mses.append(mse.cpu())
        arguments.append(i)

    #print("mses={}".format(mses))
    argmin = np.argmin(mses)
    calib_amax = centers[arguments[argmin]]

    return calib_amax


def _compute_amax_percentile(calib_hist, calib_bin_edges, percentile):
    """Returns amax that clips the percentile fraction of collected data"""

    if percentile < 0 or percentile > 100:
        raise ValueError("Invalid percentile. Must be in range 0 <= percentile <= 100.")

    # If calibrator hasn't collected any data, return none
    if calib_bin_edges is None and calib_hist is None:
        return None

    total = calib_hist.sum()
    cdf = np.cumsum(calib_hist / total)
    idx = np.searchsorted(cdf, percentile / 100)
    calib_amax = calib_bin_edges[idx]
    calib_amax = torch.tensor(calib_amax.item())  #pylint: disable=not-callable

    return calib_amax




