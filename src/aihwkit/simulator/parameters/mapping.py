# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=too-many-instance-attributes

"""Mapping parameters for resistive processing units."""

from torch import Tensor
from typing import Type, Optional, ClassVar, Union, List
from dataclasses import dataclass, fields, field

from aihwkit.exceptions import ConfigError

from .base import RPUConfigBase
from .helpers import _PrintableMixin

from .enums import WeightQuantizerType


@dataclass
class MappingParameter(_PrintableMixin):
    """Parameter related to hardware design and the mapping of logical
    weight matrices to physical tiles.

    Caution:

        Some of these parameters have only an effect for modules that
        support tile mappings.
    """

    digital_bias: bool = True
    """Whether the bias term is handled by the analog tile or kept in
    digital.

    Note:
        Default is having a *digital* bias so that bias values are
        *not* stored onto the analog crossbar. This needs to be
        supported by the chip design. Set to False if the analog bias
        is instead situated on the the crossbar itself (as an extra
        column)

    Note:
        ``digital_bias`` is supported by *all* analog modules.
    """

    weight_scaling_omega: float = 0.0
    """omega_scale is a user defined parameter used to scale the weights
    while remapping these to cover the full range of values allowed.
    By default, no remapping is performed. If values > 0.0 are supplied
    the abs-max of the weight is scaled to that value.
    """

    weight_scaling_columnwise: bool = False
    """Whether the weight matrix will be remapped column-wise over
    the maximum device allowed value."""

    weight_scaling_lr_compensation: bool = False
    """Whether to adjust the LR to compensate for the mapping factors
    that are not learned.

    The learning rate will be divided
    for a tile individually by the mean of the mapping scales that are
    determined by the ``weight_scaling_omega`` setting.

    Otherwise the gradient information will be divided isntead before
    the update.
    """

    learn_out_scaling: bool = False
    """Define (additional) out scales that are learnable parameter
    used to scale the output."""

    out_scaling_columnwise: bool = False
    """Whether the learnable out scaling parameter enabled by
    ``learn_out_scaling`` is a scalar (``False``) or learned for
    each output (``True``).
    """

    max_input_size: int = 512
    """Maximal input size (number of columns) of the weight matrix
    that is handled on a single analog tile.

    If the logical weight matrix size exceeds this size it will be
    split and mapped onto multiple analog tiles.

    Caution:
        Only relevant for ``Mapped`` modules such as
        :class:`aihwkit.nn.modules.linear_mapped.AnalogLinearMapped`.
    """

    max_output_size: int = 512
    """Maximal output size (number of rows) of the weight matrix
    that is handled on a single analog tile.

    If the logical weight matrix size exceeds this size it will be
    split and mapped onto multiple analog tiles.

    Caution:
        Only relevant for ``Mapped`` modules such as
        :class:`aihwkit.nn.modules.linear_mapped.AnalogLinearMapped`.
    """

    def compatible_with(self, mapping: "MappingParameter") -> bool:
        """Checks compatiblity

        Args:
            mapping: param to check

        Returns:
            success:  if compatible
        """
        if mapping == self:
            return True

        for key in fields(mapping):
            if key.name in [
                "weight_scaling_omega",
                "weight_scaling_columnwise",
                "weight_scaling_lr_compensation",
            ]:
                continue

            if mapping.__dict__[key.name] != self.__dict__[key.name]:
                return False
        return True
    

# -- MODIFIED: added quantize parameter
@dataclass
class WeightQuantizerParameter(_PrintableMixin):
    """Parameter related to quantization of weights."""

    bindings_class: ClassVar[Optional[Union[str, Type]]] = "WeightQuantizerParameter"
    bindings_module: ClassVar[str] = "tiles"

    resolution: float = 0
    """Whether to quantize the weights to the tile's precision.

    If set to a integer value, the original weights will be quantized to
    this number of quantization levels.
    """

    eps: float = 0
    """If set to a value in (0,0.99], it allows to fine tun the resolution parameter
    to include up to a fraction (1-eps) of the weight population inside the FSR derived
    """

    levels: int = 0
    """The number of quantization levels.

    If set to 0, the quantization levels will be ignored and the quantization
    will just be based on the quantize (resolution) parameter.
    """

    quantize_last_column: bool = True
    """Whether to quantize the last column of the weight matrix (usually the bias).
    
    
    If set to True, the last column of the weight matrix will be quantized
    along with the other weights.
    """

    quantizer_type: WeightQuantizerType = field(
        default_factory=lambda: WeightQuantizerType.UNIFORM, metadata={"always_show": True}
    )
    """Specifies the type of quantizer to use.

    The quantizer type can be one of the following:
    - Uniform: quantizes the weights uniformly between the quantization levels.
    - FixedValued: quantizes the weights to some fixed values.
    """

    quant_values: List[float] = field(
        default_factory=lambda: [-1.0, 1.0],
        metadata={"hide_if": [-1.0, 1.0]},
    )

    stochastic_round: bool = False
    """Whether to use stochastic rounding when quantizing the weights.

    If set to True, the weights will be rounded to the nearest quantization
    level with a probability proportional to the distance to the two closest
    quantization levels.
    """

    debug: bool = True
    """Whether to print debug information during quantization."""

    def fit(self, weights: Tensor) -> None:
        """The function is used to fit the resolution parameter for the current weights
        considered, so that up to (1-eps)% of the weghts population (at least) is covered
        by the FSR

        Args:
        weights: list of weights to be quantized
        """

        if (self.eps == 0):
            return
        if (self.eps >0.99):
            raise ValueError("The eps parameter must be less than 0.99")
            return
        # Create a deepcopy of the weights tensor
        w = weights.detach().clone()

        # Sort the weights
        w = w.reshape(-1).tolist()
        w.sort()
        max_elem = abs(max(w, key=abs))
        tot_size = len(w)
        max_count = int(tot_size * (1 - self.eps))

        # starting from the ends, move towards the center to find the min and max elements
        # delimiting the (1 - eps)% of the population
        r_idx, l_idx = 0, 0
        max_bound, min_bound = w[0], w[tot_size -1]
        for i in range(max_count):
            if abs(w[l_idx]) > abs(w[tot_size - 1 - r_idx]):
                limit = abs(w[l_idx])
                l_idx +=1
            else:
                limit = abs(w[tot_size - 1 - r_idx])
                r_idx +=1

        self.resolution = (2/(self.levels - 1)) * (limit/max_elem)
        return
        

# -- MODIFIED: added quantize parameter


@dataclass
class MappableRPU(RPUConfigBase, _PrintableMixin):
    """Defines the mapping parameters and utility factories"""

    tile_array_class: Optional[Type] = None
    """Tile array class that correspond to the RPUConfig.

    This is used to build logical arrays of tiles. Needs to be defined
    in the derived class.
    """

    mapping: MappingParameter = field(default_factory=MappingParameter)
    """Parameter related to mapping weights to tiles for supporting modules."""

    quantization: WeightQuantizerParameter = field(default_factory=WeightQuantizerParameter, metadata=dict(bindings_include=True))
    """Parameter for weight quantizer.

    If the modifier type is set, t is called just once, to quantize the weights at the
    beginning of the testing/evaluation phase.
    """

    def get_default_tile_module_class(self, out_size: int = 0, in_size: int = 0) -> Type:
        """Returns the default TileModule class.

        Args:
            out_size: overall output size
            in_size: overall output size

        Raises:
            ConfigError: in case tile array is not defined.
        """

        if self.tile_array_class is None:
            ConfigError("RPUConfig does not support any tile array class")

        if self.tile_array_class is None or (
            self.mapping.max_input_size == 0 and self.mapping.max_output_size == 0
        ):
            return self.tile_class
        if self.mapping.max_input_size < in_size or self.mapping.max_output_size < out_size:
            return self.tile_array_class
        return self.tile_class
