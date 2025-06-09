import functools
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union


# Mirrors enum in `core/scalar_type.hpp`
class SglangNanRepr(Enum):
    NONE = 0  # nans are not supported
    IEEE_754 = 1  # nans are: Exp all 1s, mantissa not all 0s
    EXTD_RANGE_MAX_MIN = 2  # nans are: Exp all 1s, mantissa all 1s


# This SglangScalarType class is a parallel implementation of the C++ ScalarType
# class found in csrc/core/scalar_type.hpp.
@dataclass(frozen=True)
class SglangScalarType:
    """
    SglangScalarType can represent a wide range of floating point and integer
    types, in particular it can be used to represent sub-byte data types
    (something that torch.dtype currently does not support). It is also
    capable of  representing types with a bias, i.e.:
      `stored_value = value + bias`,
    this is useful for quantized types (e.g. standard GPTQ 4bit uses a bias
    of 8). The implementation for this class can be found in
    csrc/core/scalar_type.hpp, these type signatures should be kept in sync
    with that file.
    """

    exponent: int
    """
    Number of bits in the exponent if this is a floating point type
    (zero if this an integer type)
    """

    mantissa: int
    """
    Number of bits in the mantissa if this is a floating point type,
    or the number bits representing an integer excluding the sign bit if
    this an integer type.
    """

    signed: bool
    "If the type is signed (i.e. has a sign bit)"

    bias: int
    """
    bias used to encode the values in this scalar type
    (value = stored_value - bias, default 0) for example if we store the
    type as an unsigned integer with a bias of 128 then the value 0 will be
    stored as 128 and -1 will be stored as 127 and 1 will be stored as 129.
    """

    _finite_values_only: bool = False
    """
    Private: if infs are supported, use `has_infs()` instead.
    """

    nan_repr: SglangNanRepr = SglangNanRepr.IEEE_754
    """
    How NaNs are represent in this scalar type, returns SglangNanRepr value.
    (not applicable for integer types)
    """

    def _floating_point_max_int(self) -> int:
        assert (
            self.mantissa <= 52 and self.exponent <= 11
        ), f"Cannot represent max/min as a double for type {self.__str__()}"

        max_mantissa = (1 << self.mantissa) - 1
        if self.nan_repr == SglangNanRepr.EXTD_RANGE_MAX_MIN:
            max_mantissa = max_mantissa - 1

        max_exponent = (1 << self.exponent) - 2
        if (self.nan_repr == SglangNanRepr.EXTD_RANGE_MAX_MIN
                or self.nan_repr == SglangNanRepr.NONE):
            assert (
                self.exponent < 11
            ), f"Cannot represent max/min as a double for type {self.__str__()}"
            max_exponent = max_exponent + 1

        exponent_bias = (1 << (self.exponent - 1)) - 1
        exponent_bias_double = (1 << 10) - 1

        max_exponent_double = (max_exponent - exponent_bias +
                               exponent_bias_double)

        return (max_mantissa <<
                (52 - self.mantissa)) | (max_exponent_double << 52)

    def _floating_point_max(self) -> float:
        double_raw = self._floating_point_max_int()
        return struct.unpack('!d', struct.pack('!Q', double_raw))[0]

    def _raw_max(self) -> Union[int, float]:
        if self.is_floating_point():
            return self._floating_point_max()
        else:
            assert (self.size_bits < 64 or self.size_bits == 64
                    and self.is_signed()), "Cannot represent max as an int"
            return (1 << self.mantissa) - 1

    def _raw_min(self) -> Union[int, float]:
        if self.is_floating_point():
            assert self.is_signed(
            ), "We currently assume all floating point types are signed"
            sign_bit_double = 1 << 63

            max_raw = self._floating_point_max_int()
            min_raw = max_raw | sign_bit_double
            return struct.unpack('!d', struct.pack('!Q', min_raw))[0]
        else:
            assert (not self.is_signed() or
                    self.size_bits <= 64), "Cannot represent min as a int64_t"

            if self.is_signed():
                return -(1 << (self.size_bits - 1))
            else:
                return 0

    @functools.cached_property
    def id(self) -> int:
        val = 0
        offset = 0

        def or_and_advance(member, bit_width):
            nonlocal val
            nonlocal offset
            bit_mask = (1 << bit_width) - 1
            val = val | (int(member) & bit_mask) << offset
            offset = offset + bit_width

        or_and_advance(self.exponent, 8)
        or_and_advance(self.mantissa, 8)
        or_and_advance(self.signed, 1)
        or_and_advance(self.bias, 32)
        or_and_advance(self._finite_values_only, 1)
        or_and_advance(self.nan_repr.value, 8)

        assert offset <= 64, \
            f"SglangScalarType fields too big {offset} to fit into an int64"

        return val

    @property
    def size_bits(self) -> int:
        return self.exponent + self.mantissa + int(self.signed)

    def min(self) -> Union[int, float]:
        return self._raw_min() - self.bias

    def max(self) -> Union[int, float]:
        return self._raw_max() - self.bias

    def is_signed(self) -> bool:
        return self.signed

    def is_floating_point(self) -> bool:
        return self.exponent != 0

    def is_integer(self) -> bool:
        return self.exponent == 0

    def has_bias(self) -> bool:
        return self.bias != 0

    def has_infs(self) -> bool:
        return not self._finite_values_only

    def has_nans(self) -> bool:
        return self.nan_repr != SglangNanRepr.NONE.value

    def is_ieee_754(self) -> bool:
        return self.nan_repr == SglangNanRepr.IEEE_754.value and \
            not self._finite_values_only

    def __str__(self) -> str:
        if self.is_floating_point():
            ret = "float" + str(self.size_bits) + "_e" + str(
                self.exponent) + "m" + str(self.mantissa)

            if not self.is_ieee_754():
                if self._finite_values_only:
                    ret = ret + "f"
                if self.nan_repr != SglangNanRepr.NONE:
                    ret = ret + "n"
            return ret
        else:
            ret = ("int" if self.is_signed() else "uint") + str(self.size_bits)
            if self.has_bias():
                ret = ret + "b" + str(self.bias)
            return ret

    def __repr__(self) -> str:
        return "SglangScalarType." + self.__str__()

    def __len__(self) -> int:
        # __len__ needs to be defined (and has to throw TypeError) for pytorch's
        # opcheck to work with ScalarType.
        raise TypeError

    @classmethod
    def int_(cls, size_bits: int, bias: Optional[int]) -> 'SglangScalarType':
        ret = cls(0, size_bits - 1, True, bias if bias else 0)
        ret.id  # noqa B018: make sure the id is cached
        return ret

    @classmethod
    def uint(cls, size_bits: int, bias: Optional[int]) -> 'SglangScalarType':
        ret = cls(0, size_bits, False, bias if bias else 0)
        ret.id  # noqa B018: make sure the id is cached
        return ret

    @classmethod
    def float_IEEE754(cls, exponent: int, mantissa: int) -> 'SglangScalarType':
        assert (mantissa > 0 and exponent > 0)
        ret = cls(exponent, mantissa, True, 0)
        ret.id  # noqa B018: make sure the id is cached
        return ret

    @classmethod
    def float_(cls, exponent: int, mantissa: int, finite_values_only: bool,
               nan_repr: SglangNanRepr) -> 'SglangScalarType':
        assert (mantissa > 0 and exponent > 0)
        assert (nan_repr != SglangNanRepr.IEEE_754), (
            "use `float_IEEE754` constructor for floating point types that "
            "follow IEEE 754 conventions")
        ret = cls(exponent, mantissa, True, 0, finite_values_only, nan_repr)
        ret.id  # noqa B018: make sure the id is cached
        return ret


class sglang_scalar_types:
    int4 = SglangScalarType.int_(4, None)
    uint4 = SglangScalarType.uint(4, None)
    int8 = SglangScalarType.int_(8, None)
    uint8 = SglangScalarType.uint(8, None)
    float8_e4m3fn = SglangScalarType.float_(4, 3, True, SglangNanRepr.EXTD_RANGE_MAX_MIN)
    float8_e5m2 = SglangScalarType.float_IEEE754(5, 2)
    float16_e8m7 = SglangScalarType.float_IEEE754(8, 7) # bfloat16
    float16_e5m10 = SglangScalarType.float_IEEE754(5, 10) # float16

    float6_e3m2f = SglangScalarType.float_(3, 2, True, SglangNanRepr.NONE)

    # "gptq" types (unsigned integers with bias)
    uint2b2 = SglangScalarType.uint(2, 2)
    uint3b4 = SglangScalarType.uint(3, 4)
    uint4b8 = SglangScalarType.uint(4, 8)
    uint8b128 = SglangScalarType.uint(8, 128)

    # colloquial names
    bfloat16 = float16_e8m7
    float16 = float16_e5m10
