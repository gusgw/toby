# Numerical configuration and constants

import numpy
from datetime import datetime, timedelta
from iris.time import PartialDateTime

allowed_types = {
                    'float32' : numpy.float32,
                    'float64' : numpy.float64,
                    'float128' : numpy.float128,
                    'complex64' : numpy.complex64,
                    'complex128' : numpy.complex128,
                    'complex256' : numpy.complex256
                }

_FP_TYPE_STRING = 'float32'
_FP_DOUBLE_TYPE_STRING = 'float64'
_COMPLEX_FP_TYPE_STRING = 'complex64'
_COMPLEX_FP_DOUBLE_TYPE_STRING = 'complex128'

_FP_TYPE = allowed_types[_FP_TYPE_STRING]
_FP_DOUBLE_TYPE = allowed_types[_FP_DOUBLE_TYPE_STRING]
_COMPLEX_FP_TYPE = allowed_types[_COMPLEX_FP_TYPE_STRING]
_COMPLEX_FP_DOUBLE_TYPE = allowed_types[_COMPLEX_FP_DOUBLE_TYPE_STRING]

_FP_INFO = numpy.finfo(_FP_TYPE)
_FP_DOUBLE_TYPE_INFO = numpy.finfo(_FP_DOUBLE_TYPE)
_COMPLEX_FP_INFO = numpy.finfo(_COMPLEX_FP_TYPE)
_COMPLEX_DOUBLE_FP_INFO = numpy.finfo(_COMPLEX_FP_DOUBLE_TYPE)

# The choices of *_STRING above should ensure these
# quantities are relevant to complex as well as real types.
_EPS = _FP_INFO.eps
_EPSNEG = _FP_INFO.epsneg
_MAX = _FP_INFO.max
_MIN = _FP_INFO.min
_TINY = _FP_INFO.tiny

_RELATIVE_TOLERANCE = _EPS

_TIME_UNIT = timedelta(hours=1)
_DAY = timedelta(days=1)
_YEAR = timedelta(days=365.25)
_YEAR_START = PartialDateTime(month=1, day=1)
_YEAR_END = PartialDateTime(month=12, day=31)
_BEFORE_LEAP_DAY = 59
_LEAP_DAY = PartialDateTime(month=2, day=29)
_MARCH_START = PartialDateTime(month=3, day=1)
_AFTER_LEAP_DAY = 306
_TIME_UNIT_STRING = "hours since 1970-01-01 00:00:0.0"
_TIME_UNIT_ONLY_STRING = "hour"
_TIME_UNIT_START = datetime.strptime("19700101T0000+0000", "%Y%m%dT%H%M%z")

radius_in_m = _FP_TYPE(6371229.0)

degrees_in_pi_radians = _FP_TYPE(180)

dd_temperature_in_celsius = _FP_TYPE(18.333333333)

triple_point_of_water_in_kelvin = _FP_TYPE(273.15)
