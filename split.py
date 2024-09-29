"""
Test and train splits for seasonal data
"""
import numpy

from typing import Callable
from datetime import datetime, timedelta
from iris.time import PartialDateTime
from iris.coords import DimCoord, AuxCoord

from .numbers import _FP_TYPE
from .numbers import _TIME_UNIT, _TIME_UNIT_ONLY_STRING, _TIME_UNIT_START
from .numbers import _DAY, _YEAR, _YEAR_START, _YEAR_END

from .useful import rnd

_SEASON_SIZE = timedelta(days=45)

def tconvert(t: datetime | _FP_TYPE) -> datetime:
    """
    Allow datetime objects through, but convert
    floating point times to datetime objects using
    the standard coordinate settings.
    """
    if type(t) == _FP_TYPE:
        return _TIME_UNIT_START + t * _TIME_UNIT
    else:
        return t

def in_partial_range(rng: list[PartialDateTime],
                     t: datetime | _FP_TYPE) -> bool:
    """
    Check if time is on or between two partials.
    """
    tt = tconvert(t)
    return (rng[0] <= tt) and (tt <= rng[1])

def in_many_partial_ranges(rngs: list[list[PartialDateTime]],
                           t: datetime | _FP_TYPE) -> bool:
    """
    Check if t is on or between any of the pairs of partials.
    """
    tt = tconvert(t)
    for r in rngs:
        if in_partial_range(r, tt):
            return True
    return False

def same_season(t0: datetime | _FP_TYPE) -> Callable[[datetime | _FP_TYPE], bool]:
    """
    Construct a lambda that checks if a datetime is in the
    same season as t0
    """
    tt0 = tconvert(t0)

    start = tt0 - _SEASON_SIZE
    end = tt0 + _SEASON_SIZE

    if start.year == tt0.year - 1:
        wrap = True
    elif end.year == tt0.year + 1:
        wrap = True
    else:
        wrap = False

    intervals = []
    if wrap:
        partial_start = _YEAR_START
        partial_end = PartialDateTime(month=end.month,
                                      day=end.day)
        intervals.append([partial_start, partial_end])
        partial_start = PartialDateTime(month=start.month,
                                        day=start.day)
        partial_end = _YEAR_END
        intervals.append([partial_start, partial_end])
    else:
        partial_start = PartialDateTime(month=start.month,
                                        day=start.day)
        partial_end = PartialDateTime(month=end.month,
                                      day=end.day)
        intervals.append([partial_start, partial_end])

    return lambda t: in_many_partial_ranges(intervals, t)

def season_mask(t0: datetime | _FP_TYPE) -> Callable[[DimCoord], list[bool]]:
    """
    Construct a mask compatible with the given DimCoord.
    """
    tt0 = tconvert(t0)
    fn = same_season(tt0)
    return lambda crd: numpy.logical_not(numpy.array(list(map(fn, crd.points))))

def count_seasons(t0: datetime | _FP_TYPE) -> Callable[[_FP_TYPE], int]:
    """
    Count the seasons between two times
    """
    tt0 = tconvert(t0)
    return lambda t: int(rnd(((_TIME_UNIT_START +  t * _TIME_UNIT) - tt0) / _YEAR))

def season_group(t0: datetime | _FP_TYPE) ->  Callable[[DimCoord], list[int]]:
    """
    Assign season number as group to each day in in the time coordinate
    """
    tt0 = tconvert(t0)
    fn = count_seasons(tt0)
    return lambda crd: numpy.array(list(map(fn, crd.points)))
