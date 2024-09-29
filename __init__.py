"""
Schnauzer fetches climate data and processes it into usable form
"""
from .numbers import _TIME_UNIT_START, _TIME_UNIT, _DAY, _YEAR
from .source import source
from .ncep import ncep_on_pressure_levels, ncep_at_surface
from .grids import grids
from .split import season_mask, season_group
from .learn import test, train
