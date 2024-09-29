"""
Set up data from the NCEP analysis
"""
import logging
import requests

from typing import Union, Optional
from pathlib import Path

from iris import Constraint
from iris.cube import Cube

from .source import source
from .source import _TMP_FOLDER
from .source import _DEFAULT_FIELD_MAP, _DEFAULT_CRD_MAP, _DEFAULT_UNIT_MAP

_REANALYSIS_PATH = "ncep.reanalysis/"

base_url = "https://downloads.psl.noaa.gov/Datasets/"
base_url += _REANALYSIS_PATH

class ncep_at_surface(source):
    """
    Download and process the NCEP analysis
    using the source class.
    """

    def get(self):
        """
        Get the raw data file from NCEP.
        """
        msg = "ncep_at_surface::get()"
        dl = self.tmp / self.raw_file_name
        if dl.is_file():
            if self.use_existing:
                msg += " using previous download"
                return dl
            else:
                dl.unlink()
        response = requests.get(self.url)
        with dl.open("wb") as handle:
            handle.write(response.content)
        return dl

    def __init__(self,
                 fld: str,
                 year: int | str,
                 local_path: str | Path,
                 transform: bool = True,
                 pack: bool = True,
                 selection: Optional[Constraint] = None,
                 target_grid: Optional[Cube] = None,
                 member: int = 0,
                 tmp: str | Path = _TMP_FOLDER / _REANALYSIS_PATH / 'surface/',
                 keep: bool = False,
                 use_existing: bool = False,
                 garbage_collect: bool = True,
                 field_map: dict[str, str] = _DEFAULT_FIELD_MAP,
                 crd_map: dict[str, str] = _DEFAULT_CRD_MAP,
                 unit_map: dict[str, str] = _DEFAULT_UNIT_MAP):
        """
        NCEP analysis is organised into years.
        This class downloads one year of the given field.
        """
        self.use_existing = use_existing
        self.raw_file_name = fld + "." + str(year) + ".nc"
        logging.info(self.raw_file_name)
        self.url = base_url + "surface/" + self.raw_file_name
        logging.info(self.url)
        source.__init__(self,
                        local_path,
                        transform,
                        pack,
                        selection,
                        target_grid,
                        member,
                        tmp,
                        keep,
                        garbage_collect,
                        field_map,
                        crd_map,
                        unit_map)
        return

class ncep_on_pressure_levels(source):
    """
    Download and process the NCEP analysis
    using the source class.
    """

    def get(self):
        """
        Get the raw data file from NCEP.
        """
        msg = "ncep_on_pressure_levels::get()"
        dl = self.tmp / self.raw_file_name
        if dl.is_file():
            if self.use_existing:
                msg += " using previous download"
                return dl
            else:
                dl.unlink()
        response = requests.get(self.url)
        with dl.open("wb") as handle:
            handle.write(response.content)
        return dl

    def __init__(self,
                 fld: str,
                 year: int | str,
                 local_path: str | Path,
                 transform: bool = True,
                 pack: bool = True,
                 selection: Optional[Constraint] = None,
                 target_grid: Optional[Cube] = None,
                 member: int = 0,
                 tmp: str | Path = _TMP_FOLDER / _REANALYSIS_PATH / 'pressure/',
                 keep: bool = False,
                 use_existing: bool = False,
                 garbage_collect: bool = True,
                 field_map: dict[str, str] = _DEFAULT_FIELD_MAP,
                 crd_map: dict[str, str] = _DEFAULT_CRD_MAP,
                 unit_map: dict[str, str] = _DEFAULT_UNIT_MAP):
        """
        NCEP analysis is organised into years.
        This class downloads one year of the given field.
        """
        self.use_existing = use_existing
        self.raw_file_name = fld + "." + str(year) + ".nc"
        logging.info(self.raw_file_name)
        self.url = base_url + "pressure/" + self.raw_file_name
        logging.info(self.url)
        source.__init__(self,
                        local_path,
                        transform,
                        pack,
                        selection,
                        target_grid,
                        member,
                        tmp,
                        keep,
                        garbage_collect,
                        field_map,
                        crd_map,
                        unit_map)
        return
