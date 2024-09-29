"""
Toby fetches climate data
"""

import logging
import time
import copy
import datetime
import os
import gc

import numpy
import iris

from pathlib import Path
from typing import Union, Optional
from datetime import timedelta
from scipy.fft import fft
from iris import Constraint
from iris.cube import Cube, CubeList
from iris.coords import DimCoord, AuxCoord
from iris.exceptions import CoordinateNotFoundError
from iris.coord_systems import GeogCS
from iris.time import PartialDateTime

from .useful import copy_file, load_with_retry, cname
from .useful import print_cubes_with_title, fields_are_ok, frequency_coordinate
from .numbers import _FP_TYPE, _COMPLEX_FP_TYPE
from .numbers import _FP_TYPE_STRING, _COMPLEX_FP_TYPE_STRING
from .numbers import radius_in_m, _EPS, _RELATIVE_TOLERANCE
from .numbers import _TIME_UNIT, _TIME_UNIT_STRING, _TIME_UNIT_ONLY_STRING, _DAY
from .grids import get_area_wt, _AREA_WEIGHT_COORDINATE

iris.FUTURE.datum_support = True

_TMP_FOLDER = Path("/mnt/data/seasons/tmp")
_DUMMY_DOWNLOAD = _TMP_FOLDER / "test.download.nc"

_ZULU_POINT = timedelta(hours=12)
_MEMBER_ZERO_PAD = 3

# Field names should be mapped to new names with no spaces or underscores
_DEFAULT_FIELD_MAP = {"4xDaily Geopotential height" : "hgt",
                      "4xDaily Air temperature" : "air",
                      "4xDaily Sea Level Pressure": 'prmsl'}
_DEFAULT_CRD_MAP = {"Level" : "pressure"}
_DEFAULT_UNIT_MAP = {'prmsl': 'kPa',
                     'hgt': 'm',
                     'time': _TIME_UNIT_STRING,
                     'pressure': 'kPa',
                     'latitude': 'radians',
                     'longitude': 'radians'}

class source:
    """
    Download, process, and check a file containing
    weather or climate data.
    """

    @property
    def mtime(self) -> float:
        """
        Record new mtime and check it is sensible

        If self.mtime is None then we have no information about
        modification time and should get the modification time, and
        save it. If self.mtime is already set, then make sure it is
        either unchanged or increased.

        We need to save the modification time because when cubes
        from another source to this source, we need to record
        the latest time. In other words, always record the latest
        modification time from any of the data in the source.

        Returns
        -------
        Return the best modification time.
        """
        lp = self.local_path
        if lp.is_file():
            mtime_now = os.path.getmtime(lp.as_posix())
            if self._mtime is None:
                self._mtime = mtime_now
            else:
                if mtime_now > self._mtime:
                    logging.info("source::ok() processed file has been modified")
                    self._mtime = mtime_now
        else:
            msg = "source::mtime() cannot find " + lp.as_posix()
        return self._mtime

    @mtime.setter
    def mtime(self, value: float):
        """
        Set the mtime for the processed output file.

        See the documentation for the corresponding getter.

        Parameters
        ----------
        value: float
            The value to be stored in self._mtime. Normally
            this will be the output of a call to os.path.getmtime().
        """
        self._mtime = value
        return

    def check_data_and_crd_type(self, test: CubeList) -> bool:
        """
        Check floating point types for all cubes and coordinates

        Make sure all the types loaded are as expected. See .numbers
        for the setup of allowed types and 

        https://numpy.org/doc/stable/reference/arrays.dtypes.html

        for more information.

        Parameters
        ----------
        test: CubeList
            The result of a successful iris.load() command on the Path that
            is expected to contain the properly processed output file.

        Returns
        -------
        Returns True if all cubes and their coordinates have the expected type,
        otherwise False.
        """
        msg = "source::check_data_and_crd_type()"
        if len(test) == 0:
            msg += " no cubes found"
            logging.error(msg)
            raise RuntimeError(msg)
        if self.transform and not self.pack:
            expected = _COMPLEX_FP_TYPE
        else:
            expected = _FP_TYPE
        for c in test:
            if c.dtype != expected:
                msg += " cube " + c.name() + " has unexpected type " + str(c.dtype)
                logging.warning(msg)
                return False
            for x in c.coords(dim_coords=True):
                if x.dtype != _FP_TYPE:
                    msg += " coordinate " + x.name()
                    msg += " in " + c.name()
                    msg += " has unexpected type " + str(x.dtype)
                    msg += " which should be " + str(_FP_TYPE)
                    logging.warning(msg)
                    return False
        return True

    def check_for_coordinates(self, test: CubeList) -> bool:
        """
        Confirm that the expected coordinates are present in all cubes in test.

        This routine should be called by source::ok() to make sure that the
        processed file on disk has cubes containing the desired coordinates.
        It should be called by other methods in the source class, but not
        otherwise.

        If there are no cubes to check then an exception is raised,
        because we expect other checks to have been done before this
        routine is called.

        Parameters
        ----------
        test: CubeList
            The result of a successful iris.load() command on the Path that
            is expected to contain the properly processed output file.

        Returns
        -------
        Returns True if all expected coordinates are present in every
        cube in test, False if one is found to be missing.
        """
        msg = "source::check_for_coordinates()"
        if len(test) == 0:
            msg += " no cubes found"
            logging.error(msg)
            raise RuntimeError(msg)
        for c in test:
            dimension_crds = list(map(cname, c.coords(dim_coords=True)))
            for xname in dimension_crds:
                logging.info(msg + " dimension coordinate: " + xname)
            auxiliary_crds = list(map(cname, c.coords(dim_coords=False)))
            for xname in auxiliary_crds:
                logging.info(msg + " auxiliary coordinate: " + xname)
            if 'member' not in auxiliary_crds:
                logging.warning(str(auxiliary_crds))
                msg += " ensemble member coordinate missing"
                logging.warning(msg)
                return False
            if _AREA_WEIGHT_COORDINATE not in auxiliary_crds:
                msg += " area weight coordinate missing"
                logging.warning(msg)
                return False
            if 'time' not in dimension_crds:
                msg += " time coordinate missing"
                logging.warning(msg)
                return False
            if 'latitude' not in dimension_crds:
                msg += " latitude coordinate missing"
                logging.warning(msg)
                return False
            if 'longitude' not in dimension_crds:
                msg += " longitude coordinate missing"
                logging.warning(msg)
                return False
        return True

    def check_names(self, test: CubeList) -> bool:
        """
        Make sure all cubes and coordinates have been renamed correctly.

        Maps for the names of fields and coordinates are provided so that
        the data provided by the source class can be consistently named.
        This method checks that none of the cubes in test, and none of
        their coordinates, have names that could be mapped.

        Parameters
        ----------
        test: CubeList
            The result of a successful iris.load() command on the Path that
            is expected to contain the properly processed output file.

        Returns
        -------
        Returns False if any field or coordinate is found that would
        be renamed if the maps stored as members of source were applied,
        otherwise True.
        """
        msg = "source::check_names()"
        if len(test) == 0:
            msg += " no cubes found"
            logging.error(msg)
            raise RuntimeError(msg)
        if self.field_map is None:
            msg += " no field map to check"
            logging.warning(msg)
        if self.crd_map is None:
            msg += " no coordinate map to check"
            logging.warning(msg)
        for c in test:
            if self.field_map is not None:
                if c.name() in self.field_map:
                    msg += " unmapped field name " + c.name()
                    msg += " and not ok()"
                    logging.warning(msg)
                    return False
            if self.crd_map is not None:
                for x in c.coords():
                    if x.name() in self.crd_map:
                        msg += " unampped coordinate name " + x.name()
                        msg += " and not ok()"
                        logging.warning(msg)
                        return False
        return True

    def check_units(self, test: CubeList) -> bool:
        """
        Make sure all cubes and coordinates have the desired units.

        Units should have been converted as specified in the unit mapping.
        This method checks that the units correctly match the cube and 
        coordinate names, it does not actually checked the values of the
        fields or coordinates.

        Parameters
        ----------
        test: CubeList
            The result of a successful iris.load() command on the Path that
            is expected to contain the properly processed output file.

        Returns
        -------
        Returns True if cube and coordinate units are listed as the
        values allowed by the unit mapping.
        """
        msg = "source::check_units()"
        if len(test) == 0:
            msg += " no cubes found"
            logging.error(msg)
            raise RuntimeError(msg)
        if self.unit_map is None:
            msg += " no unit map to check"
            logging.warning(msg) 
        for c in test:
            if self.unit_map is not None:
                if c.name() in self.unit_map:
                    if not str(c.units) == self.unit_map[c.name()]:
                        msg += " units " + str(c.units) + " in " + c.name()
                        msg += " should be " + self.unit_map[c.name()]
                        logging.warning(msg)
                        return False
                    for x in c.coords():
                        if x.name() in self.unit_map:
                            if not str(x.units) == self.unit_map[x.name()]:
                                msg += " units " + str(x.units) + " in " + x.name()
                                msg += " from " + c.name()
                                msg += " should be " + self.unit_map[c.name()]
                                logging.warning(msg)
                                return False
        return True

    def check_surface_grid(self, test: CubeList) -> bool:
        """
        Make sure the latitude and longitude coordinates are as expected.

        The longitude and latitude coordinates are checked against
        self.converted_grid. The desired grid is stored in self.target_grid,
        and on unit conversion, not only the cubes but also target_grid
        is converted and stored in converted_grid.

        Note that longitude points separated by 2\pi radians or 360 degrees
        are *not* recorded as the same. The reason for this is that the
        outputs should use exactly the same coordinates. If they're not
        identical something must have gone wrong.

        Note that the order of the coordinates is not checked here.

        Parameters
        ----------
        test: CubeList
            The result of a successful iris.load() command on the Path that
            is expected to contain the properly processed output file.

        Returns
        -------
        Returns True if the longitude and latitude grids are as expected.
        """
        msg = "source::check_surface_grid()"
        if len(test) == 0:
            msg += " no cubes found"
            logging.error(msg)
            raise RuntimeError(msg)

        tgt = self.target_grid
        cnv = self.converted_grid
        lp = self.local_path
        if (tgt is not None) and (cnv is not None):
            for c in test:
                try:
                    clat = c.coord('latitude').points
                    glat = cnv.coord('latitude').points
                except CoordinateNotFoundError as e:
                    msg += " cannot find latitude coordinate"
                    logging.error(msg)
                    logging.exception(e)
                    raise
                else:
                    if len(clat) != len(glat):
                        msg += " incorrect latitude grid size in "
                        msg += lp.as_posix()
                        logging.warning(msg)
                        return False

                try:
                    clon = c.coord('longitude').points
                    glon = cnv.coord('longitude').points
                except CoordinateNotFoundError as e:
                    msg += " cannot find longitude coordinate"
                    logging.error(msg)
                    logging.exception(e)
                    raise
                else:
                    if len(clon) != len(glon):
                        msg += " incorrect longitude grid size in "
                        msg += lp.as_posix()
                        logging.warning(msg)
                        return False

                delta = numpy.sqrt(numpy.mean((clat-glat)**2))
                if delta > 0.0:
                    msg += " incorrect latitude grid with delta "
                    msg += str(delta)
                    msg += " in " + lp.as_posix()
                    logging.warning(msg)
                    return False

                # delta = numpy.sqrt(numpy.mean((clon-glon)**2))
                # if delta > 0.0:
                #     msg += " incorrect longitude grid with delta "
                #     msg += str(delta)
                #     msg += " in " + lp.as_posix()
                #     logging.warning(msg)
                #     return False
        return True

    def check_clean(self, test: CubeList) -> bool:
        """
        Make sure coordinate attributes have been removed.

        Attributes attached to coordinates often cause problems
        when merging and concatenating. It is simplest to
        remove these. The attributes are still available in
        the raw downloaded file.

        Parameters
        ----------
        test: CubeList
            The result of a successful iris.load() command on the Path that
            is expected to contain the properly processed output file.

        Returns
        -------
        True if coordinates have been stripped of attributes as expected.
        """
        msg = "source::check_clean()"
        if len(test) == 0:
            msg += " no cubes found"
            logging.error(msg)
            raise RuntimeError(msg)
        for c in test:
            for x in c.coords():
                if x.attributes != {}:
                    msg += " attributes remaining in "
                    msg += x.name()
                    msg += " in " + c.name()
                    msg += " in " + self.local_path.as_posix()
                    logging.warning(msg)
                    return False
        return True

    def ok(self) -> bool:
        """
        Check the processed output file is as expected.

        If the file containing the processed output cubes exists,
        try to load cubes from the file, and if this is successful
        check that the desired coordinates are present, the cube
        and coordinate names are as expected, the units match
        the cube and coordinate names, the surface grid is
        correct, and the coordinate attributes have been removed.

        Returns
        -------
        True if all tests passed, False otherwise.
        """
        msg = "source:ok()"
        lp = self.local_path
        logging.info(msg + " checking " + lp.as_posix())
        if lp.is_file():
            if not fields_are_ok(lp.as_posix()):
                msg += "fields not ok in " + lp.as_posix()
                logging.warning(msg)
                return False
            test = iris.load(lp.as_posix())
            if len(test) > 0:
                if not self.check_data_and_crd_type(test): return False
                if not self.check_for_coordinates(test): return False
                if not self.check_names(test): return False
                if not self.check_units(test): return False
                if not self.check_surface_grid(test): return False
                if not self.check_clean(test): return False
                logging.info(msg + " " + lp.as_posix() + " is ok()")
                return True
            else:
                msg += " no cubes in " + lp.as_posix()
                logging.warning(msg)
                return False
        else:
            msg += " " + lp.as_posix() + " not found"
            logging.warning(msg)
            return False

    def add(self, other):
        """
        Combine cubes from another source into this object's cube list.

        Add the cubes stored in another source to this source. Details of
        the file from which the cubes are taken are not copied across,
        with the exception of the modification time. The stored modification
        time should be the latest time obtained from any of the files
        from which the cubes have been loaded.
        """
        msg = "source:add() adding cubes from " + other.local_path.as_posix()
        msg += " to " + self.local_path.as_posix()
        logging.info(msg)
        self.cubes.extend(other.cubes)
        self.cubes.sort(key=cname)
        if other.mtime > self.mtime:
            self.mtime = other.mtime
        self.all_files += other.all_files
        return

    def get(self) -> Path:
        """
        Dummy download for testing and development.

        This method copies a previously downloaded file
        instead of downloading. Child classes implement the
        actual download of the appropriate data type.

        Returns
        -------
        Return the pathlib.Path of the 'downloaded' file.
        """
        msg = "source:get() base class source.get()"
        msg += " uses a fake download"
        logging.info(msg)
        dl = self.tmp /  "hgt.2000.nc"
        if dl.is_file():
            dl.unlink()
        if not _DUMMY_DOWNLOAD.is_file():
            msg = "cannot run base class code without dummy download"
            logging.error(msg)
            raise RuntimeError(msg)
        copy_file(_DUMMY_DOWNLOAD, dl)
        return dl

    def bounds(self):
        """
        Add bounds guessed by Iris where possible

        Iris provides methods to check for bounds and add them
        if they are missing. Attempt this for all dimension 
        coordinates present in each cube.
        """
        msg = "source::bounds()"
        if self.cubes is not None:
            if len(self.cubes) > 0:
                for c in self.cubes:
                    for x in c.coords(dim_coords=True):
                        if not x.has_bounds():
                            x.guess_bounds()
            else:
                logging.warning(msg + " cannot process an empty cube list")
        else:
            logging.warning(msg + " no cube list to process")
        return

    def add_surface_coordinate_system(self):
        """
        Regridding requires that the coordinate system be set.

        Operations including regridding require that the cubes
        involved have consistent settings for the coordinate system.
        This method sets all cubes to the coordinate system for a
        sphere with radius set in numbers.py.

        Note that no exception is raised if self.cubes is missing
        or if it contains no cubes. A warning is logged.
        """
        msg = "source::add_surface_coordinate_system()"
        if self.cubes is not None:
            if len(self.cubes) > 0:
                for c in self.cubes:
                    c.coord('latitude').coord_system = GeogCS(radius_in_m)
                    c.coord('longitude').coord_system = GeogCS(radius_in_m)
            else:
                logging.warning(msg + " cannot process an empty cube list")
        else:
            logging.warning(msg + " no cube list to process")
        return

    def map_names(self):
        """
        Map the names of cubes and coordinates to a consistent scheme.

        Dictionaries stored in self.field_map and self.crd_map
        store the name changes and for field and coordinates
        respectively. This method applies those changes to all
        cubes and all coordinates in those fields.

        Note that this should be done before the unit conversions
        are done.
        """
        msg = "source::map_names()"
        if self.cubes is not None:
            if len(self.cubes) > 0:
                if self.field_map is not None:
                    for c in self.cubes:
                        if c.name() in self.field_map:
                            c.rename(self.field_map[c.name()])
                if self.crd_map is not None:
                    for c in self.cubes:
                        for x in c.coords():
                            if x.name() in self.crd_map:
                                x.rename(self.crd_map[x.name()])
            else:
                logging.warning(msg + " cannot process an empty cube list")
        else:
            logging.warning(msg + " no cube list to process")
        return

    def regrid(self):
        """
        Regrid all cubes

        If self.target_grid has been set, use it to regrid all
        available self.cubes. Note that this step is intended
        only to regrid the latitude and longitude coordinates.

        Note that the choice of regridding algorithm provided
        by iris.analysis is set here.
        """
        msg = "source::regrid()"
        if self.cubes is not None:
            if len(self.cubes) > 0:
                if self.target_grid is not None:
                    regridded = CubeList()
                    for c in self.cubes:
                        g = c.regrid(self.target_grid, iris.analysis.AreaWeighted())
                        regridded.append(g)
                    self.cubes = regridded
            else:
                logging.warning(msg + " cannot process an empty cube list")
        else:
            logging.warning(msg + " no cube list to process")
        return

    def convert_grid(self):
        """
        Convert the units in the target grid's coordinates

        This is the same conversion as has been, or should have been,
        applied to each of the stored cubes.
        """
        msg = "source::convert_grid()"
        if self.target_grid is not None:
            self.converted_grid = copy.deepcopy(self.target_grid)
            if self.unit_map is not None:
                for x in self.converted_grid.coords(dim_coords=True):
                    if x.name() in self.unit_map:
                        x.convert_units(self.unit_map[x.name()])
            else:
                logging.info(msg + " no unit map so no grid conversion")
        else:
            msg += " no target grid to convert"
            logging.info(msg)
        return

    def convert_units(self):
        """
        Convert units for cubes and coordinates

        Use the convert_units() methods provided by Cube,
        iris.coords.DimCoord, and iris.cube.AuxCoord to convert all
        to a consistent and convenient set of units.
        """
        msg = "source::convert_units()"
        if self.cubes is not None:
            if len(self.cubes) > 0:
                if self.unit_map is not None:
                    for c in self.cubes:
                        if c.name() in self.unit_map:
                            c.convert_units(self.unit_map[c.name()])
                        for x in c.coords():
                            if x.name() in self.unit_map:
                                x.convert_units(self.unit_map[x.name()])
            else:
                logging.warning(msg + " cannot process an empty cube list")
        else:
            logging.warning(msg + " no cube list to process")
        return

    def process(self):
        """
        Process the raw download into the desired field.

        The raw download is converted into the desired form by this method,
        with the result stored in self.cubes. Most of the cube processing
        tasks are performed by other methods.
        """
        self.add_surface_coordinate_system()
        self.bounds()
        self.map_names()
        self.regrid()
        self.convert_units()
        self.clean()
        self.ensemble()
        self.coordinate_types()
        self.bounds()
        self.weight()
        return

    def clean(self):
        """
        Remove metadata not strictly needed.
        """
        msg = "source::clean()"
        if self.cubes is not None:
            if len(self.cubes) > 0:
                for c in self.cubes:
                    c.attributes = {}
                    for x in c.coords():
                        x.attributes = {}
            else:
                logging.warning(msg + " cannot clean an empty cube list")
        else:
            logging.warning(msg + " no cube list to clean")
        return

    def save(self):
        """
        Save the processed file.

        Store the processed cubes in the destination file,
        and store the modification time so that we can recreate
        downstream cubes if this processed output is updated.
        """
        iris.save(self.cubes, self.local_path.as_posix())
        self.mtime = os.path.getmtime(self.local_path.as_posix())
        self.cubes = None
        self.cubes = iris.load(self.local_path.as_posix())
        return

    def load(self):
        """
        Load the downloaded cubes before processing.

        First load the cubes without using the Iris constraint
        stored in self.selection, and then load with the constraint.
        The reason for this is that failure with a constraint is more
        common and more easily debugged if the result without the
        constraint is available.
        """
        self.cubes = load_with_retry(self.download,
                                     title="without selection")

        if self.selection is not None:
            self.cubes = load_with_retry(self.download,
                                         selection=self.selection,
                                         title="with selection")
        return

    def describe(self):
        """
        Print a description of the processed data.
        """
        print_cubes_with_title("processed cubes",
                               self.cubes,
                               include_crds=True)
        return

    def remove(self):
        """
        Remove any remaining files. This will delete
        both the temporary download file and the processed
        file. This should only be called if the data have
        been successfully incorporated into a more
        useful data structure.
        """
        msg = "source:remove() removing download and processed file"
        logging.info(msg)
        if self.download.is_file():
            self.download.unlink()
        if self.local_path.is_file():
            self.local_path.unlink()
        return

    def ensemble(self):
        """
        Set up membership of an ensemble.
        """
        msg = "source:ensemble()"
        if self.cubes is not None:
            if len(self.cubes) > 0:
                msg +=  " setting member number in cubes"
                logging.info(msg)
                m = AuxCoord(self.member,
                             long_name='member',
                             var_name='member',
                             units='unknown')
                for c in self.cubes:
                    try:
                        mm = c.coord('member')
                    except CoordinateNotFoundError as e:
                        c.add_aux_coord(m)
                    else:
                        msg += " member coordinate already present"
                        logging.warning(msg)
            else:
                logging.warning(msg + " cannot process an empty cube list")
        else:
            logging.warning(msg + " no cube list to process")
        return

    def coordinate_types(self):
        """
        Set the types of coordinate points

        Make sure each coordinate uses points of the expected type.
        Use the floating point type specified as the default in .numbers.
        """
        msg = "source::coordinate_types()"
        if self.cubes is not None:
            if len(self.cubes) > 0:
                for c in self.cubes:
                    for k, x in enumerate(c.coords(dim_coords=True)):
                        pts = x.points.astype(_FP_TYPE)
                        crd = x.copy(points=pts)
                        if not crd.has_bounds():
                            crd.guess_bounds()
                        if crd.name() == 'longitude':
                            crd.circular = True
                        c.remove_coord(x.name())
                        c.add_dim_coord(crd, k)
            else:
                logging.warning(msg + " cannot process an empty cube list")
        else:
            logging.warning(msg + " no cube list to process")
        return

    def weight(self):
        """
        Add an area weight auxiliary coordinate
        to each cube.
        """
        msg = "source::weight()"
        if self.cubes is not None:
            if len(self.cubes) > 0:
                msg += " add area weights to cubes"
                logging.info(msg)
                for c in self.cubes:
                    wt = get_area_wt(c)
                    try:
                        latdims = c.coord_dims('latitude') 
                        londims = c.coord_dims('longitude')
                    except CoordinateNotFoundError as e:
                        logging.error(msg + "latitude or longitude not found")
                        logging.exception(e)
                        raise
                    dims = latdims + londims
                    c.add_aux_coord(wt, data_dims=dims)
            else:
                logging.warning(msg + " cannot process an empty cube list")
        else:
            logging.warning(msg + " no cube list to process")
        return

    @property
    def tag(self) -> str:
        """
        Make a tag to go in the file name

        Make a string that describes the settings used in
        construction of the processed cubes.

        Returns
        -------
        String for inclusion in file name.
        """
        s = "m" + str(self.member).zfill(_MEMBER_ZERO_PAD)
        if self.transform:
            if self.pack:
                s += ".pack"
        if self.transform and not self.pack:
            s += "." + _COMPLEX_FP_TYPE_STRING
        else:
            s += "." + _FP_TYPE_STRING
        return s

    @property
    def fmt(self):
        """
        Provide the extension that specifies the file type.
        """
        return '.nc'

    def set_output_file(self):
        """
        Check and update the file name and type for processed data

        Ensure that the destination file, self.local_path, for
        the processed data is of the correct type.
        If self.local_path already contains self.tag, then
        leave it unchanged. Otherwise, add the tag.
        """
        msg = "source::check_file_type()"
        lpfmt = self.local_path.suffix
        if not lpfmt == self.fmt:
            msg += " expected extension " + self.fmt
            logging.error(msg)
            raise RuntimeError(msg)
        lp = self.local_path
        if self.tag not in lp.as_posix():
            self.local_path = lp.with_suffix("." + self.tag + self.fmt)
        return

    def __init__(self,
                 local_path: str | Path,
                 transform: bool = True,
                 pack: bool = True,
                 selection: Optional[Constraint] = None,
                 target_grid: Optional[Cube] = None,
                 member: int = 0,
                 tmp: str | Path = _TMP_FOLDER,
                 keep: bool = False,
                 garbage_collect: bool = True,
                 field_map: dict[str, str] = _DEFAULT_FIELD_MAP,
                 crd_map: dict[str, str] = _DEFAULT_CRD_MAP,
                 unit_map: dict[str, str] = _DEFAULT_UNIT_MAP):
        """
        Download a file containing weather or climate
        data, select variables and regions of interest,
        regrid, and process the data.
        """
        logging.info("source::__init__() checking for " + local_path)
        if selection is not None:
            logging.info("source::__init__() selection is set")
        else:
            logging.info("source::__init__() selection is not set")

        # Save the details of the download
        self.transform: bool = transform
        self.pack: bool = pack
        self.selection: Optional[Constraint] = copy.deepcopy(selection)
        self.target_grid: Optional[Cube] = copy.deepcopy(target_grid)
        self.converted_grid: Optional[Cube] = None
        self.member: int = member
        self.tmp: Path = Path(tmp)
        self.gc: bool = garbage_collect
        self.field_map: dict[str, str] = copy.deepcopy(field_map)
        self.crd_map: dict[str, str] = copy.deepcopy(crd_map)
        self.unit_map: dict[str, str] = copy.deepcopy(unit_map)

        # Set, check, and save file that holds processed cubes
        self.local_path: Path = Path(local_path)
        self.set_output_file()
        self.all_files: list[Path] = [self.local_path]

        # Members to be set later
        self.download = None
        self._mtime = None

        # Make a grid with unit conversion applied
        self.convert_grid()

        # Have we already got the data?
        # Do not download again if we do.
        if self.ok():
            logging.info(self.local_path.as_posix() + " is ok()")
            self.cubes = iris.load(self.local_path.as_posix())
        else:
            logging.info("need to get() " + self.local_path.as_posix())
            try:
                # Discard the file that is not ok()
                if self.local_path.is_file():
                    self.local_path.unlink()
                # Ensure temporary download folder exists
                self.tmp.mkdir(parents=True, exist_ok=True)
                # Attempt a download
                self.download = self.get()
                # Check download is usable
                if not fields_are_ok(self.download):
                    raise RuntimeError("get() produced broken file")
            except RuntimeError as e:
                msg = "source::__init__() exception while downloading "
                msg += self.local_path.as_posix()
                logging.error(msg)
                logging.exception(e)
                raise

            self.load()

            try:
                self.save_raw_cubes = copy.deepcopy(self.cubes)
                self.process()
            except RuntimeError as e:
                msg = "source::__init__() failed to process "
                msg += self.local_path.as_posix()
                logging.error(msg)
                logging.exception(e)
                raise
            else:
                try:
                    self.save()
                except RuntimeError as e:
                    msg = "source::__init__() failed to save "
                    msg += self.local_path.as_posix()
                    logging.error(msg)
                    logging.exception(e)
                    raise
                else:
                    if self.ok():
                        msg = "source::__init__()"
                        msg += " processed file ok()"
                        logging.info(msg)
                        if not keep:
                            msg = "source::__init__() removing download"
                            logging.info(msg)
                            self.download.unlink()
                    else:
                        msg = "source::__init__()"
                        msg += " processed file not ok()"
                        logging.error(msg)
                        raise RuntimeError(msg)
                    return
