"""
Combine fields into useful structures
"""
import logging
import os
import gc

import numpy
import iris

from pathlib import Path
from typing import Union, Optional, Callable
from datetime import datetime, timedelta
from copy import deepcopy

from numpy import ndarray
from scipy.constants import pi

from iris import Constraint
from iris.cube import Cube,CubeList
from iris.coords import DimCoord,AuxCoord

from .useful import _RULE_WIDTH, cube_present, assemble
from .useful import cname, stamp, parts, update_file
from .useful import fields_are_ok, frequency_coordinate
from .useful import is_my_real_type, is_my_complex_type, rnd

from .numbers import _FP_TYPE, _COMPLEX_FP_TYPE
from .numbers import _FP_DOUBLE_TYPE, _COMPLEX_FP_DOUBLE_TYPE
from .numbers import _RELATIVE_TOLERANCE
from .numbers import _DAY, _TIME_UNIT, _TIME_UNIT_ONLY_STRING
from .numbers import _TIME_UNIT_STRING, _TIME_UNIT_START
from .numbers import _YEAR, _YEAR_START, _BEFORE_LEAP_DAY
from .numbers import _LEAP_DAY, _MARCH_START, _AFTER_LEAP_DAY

from .grids import _AREA_WEIGHT_COORDINATE

from .source import source
from .field import field

from .field import _DEFAULT_SCALE_REGION, _DEFAULT_WORKERS, _FFT_NORM
from .field import _N_EVALUE_CHECK, _SPECTRUM, _CORRECTION
from .field import _CLIMATE_MEAN, _CLIMATE_MS, _ANOMALY, _SMOOTH, _SCALED
from .field import _REGION_MEAN, _REGION_SPREAD, _STATIONARY
from .field import _REGION_NC, _CLIMATE_NC, _PROCESSED_NC, _COVARIANCE_NC
from .field import _EIGEN_NC, _ZETA_NC, _DUMP, _EXTRA
from .field import _FIGURE_SUBFOLDER, _MODE_SUBFOLDER, _TRAIN, _TEST

class train(field):
    """
    Train a prediction on a selection of the data.
    """

    def __init__(self,
                 local_path: str | Path,
                 src: Optional[list[source]] = None,
                 selection: Optional[iris.Constraint] = None,
                 clean: bool = False,
                 garbage_collect: bool = True):
        """
        Pass through to base class constructor.
        """
        field.__init__(self,
                       local_path,
                       src,
                       selection,
                       clean,
                       garbage_collect)
        return

class test(field):
    """
    Test a prediction on data that does not overlap with the test set.
    """

    def __init__(self,
                 local_path,
                 trainer: train,
                 selection: Optional[iris.Constraint] = None,
                 debug: bool = False,
                 clean: bool = False):
        """
        Test the model set up by the trainer

        The data used for testing is the complement of
        the training data used by the trainer.
        """
        self.trainer = trainer
        # TODO Fix this when numpy docs available!
        mask = deepcopy(self.trainer.split_mask)
        if not debug: mask = numpy.logical_not(mask)
        field.__init__(self,
                       local_path,
                       self.trainer.src,
                       selection,
                       clean,
                       self.trainer.gc)
        self.split(mask)
        self.choose_seasons(self.trainer.season_mask)
        self.mode_figure_folder.rmdir()
        return


    def load_scale(self, member:int = 0):
        field.load_scale(self, member=member, region_path=self.trainer.region_path)
        return

    def load_climate(self, member:int = 0):
        field.load_climate(self, member=member, climate_path=self.trainer.climate_path)
        return

    def mask_ok(self) -> bool:
        """
        Make sure the test mask is the complement of the train

        The test class in debug mode has the same mask as the corresponding
        training class. This routine makes sure that predictions are not
        made in debug mode.

        Return
        ------
        True if the masks are complementary as expected, False otherwise.
        """
        msg = 'test:check_mask()'
        logging.info(msg + ' checking masks are complementary')

        train_mask = self.trainer.split_mask
        mask = self.split_mask
        xor = numpy.logical_xor(mask, train_mask)
        if not numpy.all(xor):
            msg += ' masks not complementary at '
            locations = numpy.nonzero(numpy.logical_not(xor))[0]
            msg += str(len(locations))
            msg += ' locations'
            logging.warning(msg)
            return False

        msg += ' masks ok'
        logging.info(msg)
        return True

    def modes(self,
              member:int = 0,
              sigma:int = 4,
              delta: timedelta = _YEAR,
              reciprocal:str ='omega'):
        """
        Remove the Fourier modes corresponding to seasonal variation.

        This method of calculation of anomalies is similar to the one
        used by Albers and Newman in their Linear Inverse Model
        (See DOI 10.1029/2019GL085270). The zero frequency and lowest
        sigma frequencies commensurate with the given year length are
        removed.

        Here the modes calculated for the training set are subtracted.

        Parameters
        ----------
        member:int = 0
            Member to operate on. Normally operations are performed
            on all members, but this can be useful to implement
            parallel computation.
        sigma:int = 4
            The number of nonzero seasonal modes to remove. Both
            positive and negative modes are removed, along with
            the zero frequency, so the total number of modes
            removed is 2 * sigma + 1.
        delta:timedelta = _YEAR
            The length of a year, usually set to 365.25 days.
            It is unlikely the length is needed more precisely than this,
            but in some older models a year is 360 = 12 * 30 days.
        """
        msg = "test:modes() remove seasonal modes from "
        msg += self.local_path.as_posix()
        logging.info(msg)
        msg = "test:modes()"

        climate_mode_parts = iris.load(self.trainer.climate_path)
        all_climate_modes = assemble(climate_mode_parts)

        self.climate_modes = CubeList()

        season = []
        w = 2 * pi / (_YEAR / _TIME_UNIT)
        self.transform(real='time', reciprocal=reciprocal)
        for f,c in zip(self.ft, self.ensemble[member]):

            count = 0
            for cc in all_climate_modes:
                if c.name() in cc.name():
                    if _CLIMATE_MEAN + _SPECTRUM in cc.name():
                        count += 1
                        rc = cc
            if count == 0:
                msg += " could not find climate mean spectrum for " + c.name()
                logging.error(msg)
                raise RuntimeError(msg)
            elif count > 1:
                msg += " multiple climate mean spectrum entries for " + c.name()
                logging.error(msg)
                raise RuntimeError(msg)

            omega = c.coord(reciprocal).points
            square = numpy.fabs(omega - w * numpy.ones_like(omega))**2
            season.append(numpy.argmin(square))
            shp = numpy.shape(c.data)
            pts = numpy.zeros(2*sigma+1, dtype=omega.dtype)
            removed = rc.data
            for k in range(sigma+1):
                # pts[k] = k * omega[season[-1]]
                # removed[k,...] = f[k*season[-1],...]
                f[k*season[-1],...] -= removed[k,...]
                if k!=0:
                    # pts[-k] = k * omega[season[-1]]
                    # removed[-k,...] = f[-k*season[-1],...]
                    f[-k*season[-1],...] -= removed[-k,...]
            c.rename(c.name()+_ANOMALY)
            self.climate_modes.append(rc)

        self.inverse(real='time', reciprocal=reciprocal)

        for s in season[1:]:
            if s != season[0]:
                msg += " seasonal modes not in the same place"
                logging.error(msg)
                raise RuntimeError(msg)

        self.record({
                        'step': 'modes',
                        'sigma': sigma,
                        'delta': str(delta/_DAY),
                        'reciprocal': reciprocal,
                        'member': member
                    })
        logging.info(msg + " seasonal modes removed")
        return

    def scale(self,
              member:int = 0,
              region:list[str] = _DEFAULT_SCALE_REGION):
        """
        Make sure the variance of each dynamical variable
        is of manageable magnitude.
        """
        msg = "test::scale()"
        logging.info(msg + " scale across each region")

        self.order()

        self.scale_mn = CubeList()
        self.scale_sd = CubeList()

        cubes = iris.load(self.trainer.region_path)

        for c in self.ensemble[member]:

            count = 0
            for cc in cubes:
                if c.name() in cc.name():
                    if _REGION_MEAN in cc.name():
                        count += 1
                        mn = cc
            if count == 0:
                msg += " could not find regional mean for " + c.name()
                logging.error(msg)
                raise RuntimeError(msg)
            elif count > 1:
                msg += " multiple regional mean entries for " + c.name()
                logging.error(msg)
                raise RuntimeError(msg)

            count = 0
            for cc in cubes:
                if c.name() in cc.name():
                    if _REGION_SPREAD in cc.name():
                        count += 1
                        std = cc
            if count == 0:
                msg += " count not find regional spread for " + c.name()
                logging.error(msg)
                raise RuntimeError(msg)
            elif count > 1:
                msg += " multiple regional spread entries for " + c.name()
                logging.error(msg)
                raise RuntimeError(msg)

            self.scale_mn.append(mn)
            self.scale_sd.append(std)

            c.data = c.data / std.data
            numpy.ma.fix_invalid(c.data, fill_value=0.0)

            c.rename(c.name()+_SCALED)

        self.record({
                        'step': 'scale',
                        'region': region,
                        'member': member
                    })
        logging.info(msg + " trajectory scaled")
        return

    def stationary(self,
                   member:int = 0,
                   roll:int = 42,
                   apply:bool = True):
        """
        Make sure the variance of each dynamical variable
        is of manageable magnitude.
        """
        msg = "field::stationary()"
        logging.info(msg + " make each variable stationary and save climate")

        self.order()

        self.climate_mn = CubeList()
        self.climate_ms = CubeList()

        pts = numpy.linspace(0,
                             rnd(_YEAR / _DAY),
                             int(rnd(_YEAR / _DAY)),
                             endpoint=False,
                             dtype=_FP_TYPE)
        climate_time = iris.coords.DimCoord(pts,
                                            standard_name='time',
                                            long_name='time',
                                            var_name='time',
                                            units=_TIME_UNIT_STRING)

        self.climate_mn = CubeList()
        self.climate_ms = CubeList()

        cubes = iris.load(self.trainer.climate_path)

        for c in self.ensemble[member]:

            count = 0
            for cc in cubes:
                if c.name() in cc.name():
                    if _CLIMATE_MEAN + _CORRECTION in cc.name():
                        count += 1
                        mn = cc
            if count == 0:
                msg += " count not find climate mean correction for " + c.name()
                logging.error(msg)
                raise RuntimeError(msg)
            elif count > 1:
                msg += " multiple climate mean correction entries for " + c.name()
                logging.error(msg)
                raise RuntimeError(msg)

            count = 0
            for cc in cubes:
                if c.name() in cc.name():
                    if _CLIMATE_MS in cc.name():
                        count += 1
                        ms = cc
            if count == 0:
                msg += " count not find climate mean correction for " + c.name()
                logging.error(msg)
                raise RuntimeError(msg)
            elif count > 1:
                msg += " multiple climate mean correction entries for " + c.name()
                logging.error(msg)
                raise RuntimeError(msg)

            self.climate_mn.append(mn)
            self.climate_ms.append(ms)

            for k, pt in enumerate(c.coord('time').points):
                t = _TIME_UNIT_START + pt * _TIME_UNIT
                if t == _YEAR_START:
                    spread = ms.data[:_BEFORE_LEAP_DAY]
                    logging.warning(msg + " spread**2    max = " + str(numpy.amax(spread)))
                    logging.warning(msg + " spread**2 argmax = " + str(numpy.argmax(spread)))
                    logging.warning(msg + " spread**2    min = " + str(numpy.amin(spread)))
                    logging.warning(msg + " spread**2 argmin = " + str(numpy.argmin(spread)))
                    spread = numpy.sqrt(spread)
                    if apply:
                        c.data[k:k+_BEFORE_LEAP_DAY] -= mn.data[:_BEFORE_LEAP_DAY]
                        c.data[k:k+_BEFORE_LEAP_DAY] /= spread
                if t == _LEAP_DAY:
                    spread = ms.data[_BEFORE_LEAP_DAY-1]
                    logging.warning(msg + " spread**2    max = " + str(numpy.amax(spread)))
                    logging.warning(msg + " spread**2 argmax = " + str(numpy.argmax(spread)))
                    logging.warning(msg + " spread**2    min = " + str(numpy.amin(spread)))
                    logging.warning(msg + " spread**2 argmin = " + str(numpy.argmin(spread)))
                    spread = numpy.sqrt(spread)
                    if apply:
                        c.data[k] -= mn.data[_BEFORE_LEAP_DAY-1]
                        c.data[k] /= spread
                if t == _MARCH_START:
                    spread = ms.data[-_AFTER_LEAP_DAY:]
                    logging.warning(msg + " spread**2    max = " + str(numpy.amax(spread)))
                    logging.warning(msg + " spread**2 argmax = " + str(numpy.argmax(spread)))
                    logging.warning(msg + " spread**2    min = " + str(numpy.amin(spread)))
                    logging.warning(msg + " spread**2 argmin = " + str(numpy.argmin(spread)))
                    spread = numpy.sqrt(spread)
                    if apply:
                        c.data[k:k+_AFTER_LEAP_DAY] -= mn.data[-_AFTER_LEAP_DAY:]
                        c.data[k:k+_AFTER_LEAP_DAY] /= spread

            if apply: c.rename(c.name()+_STATIONARY)

        action = {
                    'step': 'stationary',
                    'roll': roll,
                    'apply': apply,
                    'member': member
                 }
        self.record(action)
        if apply:
            logging.info(msg + " stationary scaling applied")
        else:
            logging.info(msg + " stationary scaling calculated but not applied")
        return

    def eigensystem(self, count:int = None):
        """
        Calculate the eigensystem of the covariance matrix.
        """
        msg = "field:eigensystem()"
        logging.info(msg + " calculate eigenvalues and eigenvectors")

        if self.mtime is None:
            msg += " make sure field is saved and ok()"
            logging.error(msg)
            raise RuntimeError(msg)

        eigenpath = self.trainer.eigen_path
        if eigenpath.is_file():
            eigenmtime = os.path.getmtime(eigenpath.as_posix())
            if self.trainer.processed_mtime > eigenmtime:
                msg += " need to update eigensystem " + str(eigenmtime)
                logging.error(msg)
                raise RuntimeError(msg)
            else:
                logging.info(msg + " loading eigensystem from disk " + str(eigenmtime))

                cube = iris.load_cube(eigenpath.as_posix(), "eigenvector")
                if numpy.ma.isMaskedArray(cube.data):
                    self.eigenvectors = cube.data.data
                else:
                    self.eigenvectors = cube.data
                (nt, self.count) = self.eigenvectors.shape

                cube = iris.load_cube(eigenpath.as_posix(), "eigenvalue")
                if numpy.ma.isMaskedArray(cube.data):
                    self.all_eigenvalues = cube.data.data
                else:
                    self.all_eigenvalues = cube.data
                self.eigenvalues = self.all_eigenvalues[:self.count]

                return

        self.record({
                        'step': 'eigensystem',
                        'count': count
                    })
        logging.info(msg + " eigensystem loaded")

        return
