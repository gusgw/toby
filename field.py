"""
Combine fields into useful structures
"""
import logging
import os
import gc

import numpy
import json
import pickle
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import seaborn as sns

import iris
import iris.plot as irisplot
import iris.quickplot as quickplot

from pathlib import Path
from typing import Union, Optional, Callable
from datetime import datetime, timedelta
from copy import deepcopy

from matplotlib.axes import Axes
from numpy import ndarray
from scipy.constants import pi
from scipy.fft import fft, ifft, fftfreq
from scipy.linalg import svd, eigh, LinAlgError

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

from .source import source, _ZULU_POINT

_DEFAULT_SCALE_REGION = ['time', 'latitude', 'longitude']

_DEFAULT_WORKERS = 4
_FFT_NORM = "backward"
_N_EVALUE_CHECK = 10

_SPECTRUM = '_spectrum'
_CORRECTION = '_correction'
_CLIMATE_MEAN = '_climate_mn'
_CLIMATE_MS = '_climate_ms'
_ANOMALY = '_anomaly'
_SMOOTH = '_smooth'
_SCALED = '_scaled'
_REGION_MEAN = '_region_mn'
_REGION_SPREAD = '_region_sd'

_STATIONARY = '_stationary'

_REGION_NC = ".region.nc"
_CLIMATE_NC = ".climate.nc"
_PROCESSED_NC = ".processed.nc"
_COVARIANCE_NC = ".covariance.nc"
_EIGEN_NC = ".eigen.nc"
_ZETA_NC = ".zeta.nc"

_DUMP = ".json"
_EXTRA = ".extra.pickle"

_FIGURE_SUBFOLDER = 'figure'
_MODE_SUBFOLDER = 'eof'

_TRAIN = 'train'
_TEST = 'test'

class field:
    """
    Individual downloads with data cleaned and
    transformed are obtained from classes
    derived from source. Here these
    are combined into a single class.
    """

    def ok(self, test_path: Optional[str | Path] = None) -> bool:
        """
        Check that a file is usable and up to date.

        By default the file checked is self.local_path,
        which contains the main data set, cubes of trajectories,
        on which the calculation is based. Another file may
        be specified in which case that is checked. Either way,
        the file fails if any of the data sources have a later
        modification time. If test_path is set, then that file
        must have a later modification time than the main data
        file containing the trajectory as cubes.

        Parameters
        ----------
        test_path: Optional[str | Path] = None
            If desired, the file to be tested. This could be
            for example the file containing the covariance or
            the eigensystem, self.covariance_path or
            self.eigen_path respectively.

        Returns
        -------
        True if the tested file passes, False otherwise.
        """
        if test_path is None:
            lp = self.local_path
        else:
            lp = Path(test_path)
        msg = "field:ok()"
        logging.info(msg + " checking " + lp.as_posix())

        if lp.is_file():
            mtime = os.path.getmtime(lp.as_posix())
            if test_path is None:
                self.mtime = mtime
            else:
                if self.mtime > mtime:
                    msg += " " + lp.as_posix() + " not up to date"
                    logging.warning(msg)
                    return False
            for s in self.src:
                if s.mtime > mtime:
                    msg += " " + lp.as_posix() + " not up to date"
                    logging.warning(msg)
                    return False
            test = iris.load(lp.as_posix())
            for s in self.src:
                for c in s.cubes:
                    if not cube_present(c.name(), test):
                        msg += " " + lp.as_posix() + " is missing cube "
                        msg += c.name()
                        logging.warning(msg)
                        return False
            if fields_are_ok(lp.as_posix()):
                logging.info(msg + " " + lp.as_posix() + " is ok()")
                return True
            else:
                logging.warning(msg + " " + lp.as_posix() + " exists but fields fail")
                return False
        else:
            logging.info(msg + " " + lp.as_posix() + " not found")
            return False

    def transform(self,
                  member:int = 0,
                  real:str = 'time',
                  reciprocal:str = 'omega'):
        """
        Apply a Fourier transform to the selected coordinate.

        Specify both the name of the coordinate and give a new
        name for the corresponding angular frequency. 

        NB: This routine stores the Fourier transform in
        a separate array but stores the frequency
        coordinate as an auxiliary coordinate in the
        original cube.

        Parameters
        ----------
        member:int = 0
            Specify member to transform. It's unlikely that we
            would want to transform one member and not others.
            This might be useful in transforming in parallel.
        real:str = 'time'
            Specify coordinate to be transformed. Usually the
            'time' coordinate corresponds to the first
            dimension.
        reciprocal:str = 'omega'
            An auxiliary frequency coordinate with this name
            will be added to each cube.
        """
        msg = "field:transform() calculating FFT over " + real
        msg += " for " + self.local_path.as_posix()
        logging.info(msg)
        msg = "field:transform()"

        self.ft = []
        for c in self.ensemble[member]:
            crds = c.coords(dim_coords=True)
            found = False
            for j, tt in enumerate(crds):
                if tt.name() == real:
                    found = True
                    ax = j
                    t = tt
                    break
            if not found:
                msg += real + " coordinate not found for "
                msg += c.name()
                logging.error(msg)
                raise RuntimeError(msg)
            if reciprocal in list(map(cname, c.coords(dimensions=ax))):
                logging.info(msg + " already transformed, skipping ")
                return
            self.ft.append(fft(c.data.data,
                               axis=ax,
                               norm=_FFT_NORM,
                               overwrite_x=False,
                               workers=_DEFAULT_WORKERS))
            omega = frequency_coordinate(t, omega=reciprocal)
            c.add_aux_coord(omega, data_dims=ax)

        self.record({
                        'step': 'transform',
                        'real': real,
                        'reciprocal': reciprocal,
                        'member': member
                    })
        logging.info(msg + " fft over " + real + " done")
        return

    def inverse(self,
                member:int = 0,
                real:str = 'time',
                reciprocal:str = 'omega'):
        """
        Apply the inverse Fourier transform,
        to the time coordinate by default.

        Specify both the name of the other coordinate and
        give a new name for the corresponding angular frequency.
        When inverting the transform the angular frequency
        coordinate name is the coordinate removed from each cube.

        NB: This routine stores the Fourier transform in
        a separate array but stores the frequency
        coordinate as an auxiliary coordinate in the
        original cube.

        Parameters
        ----------
        member:int = 0
            Specify member to transform. It's unlikely that we
            would want to transform one member and not others.
            This might be useful in transforming in parallel.
        real:str = 'time'
            Specify coordinate to be transformed. Usually the
            'time' coordinate corresponds to the first
            dimension.
        reciprocal:str = 'omega'
            An auxiliary frequency coordinate with this name
            will be added to each cube.
        """
        msg = "field:inverse() calculating inverse FFT over " + real
        msg += " for " + self.local_path.as_posix()
        logging.info(msg)
        msg = "field:inverse()"

        for f,c in zip(self.ft, self.ensemble[member]):
            crds = c.coords(dim_coords=True)
            found = False
            for j, tt in enumerate(crds):
                if tt.name() == real:
                    found = True
                    ax = j
                    t = tt
                    break
            if not found:
                msg = real + " coordinate not found for "
                msg += c.name()
                logging.error(msg)
                raise RuntimeError(msg)
            if reciprocal not in list(map(cname, c.coords(dimensions=ax))):
                logging.info("inverse not needed")
                return
            cdata = numpy.real(ifft(f,
                                    axis=ax,
                                    norm=_FFT_NORM,
                                    overwrite_x=True,
                                    workers=_DEFAULT_WORKERS))
            c.data = numpy.ma.array(cdata,
                                    mask=c.data.mask,
                                    fill_value=0.0,
                                    hard_mask=True)
            c.remove_coord(reciprocal)
        self.ft = []

        self.record({
                        'step': 'inverse',
                        'real': real,
                        'reciprocal': reciprocal,
                        'member': member
                    })
        logging.info(msg + " inverse fft over " + real + " done")
        return

    def crdidx(self, name, member: int = 0):
        """
        Find the array indices that correspond
        to a particular coordinate for each cube
        in the given ensemble member.
        """
        msg = "field:crdidx() get data dimenssions for " + name
        logging.info(msg)

        idx = []
        for c in self.ensemble[member]:
            shp = c.shape
            ndim = len(shp)
            cidx = []
            for k in range(ndim):
                crds = c.coords(dimensions=k)
                cnames = list(map(cname, crds))
                if name in cnames:
                    cidx.append(k)
            if len(cidx) == 0:
                idx.append(None)
            elif len(cidx) == 1:
                idx.append(cidx[0])
            else:
                idx.append(cidx)
        return idx

    def check(self,
              member:int = 0,
              position:int = 0,
              crd:str = 'time',
              need:bool = True):
        """
        Check that a coordinate with name crd
        is in the expected position.
        """
        msg = "field:check() check position of " + crd
        logging.info(msg)
        msg = "field:check()"

        idx = self.crdidx(crd, member)
        for j, c in zip(idx, self.ensemble[member]):
            if position < 0:
                p = len(c.shape) + position
            else:
                p = position
            if j is None:
                msg = crd + ' coordinate missing from '
                msg += c.name()
                if need:
                    logging.error(msg)
                    raise RuntimeError(msg)
                else:
                    logging.warning(msg)
                    msg = "field:check()"
            elif p == j:
                msg += ' found coordinate '
                msg += crd + " "
                msg += " at " + str(position)
                msg += " in " + c.name()
                logging.info(msg)
                msg = "field:check()"
            elif p in j:
                msg += ' coordinate '
                msg += crd + " "
                msg += " includes dimension " + str(position)
                msg += " in " + c.name()
                logging.info(msg)
                msg = "field:check()"
            else:
                msg += ' coordinate '
                msg += crd + " "
                msg += " should be at " + str(position)
                msg += " in " + c.name()
                logging.error(msg)
                raise RuntimeError(msg)
        msg += ' done'
        logging.info(msg)
        return

    def order(self, member:int = 0):
        """
        Coordinates should be in a standard order.
        """
        msg = "field:order() checking order of coordinates"
        logging.info(msg)

        try:
            self.check(member, position=0, crd='time')
            self.check(member, position=-2, crd='latitude', need=False)
            self.check(member, position=-1, crd='longitude', need=False)
        except RuntimeError as e:
            msg = "problem with dimensions"
            logging.error(msg)
            raise

    def unpack(self, member: int = 0):
        """
        Create a single time-dependent vector

        Take the data arrays in each cube in the CubeList
        for the given member and unravel them all into one
        big vector, self.xi, and multiply by the square root of the
        measure of each cell. This makes the ordinary dot
        product of these vectors an approximation to the
        area integral.

        Parameters
        ----------
        member: int = 0
            Specify the member to unpack.
        """
        msg = "field:unpack()"
        logging.info(msg + " converting cubes to simple vectors")

        self.order(member)
        ncubes = len(self.ensemble[member])
        dt = []
        nt = []
        self.sizes = []
        self.shapes = []
        for c in self.ensemble[member]:
            shp = c.shape
            time = c.coord('time')
            if len(time.points) != shp[0]:
                msg += 'error in time coordinate in '
                msg += c.name()
                logging.error(msg)
                raise RuntimeError(msg)
            mn = numpy.mean(time.points[1:] - time.points[:-1])
            std = numpy.std(time.points[1:] - time.points[:-1])
            if std / mn > _RELATIVE_TOLERANCE:
                msg += 'time spacing not regular in '
                msg += c.name()
                logging.error(msg)
                raise RuntimeError(msg)
            nt.append(shp[0])
            dt.append(mn)
            sz = 1
            for n in shp[1:]:
                sz *= n
            self.sizes.append(sz)
            self.shapes.append(shp)

        dt = numpy.array(dt)
        mn = numpy.mean(dt)
        std = numpy.std(dt)
        if std / mn > _RELATIVE_TOLERANCE:
            msg += ' time spacing not the same in all cubes'
            logging.error(msg)
            raise RuntimeError(msg)

        c0 = self.ensemble[member][0]
        for n, c in zip(nt[1:], self.ensemble[member][1:]):
            if c0.dtype != c.dtype:
                msg += ' cubes not all the same type '
                msg += str(c0.dtype) + ' '
                msg += str(c.dtype)
                logging.error(msg)
                raise RuntimeError(msg)
            if n != nt[0]:
                msg += ' different number of time points in '
                msg += self.ensemble[member][0].name()
                msg += ' and '
                msg += c.name()
                logging.error(msg)
                raise RuntimeError(msg)
        n = nt[0]

        if c0.dtype == _FP_TYPE:
            logging.info(msg + " using real variables " + str(_FP_TYPE))
        elif c0.dtype == _COMPLEX_FP_TYPE:
            logging.info(msg + " using complex variables " + str(_COMPLEX_FP_TYPE))
        else:
            msg += ' variable for unpacking has unexpected type ' + str(c0.dtype)
            logging.error(msg)
            raise RuntimeError(msg)

        total = 0
        for j, sz in enumerate(self.sizes):
            logging.info(msg + " size number " + str(j) + " is " + str(sz))
            total += sz
        logging.info(msg + " total vector size for xi(t) is " + str(total))

        start = 0
        self.xi = numpy.ma.zeros((n, total), dtype=c0.dtype)
        for sz, c in zip(self.sizes, self.ensemble[member]):
            # block = c.data * numpy.sqrt(c.coord(_AREA_WEIGHT_COORDINATE).points)
            block = numpy.zeros_like(c.data)
            lat = c.coord('latitude').points
            lon = c.coord('longitude').points
            nlat = len(lat)
            nlon = len(lon)
            for i in range(nlat):
                for j in range(nlon):
                    block[...,i,j] = c.data[...,i,j] * numpy.sqrt(numpy.cos(lat[i]))
            self.xi[:, start:start+sz] = block.reshape((n, sz))
            start += sz

        self.record({
                        'step': 'unpack',
                        'member': member
                    })
        logging.info(msg + " unpack done")
        return

    def pack(self,
             x:Optional[ndarray] = None,
             trajectory:bool = True,
             member:int = 0):
        """
        Reconstruct cubes from time dependent vector

        Either pack the time dependent vectors in self.xi
        into the main list of CubeLists stored as self.ensemble,
        or if x is specified, pack x into a new CubeList with
        the same structure as the ones in self.ensemble.

        Parameters
        ----------
        x: Optional[ndarray] = None
            If ndarray is provided, then pack x into a new
            CubeList. If not, work on self.xi.
        trajectory: bool = True
            If x is set, this determines whether or not a
            time coordinate should be included. For example
            when unpacking an eigenvector to plot 'EOFs'
            there should be no time coordinate.
        member: int = 0
            Specify the member to unpack.
        """
        msg = 'field::pack()'
        logging.info(msg + ' put simple vector back in cubes')
        self.order(member)

        c0 = self.ensemble[member][0]

        for c in self.ensemble[member][1:]:
            if c0.dtype != c.dtype:
                msg += ' cubes not all the same type '
                msg += str(c0.dtype) + ' '
                msg += str(c.dtype)
                logging.error(msg)
                raise RuntimeError(msg)

        if x is None:
            x = self.xi
            cubes = self.ensemble[member]
        else:
            if trajectory:
                cubes = deepcopy(self.ensemble[member])
            else:
                cubes = iris.cube.CubeList()
                for c in self.ensemble[member]:
                    cubes.append(deepcopy(c[0]))

        if is_my_real_type(c0) and is_my_real_type(x):
            logging.info(msg + " packing vector into real cubes")
        elif is_my_complex_type(c0) and is_my_complex_type(x):
            logging.info(msg + " packing vector into complex cubes")
        else:
            msg += ' cube ' + str(c0.dtype) 
            msg += ' and vector ' + str(x.dtype)
            msg += ' should match'
            logging.error(msg)
            raise RuntimeError(msg)

        start = 0
        for shp, c in zip(self.shapes, cubes):
            sz = 1
            for n in shp[1:]:
                sz *= n
            if trajectory:
                data = x[:,start:start+sz].astype(c0.dtype)
                c.data = data.reshape(shp)
            else:
                data = x[start:start+sz].astype(c0.dtype)
                c.data = data.reshape(shp[1:])
            c.data /= numpy.sqrt(c.coord(_AREA_WEIGHT_COORDINATE).points)
            start += sz

        cubes.sort(key=cname)
        action = {
                    'step': 'pack',
                    'member': member,
                    'trajectory': trajectory,
                    'xi': x is None
                 }
        self.record(action)
        logging.info(msg + " pack done")
        return cubes

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
        delta: timedelta = _YEAR
            The length of a year, usually set to 365.25 days.
            It is unlikely the length is needed more precisely than this,
            but in some older models a year is 360 = 12 * 30 days.
        reciprocal:str ='omega'
            Set the name of the angular frequency coordinate.This
            is used to override the default used by the transform()
            and inverse() routines, and the coordinate should be
            removed by the inverse() routine, so the choice of this
            name is not important. 
        """
        msg = "field:modes() remove seasonal modes from "
        msg += self.local_path.as_posix()
        logging.info(msg)
        msg = "field:modes()"

        season = []
        self.climate_modes = CubeList()
        w = 2 * pi / (_YEAR / _TIME_UNIT)
        self.transform(real='time', reciprocal=reciprocal)
        for f,c in zip(self.ft, self.ensemble[member]):
            omega = c.coord(reciprocal).points
            square = numpy.fabs(omega - w * numpy.ones_like(omega))**2
            season.append(numpy.argmin(square))
            shp = numpy.shape(c.data)
            pts = numpy.zeros(2*sigma+1, dtype=omega.dtype)
            removed = numpy.zeros((2*sigma+1,) + shp[1:], dtype=f.dtype)
            for k in range(sigma+1):
                pts[k] = k * omega[season[-1]]
                removed[k,...] = f[k*season[-1],...]
                f[k*season[-1],...] *= 0.0
                if k!=0:
                    pts[-k] = k * omega[season[-1]]
                    removed[-k,...] = f[-k*season[-1],...]
                    f[-k*season[-1],...] *= 0.0
            c.rename(c.name()+_ANOMALY)
            seasonal_index = DimCoord(numpy.array(range(2*sigma+1)),
                                      long_name='idx',
                                      var_name='idx')
            seasonal_omega = AuxCoord(pts,
                                      long_name='omega',
                                      var_name='omega')
            crds = [(seasonal_index, 0)]
            for k,x in enumerate(c.coords(dim_coords=True)[1:]):
                crds.append((x, k+1))
            rcube = Cube(removed,
                         long_name=c.name()+_CLIMATE_MEAN+_SPECTRUM,
                         var_name=c.name()+_CLIMATE_MEAN+_SPECTRUM,
                         dim_coords_and_dims=crds)
            rcube.add_aux_coord(seasonal_omega, 0)
            self.climate_modes.append(rcube)
        climate_mode_parts = CubeList()
        for c in self.climate_modes:
            climate_mode_parts += parts(c)
        update_file(climate_mode_parts, self.climate_path)
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

    def plot_from_cube(self,
                       name:str,
                       ax:Axes,
                       idx:Optional[tuple[int]] = None,
                       select:Optional[Constraint] = None,
                       member:int = 0,
                       colour:str = 'C0',
                       suffix:str = "",
                       dimensionless:bool = False):
        """
        Plot the time dependent variable with given name
        at the given grid point.
        """
        msg = "field:plot_from_cube()"

        plotme = None
        for c in self.ensemble[member]:
            if c.name()[:len(name)] == name:
                if select is not None:
                    plotme = c.extract(select)
                else:
                    plotme = c
                break

        if plotme is None:
            msg += " cannot find cube " + name
            msg += " for plotting"
            logging.error(msg)
            raise RuntimeError(msg)

        if idx is None:
            if len(plotme.shape) == 3:
                jdx = (9, 18)
            elif len(plotme.shape) == 4:
                jdx = (0, 9, 18)
            elif len(plotme.shape) == 5:
                jdx = (0, 7, 9, 18)
            else:
                msg += " cannot choose grid point for plot"
                logging.error(msg)
                raise RuntimeError(msg)
        else:
            ndim = len(plotme.shape)
            jdx = idx[-ndim:]

        if plotme.dtype == _COMPLEX_FP_TYPE:
            logging.warning(msg + " plotting real variable from complex type" + plotme.name())
        elif plotme.dtype == _FP_TYPE:
            logging.info(msg + " plotting real variable " + plotme.name())
        else:
            msg += " variable for plotting has unexpected type " + plotme.name()
            logging.error(msg)
            raise RuntimeError(msg)

        t = plotme.coord('time').points
        f = numpy.real(plotme.data[...,*jdx])

        msg += " plot " + plotme.name() + " at a grid point "
        msg += str(idx)
        msg += " with mean " + str(numpy.mean(f))
        logging.info(msg)
        msg = "field:plot_from_cube()"
        msg += " " + plotme.name() + " start " + str(t[0])
        logging.info(msg)
        msg = "field:plot_from_cube()"
        msg += " " + plotme.name() + " end " + str(t[-1])
        logging.info(msg)

        wt = numpy.zeros_like(plotme.data)
        nlat = len(plotme.coord('latitude').points)
        nlon = len(plotme.coord('longitude').points)
        wt[...,:nlat,:nlon] = plotme.coord(_AREA_WEIGHT_COORDINATE).points

        mnplotme = numpy.ma.mean(plotme.data * wt) / numpy.ma.mean(wt)
        sdplotme = numpy.ma.mean(plotme.data * numpy.ma.conjugate(plotme.data) * wt)
        sdplotme /= numpy.ma.mean(wt)
        sdplotme -= mnplotme * numpy.ma.conjugate(mnplotme)
        sdplotme = numpy.sqrt(sdplotme)

        ax.plot(t, f, colour, label=plotme.name() + ' ' + suffix)
        ax.set_xlabel('time (' + _TIME_UNIT_STRING + ')')
        short_name = plotme.name().split('_')[0]
        mnstr = r"$\langle$" + short_name + r"$\rangle$ = {:8.3e} "
        if not dimensionless: mnstr += str(plotme.units)
        ax.set_title(mnstr.format(mnplotme), loc='left')
        stdstr = r"$\sigma$(" + short_name + ") = {:8.3e} "
        if not dimensionless: stdstr += str(plotme.units)
        ax.set_title(stdstr.format(sdplotme), loc='right')

        return ax

    def plot_stationary_scale(self, name, mnax, stdax, idx=None, member=0, colour='C0', suffix=""):
        """
        Plot the climate mean and spread obtained from the scaling
        that makes the variables stationary.
        """
        msg = "field:plot_stationary_scale()"
        msg += " plot climate for " + name + " at a grid point "
        msg += str(idx)
        logging.info(msg)
        msg = "field:plot_stationary_scale()"

        plotme = None
        for mn, std, c in zip(self.climate_mn, self.climate_ms, self.ensemble[member]):
            if c.name()[:len(name)] == name:
                plotmn = mn
                plotstd = std
                plotme = c

        if plotme is None:
            msg = "cannot find climate data from " + name
            msg += " for plotting"
            logging.error(msg)
            raise RuntimeError(msg)

        if idx is None:
            if len(plotme.shape) == 4:
                idx = (0, 9, 18)
            elif len(plotme.shape) == 5:
                idx = (0, 7, 9, 18)
            else:
                msg = "cannot choose grid point for plot"
                logging.error(msg)
                raise RuntimeError(msg)

        if plotmn.dtype == _COMPLEX_FP_TYPE:
            logging.warning(msg + " plotting real mean from complex type" + plotmn.name())
        elif plotmn.dtype == _FP_TYPE:
            logging.info(msg + " plotting real mean " + plotmn.name())
        else:
            msg += " mean for plotting has unexpected type " + str(plotmn.dtype)
            logging.error(msg)
            raise RuntimeError(msg)

        if plotstd.dtype == _COMPLEX_FP_TYPE:
            logging.warning(msg + " plotting real spread from complex type" + plotstd.name())
        elif plotstd.dtype == _FP_TYPE:
            logging.info(msg + " plotting real spread " + plotstd.name())
        else:
            msg += " spread for plotting has unexpected type " + str(plotstd.dtype)
            logging.error(msg)
            raise RuntimeError(msg)

        t = plotme.coord('time').points
        mnf = numpy.real(plotmn.data[..., *idx])
        stdf = numpy.real(plotstd.data[..., *idx])

        msg = "field:plot_stationary_scale()"
        msg += " plot " + plotme.name() + " at a grid point "
        msg += str(idx)
        msg += " with mean " + str(numpy.mean(mnf))
        logging.info(msg)
        msg = "field:plot_stationary_scale()"
        msg += " " + plotme.name() + " start " + str(t[0])
        logging.info(msg)
        msg = "field:plot_stationary_scale()"
        msg += " " + plotme.name() + " end " + str(t[-1])
        logging.info(msg)

        mnax.plot(t[:len(mnf)] - t[0], mnf, colour, label=plotme.name() + ' ' + suffix)
        mnax.set_xlabel('time (' + _TIME_UNIT_ONLY_STRING + ')')
        stdax.plot(t[:len(stdf)] - t[0], stdf, colour, label=plotme.name() + ' ' + suffix)
        stdax.set_xlabel('time (' + _TIME_UNIT_ONLY_STRING + ')')
        stdax.set_yscale('log')
        return (mnax, stdax)

    def smooth(self,
               window: int = 1,
               fn: Optional[ndarray] = None,
               member: int = 0):
        """
        Apply the moving average as used by AB
        """
        msg = "field::smooth() apply a moving average with window "
        msg += str(window) + " days"
        logging.info(msg)
        msg = "field::smooth()"
        self.order(member)
        if fn is not None:
            if len(fn) != window:
                msg += " the weight function should have "
                msg += str(window) + " elements"
                logging.error(msg)
                raise RuntimeError(msg)
            weights = numpy.flip(fn)
            for c in self.ensemble[member]:
                avg = numpy.ma.zeros_like(c.data)
                nt = len(c.coord('time').points)
                for j in range(nt):
                    total = 0.0
                    for k in range(window):
                        if (1+j-window+k) > 0:
                            avg[j,...] += weights[k] * c.data[1+j-window+k,...]
                            total +=  weights[k]
                    if total > 0.0:
                        avg[j,...] /= total
                c.data = avg
                c.rename(c.name()+_SMOOTH)
        else:
            for c in self.ensemble[member]:
                avg = numpy.ma.zeros_like(c.data)
                nt = len(c.coord('time').points)
                for j in range(window):
                    avg[j,...] = numpy.ma.mean(c.data[:1+j,...], axis=0)
                for j in range(window, nt):
                    avg[j,...] = numpy.ma.mean(c.data[1+j-window:1+j,...], axis=0)
                c.data = avg
                c.rename(c.name()+"_smooth")

        action ={
                    'step': 'smooth',
                    'window': window,
                    'member': member,
                    'weighted': fn is None
                }
        self.record(action)
        logging.info(msg + " trajectory smoothed")
        return

    def scale(self,
              member:int = 0,
              region:list[str] = _DEFAULT_SCALE_REGION):
        """
        Make sure the variance of each dynamical variable
        is of manageable magnitude.
        """
        msg = "field::scale()"
        logging.info(msg + " scale across a region")

        self.order()

        self.scale_mn = CubeList()
        self.scale_sd = CubeList()

        for c in self.ensemble[member]:

            wt = numpy.zeros_like(c.data)
            nlat = len(c.coord('latitude').points)
            nlon = len(c.coord('longitude').points)
            wt[...,:nlat,:nlon] = c.coord(_AREA_WEIGHT_COORDINATE).points

            mn = c.copy()
            mn.rename(c.name() + _REGION_MEAN)
            mn.remove_coord(_AREA_WEIGHT_COORDINATE)
            # mn = mn.collapsed(region,
            #                   iris.analysis.MEAN,
            #                   weights=wt)
            std = c.copy()
            std.rename(c.name() + _REGION_SPREAD)
            std.remove_coord(_AREA_WEIGHT_COORDINATE)
            # std.data = numpy.absolute(std.data * numpy.conjugate(std.data))
            # std = std.collapsed(region,
            #                     iris.analysis.MEAN,
            #                     weights=wt)
            # std.data -= numpy.absolute(mn.data * numpy.conjugate(mn.data))
            # logging.warning(msg + " std.data**2    max = " + str(numpy.amax(std.data)))
            # logging.warning(msg + " std.data**2 argmax = " + str(numpy.argmax(std.data)))
            # logging.warning(msg + " std.data**2    min = " + str(numpy.amin(std.data)))
            # logging.warning(msg + " std.data**2 argmin = " + str(numpy.argmin(std.data)))
            # std.data = numpy.sqrt(std.data)

            # Use a different method to construct the scale
            # that allows easy broadcast
            scale_axes = set()
            for crd in region:
                scale_axes = scale_axes.union(set(self.crdidx(crd)))
            scale_axes =  tuple(scale_axes)
            if numpy.ma.isMaskedArray(c.data):
                wtbcast = numpy.ma.mean(wt,
                                        axis=scale_axes,
                                        out=None,
                                        keepdims=True)
                mnbcast = numpy.ma.mean(c.data * wt,
                                        axis=scale_axes,
                                        out=None,
                                        keepdims=True)
                mnbcast /= wtbcast
                sdbcast = numpy.ma.mean(c.data * numpy.ma.conjugate(c.data) * wt,
                                        axis=scale_axes,
                                        out=None,
                                        keepdims=True)
                sdbcast /= wtbcast
                sdbcast -= numpy.absolute(mnbcast * numpy.ma.conjugate(mnbcast))
                sdbcast = numpy.sqrt(sdbcast)
            else:
                wtbcast = numpy.mean(wt,
                                     axis=scale_axes,
                                     out=None,
                                     keepdims=True)
                mnbcast = numpy.mean(c.data * wt,
                                     axis=scale_axes,
                                     out=None,
                                     keepdims=True)
                mnbcast /= wtbcast
                sdbcast = numpy.mean(c.data * numpy.conjugate(c.data) * wt,
                                     axis=scale_axes,
                                     out=None,
                                     keepdims=True)
                sdbcast /= wtbcast
                sdbcast -= numpy.absolute(mnbcast * numpy.conjugate(mnbcast))
                sdbcast = numpy.sqrt(sdbcast)

            crds = []
            for j, (crd, m) in enumerate(zip(mn.coords(dim_coords=True), numpy.shape(mnbcast))):
                if m == 1:
                    crds.append((crd.collapsed(), j))
                else:
                    crds.append((crd, j))

            mncube = Cube(mnbcast,
                          long_name=mn.name(),
                          var_name=mn.name(),
                          units=mn.units,
                          attributes=mn.attributes,
                          dim_coords_and_dims=crds)
            sdcube = Cube(sdbcast,
                          long_name=std.name(),
                          var_name=std.name(),
                          units=std.units,
                          attributes=std.attributes,
                          dim_coords_and_dims=crds)

            self.scale_mn.append(mncube)
            self.scale_sd.append(sdcube)

            c.data = c.data / sdbcast
            numpy.ma.fix_invalid(c.data, fill_value=0.0)

            c.rename(c.name()+_SCALED)

        update_file(self.scale_mn + self.scale_sd, self.region_path)

        self.record({
                        'step': 'scale',
                        'region': region,
                        'member': member
                    })
        logging.info(msg + " trajectory scaled")
        return

    def plot_stationary(self,
                        fld:str = 'hgt',
                        points:list[tuple[int]] = [(12,9),
                                                   (12,27),
                                                   (6,9),
                                                   (6,27),
                                                   (9,9),
                                                   (9,18)]):
        """
        Make sure the variance of each dynamical variable
        is of manageable magnitude.
        """
        msg = "field::plot_stationary()"
        logging.info(msg + " plot climate mean and spread")


        fig, (axtop, axbottom) = plt.subplots(2, 1, figsize=(8,4))
        colours=['C0', 'C1', 'C2', 'C3', 'C4', 'C6']
        for k, (col, pt) in enumerate(zip(colours, points)):
            axtop, axbottom = self.plot_stationary_scale(fld,
                                                         axtop,
                                                         axbottom,
                                                         idx=pt,
                                                         colour=col,
                                                         suffix='point '+str(k))
        axtop.legend(bbox_to_anchor=(1.35, 1), loc='upper right')
        axbottom.legend(bbox_to_anchor=(1.35, 1), loc='upper right')
        plt.tight_layout()
        logging.info(msg + " climate " + fld)
        plt.savefig((self.figure_folder / (fld + '.climate.pdf')).as_posix(), dpi=300)

        return


    def stationary(self,
                   member:int =0,
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

        for c in self.ensemble[member]:

            crds = [(climate_time, 0)]
            for k, x in enumerate(c.coords(dim_coords=True)[1:]):
                crds.append((x, k+1))
            base = c.name()

            mndata = numpy.ma.zeros_like(c.data[:int(_YEAR/_DAY),...])
            mn = iris.cube.Cube(mndata,
                                long_name=base + _CLIMATE_MEAN + _CORRECTION,
                                var_name=base + _CLIMATE_MEAN + _CORRECTION,
                                units=c.units,
                                dim_coords_and_dims=crds)

            msdata = numpy.zeros_like(c.data[:int(_YEAR/_DAY)], dtype=_FP_TYPE)
            ms = iris.cube.Cube(msdata,
                                long_name=base + _CLIMATE_MS,
                                var_name=base + _CLIMATE_MS,
                                units=c.units,
                                dim_coords_and_dims=crds)

            count_january = 0
            count_march = 0
            count = 0
            for k, pt in enumerate(c.coord('time').points):
                t = _TIME_UNIT_START + pt * _TIME_UNIT
                if t == _YEAR_START:
                    mn.data[:_BEFORE_LEAP_DAY] += c.data[k:k+_BEFORE_LEAP_DAY]
                    count_january += 1
                    count += _BEFORE_LEAP_DAY
                if t == _MARCH_START:
                    mn.data[-_AFTER_LEAP_DAY:] += c.data[k:k+_AFTER_LEAP_DAY]
                    count_march += 1
                    count += _AFTER_LEAP_DAY
            if count_march != count_january:
                msg = "inconsistent counts in climate calculation"
                logging.error(msg)
                raise RuntimeError(msg)
            mn.data /= count_january

            count_january = 0
            count_march = 0
            count = 0
            for k, pt in enumerate(c.coord('time').points):
                t = _TIME_UNIT_START + pt * _TIME_UNIT
                if t == _YEAR_START:
                    dc = c.data[k:k+_BEFORE_LEAP_DAY] - mn.data[:_BEFORE_LEAP_DAY]
                    ms.data[:_BEFORE_LEAP_DAY] += numpy.absolute(dc * numpy.conjugate(dc))
                    count_january += 1
                    count += _BEFORE_LEAP_DAY
                if t == _MARCH_START:
                    dc = c.data[k:k+_AFTER_LEAP_DAY] - mn.data[-_AFTER_LEAP_DAY:]
                    ms.data[-_AFTER_LEAP_DAY:] += numpy.absolute(dc * numpy.conjugate(dc))
                    count_march += 1
                    count += _AFTER_LEAP_DAY
            if count_march != count_january:
                msg = "inconsistent counts in climate calculation"
                logging.error(msg)
                raise RuntimeError(msg)
            ms.data /= count_january

            if roll > 0:
                avmn = numpy.zeros_like(mn.data)
                avms = numpy.zeros_like(ms.data)
                for k in range(-roll, roll+1):
                    avmn += numpy.roll(mn.data, k, axis=0)
                    avms += numpy.roll(ms.data, k, axis=0)
                avmn /= roll
                avms /= roll
                mn.data = avmn
                ms.data = avms

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

        update_file(self.climate_mn + self.climate_ms, self.climate_path)

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

    def covariance(self, save:bool = True):
        """
        Construct a covariance matrix.
        """
        msg = "field:covariance()"
        logging.info(msg + " calculating covariance")

        if self.mtime is None:
            msg += " make sure field is saved and ok()"
            logging.error(msg)
            raise RuntimeError(msg)

        covpath = self.covariance_path
        if covpath.is_file():
            covmtime = os.path.getmtime(covpath.as_posix())
            if self.processed_mtime > covmtime:
                logging.info(msg + " need to update covariance " + str(covmtime))
                covpath.unlink()
            else:
                logging.info(msg + " loading covariance from disk " + str(covmtime))
                covcube = iris.load_cube(covpath.as_posix())
                if numpy.ma.isMaskedArray(covcube.data):
                    self.cov = covcube.data.data
                else:
                    self.cov = covcube.data
                return

        if self.xi is not None:
            xi = self.xi
            xiT = numpy.ma.transpose(self.xi)
            nt = numpy.shape(xi)[0]
            n = numpy.shape(xi)[1]
            if xi.dtype == _FP_TYPE:
                self.cov = numpy.zeros((n,n), dtype=_FP_DOUBLE_TYPE)
                numpy.matmul(xiT, xi, out=self.cov, dtype=_FP_DOUBLE_TYPE)
            elif xi.dtype == _COMPLEX_FP_TYPE:
                self.cov = numpy.zeros((n,n), dtype=_COMPLEX_FP_DOUBLE_TYPE)
                numpy.matmul(numpy.conjugate(xiT), xi, out=self.cov, dtype=_COMPLEX_FP_DOUBLE_TYPE)
            else:
                msg += " unexpected type for self.xi " + str(xi.dtype)
                logging.error(msg)
                raise RuntimeError(msg)
            self.cov /= nt
            if save:
                n = numpy.shape(xi)[1]
                row = iris.coords.DimCoord(numpy.array(range(n)),
                                           long_name='row',
                                           var_name='row',
                                           units='1')
                col = iris.coords.DimCoord(numpy.array(range(n)),
                                           long_name='column',
                                           var_name='column',
                                           units='1')
                crds = [(row, 0), (col, 1)]
                covcube = iris.cube.Cube(self.cov,
                                         long_name='covariance',
                                         var_name='covariance',
                                         dim_coords_and_dims=crds,
                                         units='1')
                iris.save(covcube, covpath.as_posix())
        else:
            msg += " no trajectory for covariance calculation"
            logging.error(msg)
            raise RuntimeError(msg)

        self.record({'step': 'covariance'})
        logging.info(msg + " covariance calculated")
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

        eigenpath = self.eigen_path
        if eigenpath.is_file():
            eigenmtime = os.path.getmtime(eigenpath.as_posix())
            if self.processed_mtime > eigenmtime:
                logging.info(msg + " need to update eigensystem " + str(eigenmtime))
                eigenpath.unlink()
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

        if self.cov is not None:

            nt = numpy.shape(self.xi)[0]
            n = numpy.shape(self.xi)[1]

            try:
                u, s, v = svd(self.cov,
                              full_matrices=True)
                logging.info(msg + "  type of u: " + str(u.dtype))
                logging.info(msg + " shape of u: " + str(numpy.shape(u)))
                logging.info(msg + "  type of s: " + str(s.dtype))
                logging.info(msg + " shape of s: " + str(numpy.shape(s)))
                logging.info(msg + "  type of v: " + str(v.dtype))
                logging.info(msg + " shape of v: " + str(numpy.shape(v)))
            except scipy.linalg.LinAlgError as e:
                msg += " numerical problem with SVD of covariance matrix"
                logging.error(msg)
                logging.exception(e)
                raise RuntimeError(msg)

            try:
                w, z = eigh(self.cov)
                w = w[::-1]
                z = z[:, ::-1]
                logging.info(msg + "  type of w: " + str(w.dtype))
                logging.info(msg + " shape of w: " + str(numpy.shape(w)))
                logging.info(msg + "  type of z: " + str(z.dtype))
                logging.info(msg + " shape of z: " + str(numpy.shape(z)))
            except scipy.linalg.LinAlgError as e:
                msg = "numerical problem with eigensystem of covariance"
                logging.error(msg)
                raise RuntimeError(msg)

            delta = s - w
            d2real = numpy.real(numpy.mean(delta * numpy.conjugate(delta)))
            d2real /= numpy.real(numpy.mean(w * numpy.conjugate(w)))

            for k in range(_N_EVALUE_CHECK):
                Az = numpy.zeros(n, dtype=self.cov.dtype)
                numpy.matmul(self.cov, z[:,k], out=Az, dtype=self.cov.dtype)
                check = Az - w[k] * z[:,k]
                d2real = numpy.real(numpy.mean(check * numpy.conjugate(check)))
                d2real /= numpy.real(w[k])
                # msg += str(numpy.sqrt(d2real))
                # print(msg)

            identity = numpy.diag(numpy.ones(n, dtype=self.cov.dtype))

            # The 0th eigenvector is z[:,0]
            check = numpy.zeros_like(z)
            zH = numpy.conjugate(numpy.transpose(z))
            numpy.matmul(zH, z, out=check, dtype=z.dtype)
            dc = check - identity
            dc2real = numpy.real(dc * numpy.conjugate(dc))
            rmscheck = numpy.sqrt(numpy.mean(dc2real))

            check = numpy.zeros_like(v)
            vH = numpy.conjugate(numpy.transpose(v))
            numpy.matmul(vH, v, out=check, dtype=v.dtype)
            dc = check - identity
            dc2real = numpy.real(dc * numpy.conjugate(dc))
            rmscheck = numpy.sqrt(numpy.mean(dc2real))

            self.count = count
            self.all_eigenvalues = w
            self.eigenvalues = w[:count]
            self.eigenvectors = z[:,:count]
            for k in range(count):
                if w[k] > 0.0:
                    self.eigenvectors[:, k] *= numpy.sqrt(w[k])
                else:
                    msg = "nonpositive eigenvalue in use "
                    msg += str(k) + " " + str(w[k])
                    logging.error(msg)
            self.trace = numpy.sum(w)
            self.fraction = numpy.sum(self.eigenvalues) / self.trace

            allmodes = iris.coords.DimCoord(numpy.array(range(n)),
                                            long_name='mode',
                                            var_name='mode',
                                            units='1')
            allevcube = iris.cube.Cube(self.all_eigenvalues,
                                       long_name='eigenvalue',
                                       var_name='eigenvalue',
                                       dim_coords_and_dims=[(allmodes, 0)],
                                       units='1')
            modes = iris.coords.DimCoord(numpy.array(range(count)),
                                         long_name='mode',
                                         var_name='mode',
                                         units='1')
            grid = iris.coords.DimCoord(numpy.array(range(n)),
                                        long_name='point',
                                        var_name='point',
                                        units='1')
            crds = [(grid, 0), (modes, 1)]
            evectorcube = iris.cube.Cube(self.eigenvectors,
                                         long_name='eigenvector',
                                         var_name='eigenvector',
                                         dim_coords_and_dims=crds,
                                         units='1')
            eigencubes = iris.cube.CubeList()
            eigencubes.append(allevcube)
            eigencubes.append(evectorcube)
            iris.save(eigencubes, eigenpath.as_posix())
        else:
            msg = "calculate covariance before eigensystem"
            logging.error(msg)
            raise RuntimeError(msg)

        self.record({
                        'step': 'eigensystem',
                        'count': count
                    })
        logging.info(msg + " eigensystem calculated")

        return

    def pack_eof(self, plot:bool = True):
        """
        Save eigenvectors as plottable cubes.
        """
        msg = "field:eof() save each eigenvector as an EOF in cubes"
        logging.info(msg)

        self.eof = []
        if self.eigenvectors is not None:
            for k in range(self.count):
                cubes = self.pack(x=self.eigenvectors[:,k], trajectory=False)
                mode = iris.coords.AuxCoord(k,
                                            long_name='mode',
                                            var_name='mode',
                                            units='1')
                for c in cubes:
                    aux = c.coords(dim_coords=False)
                    if 'member' in list(map(cname, aux)):
                        c.remove_coord('member')
                    c.add_aux_coord(mode)
                self.eof.append(cubes)
                cubes.sort(key=cname)
                eofpath = self.mode_folder / ("eof." + str(k) + ".nc")
                iris.save(cubes, eofpath.as_posix())
            for k in range(self.count):
                for c in cubes:
                    fig = plt.figure(figsize=(8,6))
                    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
                    self.plot_eof(ax, k, c.name())
                    plt.tight_layout()
                    plt.savefig(self.mode_figure_folder / ('eof.' + c.name() + "." + str(k) + '.png'), dpi=300)

        return

    def plot_covariance(self, ax):
        """
        Heat map of the covariance matrix
        """
        msg = "field:plot_covariance() heatmap of covariance matrix"
        logging.info(msg)

        if self.cov is not None:
            sns.heatmap(numpy.log(numpy.fabs(self.cov)), ax=ax)
        else:
            msg = "no covariance to plot"
            logging.error(msg)
            raise RuntimeError(msg)
        return

    def plot_eigenvalues(self, ax, suffix=""):
        """
        Show magnitude of selected eigenvalues
        """
        msg = "field:plot_eigenvalues() plot all and selected"
        logging.info(msg)

        nt = numpy.shape(self.xi)[0]
        n = numpy.shape(self.xi)[1]

        j = range(n)
        ax.plot(j, self.all_eigenvalues, 'C0', label="all " + suffix)
        j = range(self.count)
        ax.plot(j, self.eigenvalues, 'C1', label="selected " + suffix)
        ax.set_ylabel(r"$\lambda_j$")
        ax.set_xlabel(r"$j$")
        ax.set_yscale('log')
        ax.set_ylim([0.001,100.0])
        return

    def plot_eigenvalue_sum(self, ax, suffix=""):
        """
        Show magnitude of selected eigenvalues
        """
        msg = "field:plot_eigenvalues() plot all and selected"
        logging.info(msg)

        nt = numpy.shape(self.xi)[0]
        n = numpy.shape(self.xi)[1]

        j = range(n)
        running = numpy.cumsum(self.all_eigenvalues) / numpy.sum(self.all_eigenvalues)
        ax.plot(j, running, 'C0', label="all " + suffix)
        j = range(self.count)
        running = numpy.cumsum(self.eigenvalues)
        running /= numpy.sum(self.all_eigenvalues)
        ax.plot(j, running, 'C1', label="chosen " + suffix)
        ax.set_ylabel(r"$\lambda_j$")
        ax.set_xlabel(r"$j$")
        ax.set_ylim([-0.1,1.1])
        ax.set_xscale('log')
        ax.hlines(y=0.9, xmin=0.1, xmax=n, linewidth=2, color='r')
        ax.hlines(y=0.79, xmin=0.1, xmax=n, linewidth=2, color='r')
        ax.hlines(y=0.74, xmin=0.1, xmax=n, linewidth=2, color='r')
        ax.hlines(y=0.50, xmin=0.1, xmax=n, linewidth=2, color='r')
        return

    def expand(self):
        """
        Expand the vectors over the eigensystem.
        """
        msg = "field:expand()"
        logging.info(msg + " expand over the selected eigenvalues")

        if self.mtime is None:
            msg += " make sure field is saved and ok()"
            logging.error(msg)
            raise RuntimeError(msg)

        zpath = self.zeta_path
        if zpath.is_file():
            zmtime = os.path.getmtime(zpath.as_posix())
            if self.processed_mtime > zmtime:
                logging.info(msg + " updating zeta " + str(zmtime))
                zpath.unlink()
            else:
                logging.info("field:expand() loading zeta from disk " + str(zmtime))
                zcube = iris.load_cube(zpath.as_posix())
                if numpy.ma.isMaskedArray(zcube.data):
                    self.zeta = zcube.data.data
                else:
                    self.zeta = zcube.data
                return

        if (self.xi is not None) and (self.eigenvectors is not None):
            nt = numpy.shape(self.xi)[0]
            if self.xi.dtype == _FP_TYPE:
                self.zeta = numpy.zeros((nt, self.count), dtype=_FP_DOUBLE_TYPE)
                numpy.matmul(self.xi,
                             self.eigenvectors,
                             out=self.zeta,
                             dtype=_FP_DOUBLE_TYPE)
                logging.info(msg + "   shape of xi " + str(numpy.shape(self.xi)))
                logging.info(msg + "  shape of evs " + str(numpy.shape(self.eigenvectors)))
                logging.info(msg + " shape of zeta " + str(numpy.shape(self.zeta)))
            elif self.xi.dtype == _COMPLEX_FP_TYPE:
                self.zeta = numpy.zeros((nt, self.count), dtype=_COMPLEX_FP_DOUBLE_TYPE)
                numpy.matmul(numpy.conjugate(self.xi),
                             self.eigenvectors,
                             out=self.zeta,
                             dtype=_COMPLEX_FP_DOUBLE_TYPE)
            self.zeta /= self.eigenvalues

            modes = iris.coords.DimCoord(numpy.array(range(self.count)),
                                         long_name='mode',
                                         var_name='mode',
                                         units='1')
            crds = [(self.time, 0), (modes, 1)]
            zcube = iris.cube.Cube(self.zeta,
                                   long_name='zeta',
                                   var_name='zeta',
                                   dim_coords_and_dims=crds,
                                   units='1')
            iris.save(zcube, self.zeta_path.as_posix())
        else:
            msg += " calculate eigenvectors and xi before zeta"
            logging.error(msg)
            raise RuntimeError(msg)

        self.record({'step': 'expand'})

        return

    def plot_zeta(self,ax):
        """
        Show the expansion coefficients as a function of time.
        """
        msg = "field:plot_zeta()"
        logging.info(msg + " plot the expansion coefficients")

        t = self.time.points
        for k in range(self.count-1, -1,-1):
            msg = str("mode " + str(k))
            msg += " <zeta> = " + str(numpy.mean(self.zeta[:,k]))
            msg += " sigma(zeta) = " + str(numpy.std(self.zeta[:,k]))
            logging.info(msg)
            ax.plot(t, self.zeta[:,k], 'C' + str(k), label='mode '+str(k))
            ax.set_ylabel(r"$\zeta$")
            ax.set_xlabel(r"$t$ / " + _TIME_UNIT_ONLY_STRING)
        zeta = self.zeta[:,0]
        positive = zeta[zeta>0.0]
        zero = zeta[zeta==0.0]
        negative = zeta[zeta<0.0]
        logging.info(msg + " negative " + str(len(negative)))
        logging.info(msg + " positive " + str(len(positive)))
        ax.plot(t, numpy.where(zeta < 0.0, zeta, 0.0), 'r', label='negative')
        ax.plot(t, numpy.where(zeta > 0.0, zeta, 0.0), 'b', label='positive')

        return

    def plot_eof(self, ax: Axes,  k: int, variable: int | str) -> Axes:
        """
        Plot EOF number k 

        Check first that EOFs are available in self.eof, which is
        generated by the field::eof(...) method. It should be a
        list of CubeList objects. There should be self.count EOFs.

        Parameters
        ----------
        ax: Axis 
            Axis object to which the plot should be added.
        k: int
            Integer index of the EOF to be plotted.
        variable: int | str
            Identify the variable to be plotted either by its position
            in the CubeList or by name.

        Returns
        -------
        Axis object after addition of the plot.
        """
        msg = "field::plot_eof()"
        if self.eof is not None:
            if len(self.eof) != self.count:
                msg += " wrong number of EOFs"
                logging.error(msg)
                raise RuntimeError(msg)
            if k >= self.count:
                msg += " " + str(k) + "th mode does not exist"
                logging.error(msg)
                raise RuntimeError(msg)
            if isinstance(variable, str):
                for c in self.eof[k]:
                    if c.name()[:len(variable)] == variable:
                        plotme = c
            else:
                plotme = self.eof[k][variable]
            if len(plotme.shape) == 3:
                plotme = plotme[0]
            elif len(plotme.shape) > 3:
                msg += " more than three non-time dimensions not implemented"
                logging.error(msg)
                raise RuntimeError(msg)
            plotme.coord('latitude').convert_units('degrees')
            plotme.coord('longitude').convert_units('degrees')
            bcmap = 'bwr' # plt.get_cmap("brewer_PRGn_11")
            x = plotme.coord('longitude')
            xticks = numpy.linspace(numpy.amin(x.bounds[:,0]),
                                    numpy.amax(x.bounds[:,1]),
                                    5,
                                    endpoint=True)
            y = plotme.coord('latitude')
            yticks = numpy.linspace(numpy.amin(y.bounds[:,0]),
                                    numpy.amax(y.bounds[:,1]),
                                    5,
                                    endpoint=True)
            contour = irisplot.contourf(plotme,
                                        numpy.linspace(-1, 1, 21,
                                        endpoint=True),
                                        cmap=bcmap,
                                        extend='both')
            ax.coastlines()
            ax.gridlines()
            ax.ticklabel_format(useMathText=True)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_ylabel(f"latitude / {plotme.coord('latitude').units}", usetex=True)
            ax.set_xlabel(f"longitude / {plotme.coord('longitude').units}", usetex=True)
            ax.set_title(plotme.name(), loc='left', usetex=True)
            # ax.set_title(plotme.name(), loc='left')
            plt.colorbar(contour,
                         ax=ax,
                         location='bottom',
                         orientation='horizontal',
                         drawedges=False,
                         aspect=40,
                         pad=0.1,
                         extend='both')
        else:
            msg += " calculate eofs before plotting"
            logging.error(msg)
            raise RuntimeError(msg)

    def describe(self):
        """
        Check the ensemble looks correct.
        """
        msg = "field:describe() print out cubes in "
        msg += self.local_path.as_posix()
        logging.info(msg)

        print(_RULE_WIDTH*"=")
        for x in self.ensemble:
            for c in x[:-1]:
                print(c)
                print(_RULE_WIDTH*"-")
                print(c.coord('latitude'))
                print(c.coord('longitude'))

                print(_RULE_WIDTH*"-")
            print(x[-1])
            print(_RULE_WIDTH*"-")
            print(x[-1].coord('latitude'))
            print(x[-1].coord('longitude'))
            print(_RULE_WIDTH*"=")
        return

    def load_scale(self, member:int = 0, region_path: Optional[str | Path] = None):
        """
        Load the scaling data saved by field.scale()

        These cubes are saved in self.region_path.
        """
        msg = 'field.load_scale(self)'
        logging.info(msg + ' load cube scaling')

        self.scale_mn = CubeList()
        self.scale_sd = CubeList()

        if region_path is None:
            region_path = self.region_path
        rp = Path(region_path)
        cubes = iris.load(rp)

        for c in self.ensemble[member]:

            count = 0
            for cc in cubes:
                if c.name().split("_")[0] in cc.name():
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
                if c.name().split("_")[0] in cc.name():
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

        logging.info(msg + ' scaling cubes loaded')
        return

    def load_climate(self, member:int = 0, climate_path: Optional[str | Path] = None):
        """
        Load all climate modes from disk
        """
        msg = 'field.load_climate(self)'
        logging.info(msg + ' loading climate cubes')

        self.climate_modes = CubeList()
        self.climate_mn = CubeList()
        self.climate_ms = CubeList()

        if climate_path is None:
            climate_path = self.climate_path
        cp = Path(climate_path)
        cubes = assemble(iris.load(cp))

        for c in self.ensemble[member]:

            count = 0
            for cc in cubes:
                if c.name().split("_")[0] in cc.name():
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
                if c.name().split("_")[0] in cc.name():
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

            count = 0
            for cc in cubes:
                if c.name().split("_")[0] in cc.name():
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

            self.climate_modes.append(rc)

        logging.info(msg + ' climate cubes loaded')
        return

    def load(self):
        """
        Load file(s) from self.local_path.
        Here self.local_path is a single file,
        but in derived classes it may be a glob.
        """
        msg = "field:load() loading "
        if self.local_path.is_file():
            self.mtime = os.path.getmtime(self.local_path.as_posix())
        else:
            self.mtime = None
        if self.processed_path.is_file():
            self.processed_mtime = os.path.getmtime(self.processed_path.as_posix())
            if self.processed_mtime > self.mtime:
                msg += self.processed_path.as_posix()
                logging.info(msg)
                cubes = iris.load(self.processed_path.as_posix())
                self.ensemble = [cubes]
                self.ensemble[0].sort(key=cname)
                self.load_scale()
                self.load_climate()
                self.processed = True
        else:
            self.processed_mtime = None

        msg += self.local_path.as_posix()
        logging.info(msg)
        cubes = iris.load(self.local_path.as_posix())
        self.ensemble = [cubes]
        self.ensemble[0].sort(key=cname)

        if self.extra_path.is_file():
            with open(self.extra_path, 'rb') as handle:
                load_extra_data = pickle.load(handle)
            self.__dict__.update(load_extra_data)

        return

    def save(self, member:int = 0, processed: bool = False):
        """
        Put each ensemble member in a file. Here we
        have a single member in a single file, but
        derived classes will have multiple members.
        """
        msg = "field:save() saving "
        if processed:
            msg += self.processed_path.as_posix()
            logging.info(msg)
            iris.save(self.ensemble[member], self.processed_path)
            if self.ok(self.processed_path):
                self.processed_mtime = os.path.getmtime(self.processed_path)
            else:
                msg += " processed cubes not ok()"
                logging.error(msg)
                raise RuntimeError(msg)
        else:
            msg += self.local_path.as_posix()
            logging.info(msg)
            iris.save(self.ensemble[0], self.local_path.as_posix())
            self.mtime = os.path.getmtime(self.local_path.as_posix())

        save_keys = ['split_mask', 'season_mask']
        save_extra_data = {}
        for key in save_keys:
            if hasattr(self, key):
                save_extra_data[key] = self.__dict__[key]
        if len(save_extra_data) > 0:
            with open(self.extra_path, 'wb') as handle:
                pickle.dump(save_extra_data, handle)

        return

    def record(self, action: dict):
        """
        Record a step in the calculation in the history

        The list self.steps stores the processing steps
        performed on the data, the class that performed
        the action, and the time stamp.
        """
        rcrd = {
                    'time': stamp(),
                    'worker': type(self).__name__,
                    'id': id(self),
                }
        rcrd.update(action)
        self.steps.append(rcrd)
        return

    def dump(self):
        """
        Record history of processing steps in a json file

        Dump the self.steps list to a json file at self.dump_path.
        """
        with open(self.dump_path, 'w', encoding='utf-8') as handle:
            json.dump(self.steps, handle, ensure_ascii=False, indent=4)
        return

    def process(self,
                window:int = 7,
                sigma:int = 4,
                stationary:bool = False,
                plot:bool = True,
                fld:str = 'hgt',
                start:str|datetime = '19480101T0000+0000',
                end:str|datetime = '20240101T0000+0000',
                points:list[tuple[int]] = [(12,9),
                                           (12,27),
                                           (6,9),
                                           (6,27),
                                           (9,9),
                                           (9,18)]):
        """
        Process method

        This should be overridden in child classes to perform all
        the modifications up until the unpacking of the cubes into
        simple vectors. The processing set up here is enough for
        a simple check on the routines that modify the cubes.

        Parameters
        ----------
        window:int
            The size of the window for smoothing with a simple rolling average.
        sigma:int
            The number of non-zero positive seasonal modes to remove. Corresponding
            negative modes are also removed.
        stationary:bool
            Whether to apply scaling to make the signal stationary.
        plot:bool
            Generate plot illustrating the processing?
        fld:str = 'hgt',
        start:str|datetime = '19480101T0000Z',
        end:str|datetime = '20240101T0000Z',
        points:list[tuple[int]]
        """
        msg = 'field:process()'
        logging.info(msg + ' processing cubes')
        if not self.processed:
            if plot:
                colours=['C0', 'C1', 'C2', 'C3', 'C4', 'C6']
                if isinstance(start, str):
                    start = datetime.strptime(start, "%Y%m%dT%H%M%z")
                if isinstance(end, str):
                    end = datetime.strptime(end, "%Y%m%dT%H%M%z")
                timerange = iris.Constraint(time=lambda cell: start <= cell.point <= end)
                fig, (axvtop, axtop, axmiddle, axbottom, axvbottom) = plt.subplots(5, 1, figsize=(8,12))
                for k, (col, pt) in enumerate(zip(colours, points)):
                    axvtop = self.plot_from_cube(fld,
                                                 axvtop,
                                                 select=timerange,
                                                 idx=pt,
                                                 colour=col, suffix='point '+str(k))
            logging.info(msg + ' smoothing')
            self.smooth(window=window)
            if plot:
                for k, (col, pt) in enumerate(zip(colours, points)):
                    axtop = self.plot_from_cube(fld,
                                                axtop,
                                                select=timerange,
                                                idx=pt,
                                                colour=col, suffix='point '+str(k))
            logging.info(msg + ' removing seasonal modes')
            self.modes(sigma=sigma)
            if plot:
                for k, (col, pt) in enumerate(zip(colours, points)):
                    axmiddle = self.plot_from_cube(fld,
                                                   axmiddle,
                                                   select=timerange,
                                                   idx=pt,
                                                   colour=col, suffix='point '+str(k))
            logging.info(msg + ' scaling')
            self.scale()
            if plot:
                for k, (col, pt) in enumerate(zip(colours, points)):
                    axbottom = self.plot_from_cube(fld,
                                                   axbottom,
                                                   select=timerange,
                                                   idx=pt,
                                                   colour=col, suffix='point '+str(k),
                                                   dimensionless=True)
            logging.info(msg + ' make stationary signals')
            self.stationary(apply=stationary)
            if plot:
                for k, (col, pt) in enumerate(zip(colours, points)):
                    axvbottom = self.plot_from_cube(fld,
                                                    axvbottom,
                                                    select=timerange,
                                                    idx=pt,
                                                    colour=col, suffix='point '+str(k),
                                                    dimensionless=True)

                # axbottom.set_ylim(-4,4)
                # axvbottom.set_ylim(-4,4)

                axvtop.legend(bbox_to_anchor=(1.4, 1), loc='upper right')
                # axtop.legend(bbox_to_anchor=(1.4, 1), loc='upper right')
                # axmiddle.legend(bbox_to_anchor=(1.4, 1), loc='upper right')
                # axbottom.legend(bbox_to_anchor=(1.4, 1), loc='upper right')
                # axvbottom.legend(bbox_to_anchor=(1.4, 1), loc='upper right')

                plt.tight_layout()

                plt.savefig((self.figure_folder / (fld + '.pdf')).as_posix(), dpi=300)
            self.processed = True
            if plot:
                self.plot_stationary()
            logging.info(msg + ' cube processing complete')
        else:
            msg += ' cubes already processed - skipping!'
            logging.warning(msg)
        return

    def X(self, t:_FP_TYPE | datetime | str) -> ndarray[_FP_TYPE]:
        """
        Return the input vector for time t

        The statistical learning formulation of forecasting
        attempts to construct predictions of Y from X. Each X and Y
        is indexed for this calculation by a time t. Typically
        the time will be a date, and X will be the state of the
        Earth system on that day. The corresponding Y will be the
        state of the Earth system at some time later.

        Parameters
        ----------
        t:_FP_TYPE | datetime | str
            Specific point in the time coordinate specified at a
            _FP_TYPE using the same time coordinate as in self.time,
            or a datetime, or a string that may be converted into
            a datetime using the strptime method.

        Returns
        -------
        An input vector for training or testing, and the
        corresponding dependent variable.
        """
        msg = 'field:X()'
        if isinstance(t, str):
            tt = datetime.strptime(t, "%Y%m%dT%H%M%z")
        else:
            tt = t
        if isinstance(tt, datetime):
            tt = (tt - _TIME_UNIT_START) / _TIME_UNIT
        tt -= (self.time.points[0] - _ZULU_POINT / _TIME_UNIT)
        j = int(numpy.floor((tt / (_DAY / _TIME_UNIT))))
        if (j >= 0) and (j+1 <= len(self.time.points)):
            return self.zeta[j], self.zeta[j+1]
        else:
            return None, None

    def split(self, split_mask: Optional[ndarray[bool] | Callable[[DimCoord],ndarray[bool]]] = None):
        """
        Apply the split mask to the cubes
        """
        nt = len(self.time.points)
        if split_mask is not None:
            if isinstance(split_mask, ndarray):
                if len(self.time.points) == len(split_mask):
                    self.split_mask = split_mask
                else:
                    msg += " incompatible split mask"
                    logging.error(msg)
                    raise RuntimeError(msg)
            else:
                self.split_mask = split_mask(self.time)
        else:
            shp = numpy.shape(self.time.points)
            self.split_mask = numpy.full(shp, False)
        for c in self.ensemble[0]:
            cmask = numpy.full(c.shape, False)
            for j in range(nt):
                cmask[j,...] = self.split_mask[j]
            if numpy.ma.isMaskedArray(c.data):
                c.data = numpy.ma.array(c.data.data,
                                        mask=numpy.ma.mask_or(cmask, c.data.mask),
                                        fill_value=0.0,
                                        hard_mask=True)
                c.data.harden_mask()
            else:
                c.data = numpy.ma.array(c.data,
                                        mask=cmask,
                                        fill_value=0.0,
                                        hard_mask=True)
                c.data.harden_mask()
        return

    def choose_seasons(self, season_mask: Optional[ndarray[bool] | Callable[[DimCoord],ndarray[bool]]] = None):
        """
        Apply the seasonal mask to the cubes
        """
        nt = len(self.time.points)
        if season_mask is not None:
            if isinstance(season_mask, ndarray):
                if len(self.time.points) == len(season_mask):
                    self.season_mask = season_mask
                else:
                    msg += " incompatible season mask"
                    logging.error(msg)
                    raise RuntimeError(msg)
            else:
                self.season_mask = season_mask(self.time)
        else:
            shp = numpy.shape(self.time.points)
            self.season_mask = numpy.full(shp, False)
        for c in self.ensemble[0]:
            cmask = numpy.full(c.shape, False)
            for j in range(nt):
                cmask[j,...] = self.season_mask[j]
            if numpy.ma.isMaskedArray(c.data):
                c.data = numpy.ma.array(c.data.data,
                                        mask=numpy.ma.mask_or(cmask, c.data.mask),
                                        fill_value=0.0,
                                        hard_mask=True)
                c.data.harden_mask()
            else:
                c.data = numpy.ma.array(c.data,
                                        mask=cmask,
                                        fill_value=0.0,
                                        hard_mask=True)
                c.data.harden_mask()
        return

    def plot_space(self):
        """
        Generate plots related to the space spanned by covariance eigenvectors
        """
        fig, ax = plt.subplots(figsize=(8,8))
        self.plot_covariance(ax)
        plt.savefig((self.figure_folder / 'covariance.png').as_posix(), dpi=300)

        fig,ax = plt.subplots(figsize=(4,3))
        plt.tight_layout()
        self.plot_eigenvalues(ax)
        plt.savefig((self.figure_folder / 'eigenvalue.pdf').as_posix(), dpi=300)

        fig,ax = plt.subplots(figsize=(4,3))
        plt.tight_layout()
        self.plot_eigenvalue_sum(ax)
        plt.savefig((self.figure_folder / 'eigenvalue.sum.pdf').as_posix(), dpi=300)

        fig, ax = plt.subplots(figsize=(6,6))
        self.plot_zeta(ax)
        plt.savefig((self.figure_folder / 'zeta.pdf').as_posix(), dpi=300)

        return

    def __init__(self,
                 local_path: str | Path,
                 src: Optional[list[source]] = None,
                 selection: Optional[iris.Constraint] = None,
                 clean: bool = False,
                 garbage_collect: bool = True):
        """
        Combine cubes obtained from src objects into
        one list of cubelists, and remove source files
        if clean is set to True.

        The sources in the list src should contain the same
        fields for different contiguous intervals of time, and will
        be assembled into a single trajectory. Before assembling
        the different intervals, the cubes in each source are
        sorted by name.
        """
        msg = "field::__init__() "
        logging.info(msg + str(local_path))

        self.processed: bool = False
        self.season_chosen: bool = False
        self.split_done: bool = False

        self.steps = []

        self.src: Optional[list[source]] = src
        self.gc: bool = garbage_collect

        self.local_path: str | Path = Path(local_path)
        self.parent = self.local_path.parent
        self.parent.mkdir(exist_ok=True, parents=True)
        self.figure_folder = (self.parent / _FIGURE_SUBFOLDER) / type(self).__name__
        self.figure_folder.mkdir(exist_ok=True, parents=True)
        self.mode_folder = self.parent / _MODE_SUBFOLDER
        self.mode_folder.mkdir(exist_ok=True, parents=True)
        self.mode_figure_folder = self.figure_folder / _MODE_SUBFOLDER
        self.mode_figure_folder.mkdir(exist_ok=True, parents=True)

        label = "." + type(self).__name__ + ".nc"
        self.local_path = self.local_path.with_suffix(label)
        lp = self.local_path
        self.extra_path: str | Path = lp.with_suffix(_EXTRA)
        self.climate_path: str | Path = lp.with_suffix(_CLIMATE_NC)
        self.region_path: str | Path = lp.with_suffix(_REGION_NC)
        self.processed_path: str | Path = lp.with_suffix(_PROCESSED_NC)
        self.covariance_path: str | Path = lp.with_suffix(_COVARIANCE_NC)
        self.eigen_path: str | Path = lp.with_suffix(_EIGEN_NC)
        self.zeta_path: str | Path = lp.with_suffix(_ZETA_NC)
        self.dump_path: str | Path = lp.with_suffix(_DUMP)
        self.mtime = None
        self.processed_mtime = None
        self.xi = None
        self.zeta = None
        self.eof = None

        # Take the invididual variable names
        # from the first source
        self.names = []
        src[0].cubes.sort(key=cname)
        for c in src[0].cubes:
            self.names.append(c.name())

        if self.ok():
            # File is OK - load it.
            self.load()
            self.time = self.ensemble[0][0].coord('time')
            return
        else:
            # If file failed but exists, it should be removed
            if self.local_path.is_file():
                self.local_path.unlink()

            # Start with a single member ensemble
            self.ensemble = [iris.cube.CubeList()]

            # Consistent ordering
            for s in src:
                s.cubes.sort(key=cname)
                for nm, snm in zip(self.names, map(cname, s.cubes)):
                    if nm != snm:
                        msg += " names " + nm + " and " + snm + " do not match"
                        logging.error(msg)
                        raise RuntimeError(msg)

            # Loop over the fields and concatenate the sources
            for k,nm in enumerate(self.names):
                logging.info(msg + " concatenating " + nm)
                x = iris.cube.CubeList()
                for sj,s in enumerate(src):
                    if s.cubes[k].name() != nm:
                        msg += " cubes out of order for source " + str(sj)
                        logging.error(msg)
                        raise RuntimeError(msg)
                    if s.cubes[k].is_compatible(src[0].cubes[k]):
                        if selection is None:
                            x.append(s.cubes[k])
                        else:
                            x.append(s.cubes[k].extract(selection))
                    else:
                        msg += " incompatible cubes"
                        logging.error(msg)
                        raise RuntimeError(msg)
                try:
                    c = x.concatenate_cube()
                except iris.exceptions.ConcatenateError as e:
                    logging.error("failed to concatenate " + nm)
                    logging.exception(e)
                    raise
                else:
                    self.ensemble[0].append(c)

            self.time = self.ensemble[0][0].coord('time')

            # Make sure to keep the result
            try:
                self.save()
            except RuntimeError as e:
                msg = "field::__init__() failed to save "
                msg += self.local_path.as_posix()
                logging.error(msg)
                logging.exception(e)
                raise
            else:
                if self.ok():
                    msg = "field::__init__()"
                    msg += " assembled file ok()"
                    logging.info(msg)
                    # Remove files no longer needed
                    if clean:
                        for s in src:
                            s.remove()
                else:
                    msg = "field::__init__() assembled file not ok()"
                    logging.error(msg)
                    raise RuntimeError(msg)
                return

        return
