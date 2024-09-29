"""
Miscellaneous routines used in downloading, processing,
and storing weather and climate data.
"""
import logging
import os
import shutil
import numpy
import iris

from pathlib import Path
from datetime import datetime
from scipy.constants import pi
from scipy.fft import fftfreq
from iris.cube import Cube, CubeList
from iris.exceptions import ConstraintMismatchError

from .numbers import _FP_TYPE, _FP_DOUBLE_TYPE
from .numbers import _COMPLEX_FP_TYPE, _COMPLEX_FP_DOUBLE_TYPE
from .numbers import _RELATIVE_TOLERANCE
from .numbers import _TIME_UNIT_ONLY_STRING

_MAX_ATTEMPTS = 12
_RETRY_WAIT = 12
_RULE_WIDTH = 100

def rnd(x, decimals=0):
    return numpy.floor(x * 10**decimals + 0.5) / 10**decimals

def stamp():
    """
    A time stamp or similar used as a label

    This is used to log steps in the calculation,
    or to record the times and dates of results.
    """
    return datetime.utcnow().strftime("%Y%M%dT%H%MZ")

def copy_file(src, dst):
    """
    Copy one file to another filesystem location.
    """
    with src.open('rb') as s:
        with dst.open('wb') as d:
            logging.debug("copying " + str(s))
            logging.debug("to " + str(d))
            shutil.copyfileobj(s, d)
    return

def cname(c):
    """
    Useful as a key for sorting and
    in a mapping. This works for either
    coordinate or cube.
    """
    return c.name()

def is_my_real_type(x):
    """
    Check x has one of the acceptable real dtypes

    Parameters
    ----------
    x:  ndarray or Cube
        An object with a property dtype that should be one of the 
        real types imported from .numbers.

    Returns
    -------
    True if one of our chosen real types.
    """
    return ((x.dtype == _FP_TYPE) or (x.dtype == _FP_DOUBLE_TYPE))

def is_my_complex_type(x):
    """
    Check x has one of the acceptable complex dtypes

    Parameters
    ----------
    x:  ndarray or Cube
        An object with a property dtype that should be one of the
        complex types imported from .numbers.

    Returns
    -------
    True if one of our chosen complex types.
    """
    return ((x.dtype == _COMPLEX_FP_TYPE) or (x.dtype == _COMPLEX_FP_DOUBLE_TYPE))

def parts(c):
    """
    Convert a cube to one with real data.

    If the cube is already real, then just put the
    cube in a list and return it.

    Parameters
    ----------
    c: Cube
       A cube with complex data.

    Returns
    -------
    A list of cubes with data of type _FP_TYPE or _FP_DOUBLE_TYPE
    depending on the type of c.data. The list has one element
    if the input c is real, but has real and imaginary
    parts if c is complex.
    """
    clist = CubeList()
    if is_my_real_type(c):
        clist.append(c)
    else:
        cpart = c.copy(data=numpy.real(c.data))
        cpart.rename(c.name() + "_real")
        clist.append(cpart)
        cpart = c.copy(data=numpy.imag(c.data))
        cpart.rename(c.name() + "_imag")
        clist.append(cpart)
    return clist

def cube_present(name:str, clist:CubeList) -> bool:
    """
    Check if cube with name (and suffixes) is included in clist

    Parameters
    ----------
    name:   str
            The name of the cube we are looking for.
    clist:  CubeList
            The list of cubes we're checking for name.

    Returns
    -------
    True if any of the names of the cubes in clist *starts with* name.
    """
    for c in clist:
        if name == c.name()[:len(name)]:
            return True
    return False

def assemble(clist):
    """
    Combine real and imaginary parts from separate cubes

    Parameters
    ----------
    clist:  CubeList
            A CubeList containing separate cubes containing the real
            imaginary parts, indicated by suffixes _real and _imag.

    Returns
    -------
    A CubeList containing the complex cubes.
    """
    cc = CubeList()
    for cr in clist:
        crname = cr.name()
        if crname[-5:] == '_real':
            name = crname[:-5]
            for ci in clist:
                ciname = ci.name()
                if ciname == name + '_imag':
                    cdata = cr.data + 1j * ci.data
                    c = cr.copy(data=cdata)
                    c.rename(name)
                    cc.append(c)
                    break
        else:
            if crname[-5:] != '_imag':
                cc.append(cr)
    cc.sort(key=cname)
    return cc

def update_file(clist: CubeList, fname:str|Path):
    """
    Update an existing NetCDF file with the given CubeList

    If a cube in fname has the same name as a cube in clist,
    then the cube from clist overwrites the one already in the file.

    Parameters
    ----------
    clist:  CubeList
            Cubes to add or overwrite into the given file name.
    fname:  str|Path
            File name to update with new cubes.
    """
    msg = "update_file()"
    logging.info(msg + " updating " + str(fname))
    f = Path(fname)
    if f.is_file():
        if fields_are_ok(f):
            existing = iris.load(f)
            fnames = set(map(cname, existing))
            cnames = set(map(cname, clist))
            combined = CubeList()
            for c in existing:
                if c.name() not in cnames:
                    combined.append(c)
            combined += clist
            if len(combined) != len(fnames.union(cnames)):
                msg += " wrong cube count in update of " + str(f)
                logging.error(msg)
                raise RuntimeError(msg)
            else:
                fnew = f.with_suffix(".updated.nc")
                iris.save(combined, fnew)
                f.unlink()
                fnew.rename(f.as_posix())
    else:
        iris.save(clist, f)
    return

def frequency_coordinate(time, omega='omega'):
    """
    Convert a time coordinate to an angular
    frequency coordinate.
    """
    pts = time.points
    nt = len(pts)
    mn = numpy.mean(pts[1:]-pts[:-1])
    std = numpy.std(pts[1:]-pts[:-1])
    if std / mn > _RELATIVE_TOLERANCE:
        msg = "time points not equally spaced "
        msg += str(std / mn)
        logging.warning(msg)
    wpts = 2 * pi * fftfreq(nt, d = mn)
    omega = iris.coords.AuxCoord(wpts.astype(_FP_TYPE),
                                 long_name=omega,
                                 var_name=omega,
                                 units='rad / ' + _TIME_UNIT_ONLY_STRING)
    return omega

def print_coords(crds):
    """
    Look at a list of coodinates.
    """
    print(_RULE_WIDTH*"-")
    if len(crds) == 0:
        logging.warning("no coordinates to print")
        print("no coordinates to print")
        return
    for x in crds[:-1]:
        print(x)
        print(_RULE_WIDTH*"-")
    print(crds[-1])
    return

def print_cubes_with_title(title, cubes, include_crds=False):
    """
    Looking at the cube printout is a good check.
    """
    print(_RULE_WIDTH*"*")
    logging.info(title)
    print(title)
    if len(cubes) == 0:
        logging.warning("no cubes to print: " + title)
        print("no cubes to print")
        print(_RULE_WIDTH*"=")
        return
    print(_RULE_WIDTH*"=")
    for c in cubes[:-1]:
        print(c)
        print(_RULE_WIDTH*"-")
        print(numpy.shape(c.data))
        if include_crds: print_coords(c.coords())
        print(_RULE_WIDTH*"=")
    print(cubes[-1])
    print(_RULE_WIDTH*"-")
    print(numpy.shape(cubes[-1].data))
    if include_crds: print_coords(cubes[-1].coords())
    print(_RULE_WIDTH*"*")
    return

def load_with_retry(path, selection=None, title="", verbose=True):
    """
    Load a cube list with retries if a PermissionError is thrown.
    """
    cube = None
    for k in range(_MAX_ATTEMPTS):
        try:
            if selection is None:
                cubes = iris.load(path.as_posix())
            else:
                cubes = iris.load(path.as_posix(), selection)
        except PermissionError as e:
            msg = "PermissionError while trying to load " + path.as_posix()
            logging.warning(msg)
            logging.exception(e)
            logging.warning("trying again")
            time.sleep(_RETRY_WAIT)
            continue
        except (ValueError, ConstraintMismatchError) as e:
            logging.error("cannot load " + path.as_posix())
            logging.exception(e)
            raise
        else:
            if verbose:
                print_cubes_with_title("loaded: " + title, cubes)
            break
    for c in cubes:
        if numpy.ma.isMaskedArray(c.data):
            c.data = numpy.ma.array(c.data, dtype=_FP_TYPE)
        else:
            c.data = c.data.astype(_FP_TYPE)
    return cubes

def fields_are_ok(f, expected_minimum_size=None):
    """
    Use iris and iris_grib to make
    sure a file is not corrupted.
    It is necessary to read the file and make sure
    simple operations on the fields do not fail.
    """
    msg = "fields_are_ok()"
    if not Path(f).exists():
        logging.warning(msg + " cannot find " + str(f) + " so check failed")
        return False

    if expected_minimum_size is None:
        ems = 2
    else:
        ems = expected_minimum_size

    fsize = os.path.getsize(str(f))
    if fsize < ems:
        logging.warning(msg + " " + str(f) + " is smaller than expected, fails check")
        return False
    try:
        cubes = iris.load(str(f))
    except Exception as e:
        logging.warning(msg + " exception while checking " + str(f) + ":")
        logging.exception(e)
        return False
    else:
        for c in cubes:
            try:
                cube_name = c.name()
                if c.data.dtype != _FP_TYPE:
                    logging.warning(msg + " " + cube_name + " has wrong type")
                    return False
            except Exception as e:
                logging.warning(msg + " exception while getting cube name from " + str(f))
                logging.exception(e)
                return False
            else:
                try:
                    logging.debug(msg + " <" + cube_name + "> : " + str(numpy.ma.mean(c.data)))
                except Exception as e:
                    logging.warning(msg + " exception while checking " + str(f) + ",")
                    logging.warning("   <" + cube_name + "> :")
                    logging.exception(e)
                    return False
                else:
                    return True
