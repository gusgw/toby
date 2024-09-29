import numpy
import iris
import iris.cube
import iris.coords

from numpy import pi
from iris.coord_systems import GeogCS
from .numbers import _FP_TYPE, radius_in_m, degrees_in_pi_radians

_AREA_WEIGHT_COORDINATE = 'area_wt'

def get_area_wt(cube):
    """
    Return the area weight coordinate appropriate to
    the longitude and latitude coordinates in cube.
    Note that the coordinate is not added to the cube.
    """
    latitude = cube.coord('latitude')
    convert = not ('rad' in str(latitude.units))
    longitude = cube.coord('longitude')

    lat_points = _FP_TYPE(latitude.points)
    lon_points = _FP_TYPE(longitude.points)

    nlat = len(lat_points)
    nlon = len(lon_points)

    area_wt_data = _FP_TYPE(numpy.zeros((nlat, nlon)))
    for k in range(nlon):
        if convert: 
            lat_in_rad = lat_points * pi / degrees_in_pi_radians
        else:
            lat_in_rad = lat_points
        area_wt_data[:, k] = numpy.cos(lat_in_rad)
    area_wt_crd = iris.coords.AuxCoord(area_wt_data,
                                       long_name=_AREA_WEIGHT_COORDINATE,
                                       var_name=_AREA_WEIGHT_COORDINATE,
                                       coord_system=GeogCS(radius_in_m))
    return area_wt_crd

def get_2d_grid_cube(lat_start,
                     lat_stop,
                     nlat,
                     lon_start,
                     lon_stop,
                     nlon):
    """
    Return a zero cube with a desired 2D grid used for
    the iris based regrid system.
    """

    lat_pts = numpy.linspace(lat_start,
                             lat_stop,
                             nlat,
                             endpoint=True,
                             dtype=_FP_TYPE)
    lat_crd = iris.coords.DimCoord(lat_pts,
                                   standard_name='latitude',
                                   units='degrees',
                                   coord_system=GeogCS(radius_in_m))
    lat_crd.guess_bounds()

    if lon_start < lon_stop:
        lon_pts = numpy.linspace(lon_start,
                                 lon_stop,
                                 nlon,
                                 endpoint=True,
                                 dtype=_FP_TYPE)
    else:
        lon_pts_unwrapped = numpy.linspace(lon_start - 360.0,
                                           lon_stop,
                                           nlon,
                                           endpoint=True,
                                           dtype=_FP_TYPE)
        lon_pts = lon_pts_unwrapped
    lon_crd = iris.coords.DimCoord(lon_pts,
                                   standard_name='longitude',
                                   units='degrees',
                                   coord_system=GeogCS(radius_in_m),
                                   circular=True)
    lon_crd.guess_bounds()

    dim_list = [(lat_crd, 0), (lon_crd, 1)]
    grid = iris.cube.Cube(numpy.zeros((nlat, nlon), dtype=_FP_TYPE),
                          var_name='dummy',
                          dim_coords_and_dims=dim_list)
    return grid


grids = {
            'world10p0': get_2d_grid_cube(-85.0, 85.0, 18,
                                         -175.0, 175.0, 36),
            'world5p0': get_2d_grid_cube(-87.5, 87.5, 36,
                                        -177.5, 177.5, 72),
            'world2p5': get_2d_grid_cube(-88.75, 88.75, 72,
                                         -178.75, 178.75, 144),
            'world1p25': get_2d_grid_cube(-89.375, 89.375, 144,
                                         -179.375, 179.375, 288),
            'world1p0': get_2d_grid_cube(-89.5, 89.5, 180,
                                        -179.5, 179.5, 360),
            'world0p5': get_2d_grid_cube(-89.75, 89.75, 360,
                                         -179.75, 179.75, 720),
            'world0p25': get_2d_grid_cube(-89.875, 89.875, 720,
                                         -179.875, 179.875, 1440),
            'nh10p0': get_2d_grid_cube(0.0, 85.0, 18,
                                       5.0, 355.0, 36),
            'nh5p0': get_2d_grid_cube(0.0, 87.5, 36,
                                      2.5, 357.5, 72),
            'nh2p5': get_2d_grid_cube(0.0, 88.75, 72,
                                      1.25, 358.75, 144),
            'nh1p25': get_2d_grid_cube(0.0, 89.375, 144,
                                       0.625, 359.375, 288),
            'nh1p0': get_2d_grid_cube(0.0, 89.5, 180,
                                      0.5, 359.5, 360),
            'nh0p5': get_2d_grid_cube(0.0, 89.75, 360,
                                      0.25, 359.75, 720),
            'nh0p25': get_2d_grid_cube(0.0, 89.875, 720,
                                       0.125, 359.875, 1440),
            'sh10p0': get_2d_grid_cube(-85.0, 0.0, 18,
                                       5.0, 355.0, 36),
            'sh5p0': get_2d_grid_cube(-87.5, 0.0, 36,
                                      2.5, 357.5, 72),
            'sh2p5': get_2d_grid_cube(-88.75, 0.0, 72,
                                      1.25, 358.75, 144),
            'sh1p25': get_2d_grid_cube(-89.375, 0.0, 144,
                                       0.625, 359.375, 288),
            'sh1p0': get_2d_grid_cube(-89.5, 0.0, 180,
                                      0.5, 359.5, 360),
            'sh0p5': get_2d_grid_cube(-89.75, 0.0, 360,
                                      0.25, 359.75, 720),
            'sh0p25': get_2d_grid_cube(-89.875, 0.0, 720,
                                       0.125, 359.875, 1440),
            'sydney1p0': get_2d_grid_cube(-36.5, -30.5, 7,
                                          145.5, 156.5, 12),
            'melbourne1p0': get_2d_grid_cube(-43.5, -34.5, 10,
                                             139.5, 150.5, 12),
            'sydney0p25': get_2d_grid_cube(-36.5, -30.5, 25,
                                           145.5, 156.5, 45),
            'melbourne0p25': get_2d_grid_cube(-43.5, -34.5, 37,
                                              139.5, 150.5, 45),
            'greenland1p0': get_2d_grid_cube(57.5, 85.5, 29,
                                             283.5, 351.5, 69),
            'alaska1p0': get_2d_grid_cube(54.5, 73.5, 20,
                                          190.5, 219.5, 30),
            'texas1p0': get_2d_grid_cube(24.5, 37.5, 14,
                                         252.5, 267.5, 16),
            'texas0p25': get_2d_grid_cube(24.5, 37.5, 53,
                                          252.5, 267.5, 61),
            'usa1p0':   get_2d_grid_cube(19.0, 52.0, 33,
                                         235.0, 288.0, 54),
            'australia1p0': get_2d_grid_cube(-45.0, -10.0, 36,
                                             112.0, 155.0, 44)
        }