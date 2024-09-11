import numpy as np

# defining different quadratures, per element and per shape functions

# values for the elements found here
# http://people.ucalgary.ca/~aknigh/fea/fea/triangles/hi.html
# warning: slowest website ever

# TRIANGLES
_linear_triangle_gaussian_quadrature = {
    1: (np.array([[1. / 3, 1. / 3]]), np.array([0.5])),
    2: (np.array([[1. / 6, 1. / 6],
                  [2. / 3, 1. / 6],
                  [1. / 6, 2. / 3]]),
        np.array([1. / 6,
                  1. / 6,
                  1. / 6]))
}

_quadratic_triangle_gaussian_quadrature = {1: (np.array([[1. / 6, 1. / 6],  # gauss points
                                                         [2. / 3, 1. / 6],
                                                         [1. / 6, 2. / 3]]),
                                               np.array([1. / 6,  # weights
                                                         1. / 6,
                                                         1. / 6])),
                                           2: (np.array([[0.1666666667, 0.7886751346],
                                                         [0.6220084679, 0.2113248654],
                                                         [0.0446581987, 0.7886751346],
                                                         [0.1666666667, 0.2113248654]]),
                                               np.array([0.0528312163,
                                                         0.1971687836,
                                                         0.0528312163,
                                                         0.1971687836]))
                                           }

# SEGMENTS
_linear_segment_gaussian_quadrature = {1: (np.array([0]),
                                           np.array([2])),
                                       2: (np.array(
                                           [-1. / np.sqrt(3),
                                            -1. / np.sqrt(3)]),
                                           np.array([1, 1]))}


_quadratic_segment_gaussian_quadrature = {1: (np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)]),
                                                       np.array([5/9, 8/9, 5/9]))}  # to be implemented

# defining the dictionaries containing different shape functions for an element shape
_triangle_gaussian_quadrature = {'linear': _linear_triangle_gaussian_quadrature,
                                 'quadratic': _quadratic_triangle_gaussian_quadrature}

_segment_gaussian_quadrature = {'linear': _linear_segment_gaussian_quadrature,
                                'quadratic': _quadratic_segment_gaussian_quadrature}

# defining the general dict containing everything
gaussian_quadrature = {'triangle': _triangle_gaussian_quadrature,
                       'segment': _segment_gaussian_quadrature}
