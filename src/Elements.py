import numpy as np
from GaussianQuadratures import gaussian_quadrature


class Triangle:

    def __init__(self, X, eltype='linear', simultype='2D'):
        self.X = X
        self.eltype = eltype
        if self.eltype == 'linear':
            self.number_nodes = 3
        elif self.eltype == 'quadratic':
            self.number_nodes = 6
        else:
            raise ValueError('Unknown element type')
        self.dim = 2
        self.simultype = simultype

        if self.simultype not in ('2D', 'axis'):
            raise ValueError('Not implemented yet')

        self.gaussian_quadrature = gaussian_quadrature['triangle'][eltype]

    def N(self, xil):
        # xil must be of shape (n points, n dimensions)
        # shape functions are stored column-wise: one column contains values of all shape functions per integration point
        if len(xil.shape) == 1:
            xil = np.array(xil)  # will this give x the shape (1, n)?

        if self.eltype == 'linear':
            n = np.array([1 - xil[:, 0] - xil[:, 1],
                          xil[:, 0],
                          xil[:, 1]]).T

        else:
            n = np.array([xil[:, 0] * (2 * xil[:, 0] - 1),
                          xil[:, 1] * (2 * xil[:, 1] - 1),
                          (1 - xil[:, 0] - xil[:, 1]) *
                          (2 * (1 - xil[:, 0] - xil[:, 1]) - 1),
                          4 * xil[:, 0] * xil[:, 1],
                          4 * xil[:, 1] * (1 - xil[:, 0] - xil[:, 1]),
                          4 * xil[:, 0] * (1 - xil[:, 0] - xil[:, 1])]).T

        return n

    def gradN(self, x):

        # we compute the gradient of the shape function
        # we need the gauss points in case the element is quadratic

        if self.eltype == 'linear':
            DnaDxi = np.array([[-1, 1, 0],
                               [-1, 0, 1]])
            DnaDxi = np.stack([DnaDxi for i in range(len(x))])

        else:
            DnaDxi = np.array(
                [[4 * x[:, 0] - 1, 0 * x[:, 0], 4 * (x[:, 0] + x[:, 1]) - 3, 4 * x[:, 1], -4 * x[:, 1], 4 * (1 - 2 * x[:, 0] - x[:, 1])],
                 [0 * x[:, 0], 4 * x[:, 1] - 1, 4 * (x[:, 0] + x[:, 1]) - 3, 4 * x[:, 0], 4 * (1 - 2 * x[:, 1] - x[:, 0]), - 4 * x[:, 0]]])

            # we re arrange the tensor so that it has a shape of (ngauss points, m, n)
            DnaDxi = np.moveaxis(DnaDxi, [2, 1], [0, 2])

        J = DnaDxi @ self.X
        j = np.linalg.det(J)
        assert np.all(
            j >= 0), "Negative jacobian detected, verify geometry nodes are numbered in counter-clockwise direction"
        DxiDx = np.linalg.inv(J)
        DNaDX = DxiDx @ DnaDxi

        return DNaDX, j

    def mapX(self, x):
        return self.N(x) @ self.X

    def element_conductivity_matrix(self, cond):
        # We get the coordinates and weigths of the gauss point
        xeta, Wl = self.gaussian_quadrature[1]

        # We obtain the jacobian and hessian matrix
        DNaDX, j = self.gradN(xeta)

        # Checking the type of the element and assembling the prefactor
        if self.simultype == '2D':
            premult = (Wl * j * cond)

        elif self.simultype == 'axis':
            mapx = self.mapX(xeta)
            premult = 2 * np.pi * Wl * j * cond * mapx[:, 0]

        # here we asseble the matrix
        cel = np.zeros([self.number_nodes, self.number_nodes])
        for ip in range(Wl.size):
            cel += premult[ip] * DNaDX[ip, :].T.dot(DNaDX[ip, :])

        # and return the element conductivity matrix
        return cel

    def element_stiffness_matrix(self, D):
        xg, wg = self.gaussian_quadrature[1]
        B, j = self.B_strain_matrix(xg)

        # we transpose the B matrix for every gauss point
        BT = np.moveaxis(B, -1, -2)

        # scale is a scaling factor for every gauss points
        if self.simultype == '2D':
            scale = wg * j

        elif self.simultype == 'axis':
            mapx = self.mapX(xg)[:, 0]  # we only take the x component
            scale = 2 * np.pi * mapx * wg * j

        else:
            raise ValueError('Not implemented yet')

        kel = np.sum(scale[:, None, None] * BT @ D @ B, axis=0)
        return kel

    def element_mass_matrix(self, rho):
        # chi, eta coordinates and weights of the gauss points
        xg, wg = self.gaussian_quadrature[2]

        # we compute the shape function and it's gradient for every gauss point
        N_i = self.N(xg)
        _, j = self.gradN(xg)

        # scale is a scaling factor with respect to the gauss points
        if self.simultype == '2D':
            scale = (wg * j * rho)

        elif self.simultype == 'axis':
            mapx = self.mapX(xg)
            scale = 2 * np.pi * wg * j * rho * mapx[:, 0]

        else:
            raise ValueError('Not implemented yet')

        # we do the integration for every gauss point at the same time with vectorization
        A = np.einsum('ij, ik -> ijk', N_i, N_i)
        kel = np.sum(scale[:, None, None] * A, axis=0)

        return kel

    def B_strain_matrix(self, xg):

        # we comptue the gradient of the shape function for every gauss point
        DnaDx, j = self.gradN(xg)

        # we need a zero vector with the right shape so that it fits in the matrix
        zero = 0 * DnaDx[:, 0, 0]

        if self.simultype == '2D':

            # in 2D, we have a tensor with respect to sigma_xx, sigma_yy and tau_xy
            if self.eltype == 'linear':
                # 3 x 6 matrix
                B = np.array([[DnaDx[:, 0, 0], zero, DnaDx[:, 0, 1], zero, DnaDx[:, 0, 2], zero],
                             [zero, DnaDx[:, 1, 0], zero,
                                 DnaDx[:, 1, 1], zero, DnaDx[:, 1, 2]],
                             [DnaDx[:, 1, 0], DnaDx[:, 0, 0], DnaDx[:, 1, 1], DnaDx[:, 0, 1], DnaDx[:, 1, 2], DnaDx[:, 0, 2]]])

            elif self.eltype == 'quadratic':
                # 3 x 12 matrix
                B = np.array([[DnaDx[:, 0, 0], zero, DnaDx[:, 0, 1], zero, DnaDx[:, 0, 2], zero,
                               DnaDx[:, 0, 3], zero, DnaDx[:, 0, 4], zero, DnaDx[:, 0, 5], zero],
                              [zero, DnaDx[:, 1, 0], zero, DnaDx[:, 1, 1], zero, DnaDx[:, 1, 2],
                               zero, DnaDx[:, 1, 3], zero, DnaDx[:, 1, 4], zero, DnaDx[:, 1, 5]],
                              [DnaDx[:, 1, 0], DnaDx[:, 0, 0], DnaDx[:, 1, 1], DnaDx[:, 0, 1], DnaDx[:, 1, 2], DnaDx[:, 0, 2],
                               DnaDx[:, 1, 3], DnaDx[:, 0, 3], DnaDx[:, 1, 4], DnaDx[:, 0, 4], DnaDx[:, 1, 5], DnaDx[:, 0, 5]]])

        elif self.simultype == 'axis':

            # in axissymmetry, we have a tensor with respect to sigma_rr, sigma_zz and tau_rz and sigma_thetatheta

            mapx = self.mapX(xg)
            Nx = self.N(xg)

            if self.eltype == 'linear':
                # 4 x 6 matrix
                B = np.array([[DnaDx[:, 0, 0], zero, DnaDx[:, 0, 1], zero, DnaDx[:, 0, 2], zero],
                              [zero, DnaDx[:, 1, 0], zero,
                                  DnaDx[:, 1, 1], zero, DnaDx[:, 1, 2]],
                              [DnaDx[:, 1, 0], DnaDx[:, 0, 0], DnaDx[:, 1, 1],
                                  DnaDx[:, 0, 1], DnaDx[:, 1, 2], DnaDx[:, 0, 2]],
                              [Nx[:, 0] / mapx[:, 0], zero, Nx[:, 1] / mapx[:, 0], zero, Nx[:, 2] / mapx[:, 0], zero]])

            if self.eltype == 'quadratic':
                # 4 x 12 matrix
                B = np.array([[DnaDx[:, 0, 0], zero, DnaDx[:, 0, 1], zero, DnaDx[:, 0, 2], zero,
                               DnaDx[:, 0, 3], zero, DnaDx[:, 0, 4], zero, DnaDx[:, 0, 5], zero],
                              [zero, DnaDx[:, 1, 0], zero, DnaDx[:, 1, 1], zero, DnaDx[:, 1, 2],
                               zero, DnaDx[:, 1, 3], zero, DnaDx[:, 1, 4], zero, DnaDx[:, 1, 5]],
                              [DnaDx[:, 1, 0], DnaDx[:, 0, 0], DnaDx[:, 1, 1], DnaDx[:, 0, 1], DnaDx[:, 1, 2], DnaDx[:, 0, 2],
                               DnaDx[:, 1, 3], DnaDx[:, 0, 3], DnaDx[:, 1, 4], DnaDx[:, 0, 4], DnaDx[:, 1, 5], DnaDx[:, 0, 5]],
                              [Nx[:, 0] / mapx[:, 0], zero, Nx[:, 1] / mapx[:, 0], zero, Nx[:, 2] / mapx[:, 0], zero,
                               Nx[:, 3] / mapx[:, 0], zero, Nx[:, 4] / mapx[:, 0], zero, Nx[:, 5] / mapx[:, 0], zero]])

        else:
            raise ValueError('Not implemented yet')

        # we rearrange the tensor so that it has a shape of (n gauss points, ndim, m)
        B = np.moveaxis(B, -1, 0)
        return B, j

    
    def project_element_flux(self, cond, sol):

        # chi, eta coordinates and weights of the gauss points
        xg, wg = self.gaussian_quadrature[2]

        if self.simultype == '2D':
            N_i = self.N(xg)
            DNaDx, j = self.gradN(xg)

            # nodal forces are stored column-wise: each column contains horizontal and vertical components at a single node
            nodal_force_per_elem = np.zeros([self.dim, self.number_nodes])
            for i in range(wg.size):
                flux_at_ip = -cond * DNaDx[i] @ sol
                for k in range(self.dim):
                    nodal_force_per_elem[k] += wg[i] * \
                        j[i] * N_i[:, i] * flux_at_ip[k]

        elif self.simultype == 'axis':
            raise ValueError('Not implemented yet')

        else:
            raise ValueError('Not implemented yet')

        return nodal_force_per_elem

class Segment:
    def __init__(self, X, eltype='linear', simultype='2D'):
        self.X = X
        self.eltype = eltype
        self.simultype = simultype

        if self.simultype not in ('2D', 'axis'):
            raise ValueError('Not implemented yet')

        self.gaussian_quadrature = gaussian_quadrature['segment'][eltype]

    def N(self, x):
        # x must be of shape (n points, n dimensions)
        # for example x of shape (10, 2) means that it contains ten 2d points

        if self.eltype == 'linear':
            n = np.array([0.5 * (1 - x), 0.5 * (1 + x)]).T
        elif self.eltype == 'quadratic':
            n = np.array([0.5 * x * (x - 1), (1 + x) *
                         (1 - x), 0.5 * x * (x + 1)]).T
        else:
            raise ValueError('Not implemented yet')

        return n

    def gradN(self, x):
        # gradient of the shape function
        # DN has shape (n gauss points, n segment points)

        if self.eltype == 'linear':
            DN = np.array([-0.5, 0.5])[None, :]

        elif self.eltype == 'quadratic':
            # x has len 3, DN has shape (3, 3), self.X has shape 3
            DN = np.array([x - 0.5, -2 * x, x + 0.5]).T

        else:
            raise ValueError('Not implemented yet')

        # j then has shape (n gauss points,)
        j = DN @ self.X
        DNaDX = DN / j

        return DNaDX, j

    def mapX(self, x):
        return self.N(x) @ self.X

    def B_strain_matrix(self, x):

        if self.simultype == '2D':
            DN = self.gradN(x)
            return DN

        else:
            raise ValueError('Not implemented yet')

    def neumann(self, f):
        xg, wg = self.gaussian_quadrature[1]
        DNaDX, j = self.gradN(xg)
        N = self.N(xg)

        if self.simultype == '2D':
            left = wg * j

        elif self.simultype == 'axis':
            mapx = self.mapX(xg)
            left = 2 * np.pi * wg * j * mapx

        else:
            raise ValueError('Not implemented yet')

        fel = np.sum(left[:, None] * N * f, axis=0)

        return fel
