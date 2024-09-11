import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


class Mesh:
    def __init__(self, meshio_mesh, cell_type='triangle', simultype='2D'):

        self.cell_type = cell_type

        self._parse_mesh(meshio_mesh)
        self.simultype = simultype

    def _parse_mesh(self, out):

        # we initially take the connectivity from the meshio object
        connectivity = out.cells_dict[self.cell_type].astype(int)

        # we need to filter the unused nodes that are residuals from pygmsh
        # did not find native way to do it
        used_nodes = np.zeros(len(out.points), dtype=bool)
        used_nodes[np.unique(connectivity)] = True
        self.dim = 2
        nodes = out.points[used_nodes, :self.dim]

        # we create an index map to reflect the new node indices
        index_map = np.zeros(len(out.points), dtype=int)
        index_map[used_nodes] = np.arange(len(nodes))
        new_connectivity = index_map[connectivity]

        # we now parse the arguments to the mesh object
        self.nodes = nodes
        self.connectivity = new_connectivity
        # number of elements
        self.number_els = len(self.connectivity)
        # number of nodes
        self.number_nodes = len(self.nodes)

        # material id
        self.id = np.zeros(self.number_els).astype(int)

    def plot(self, z=None, c='k', shading='gouraud', ax=None, vmin=None, vmax=None, cmap='viridis'):
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect(1)
        else:
            fig = plt.gcf()

        ax.triplot(*self.nodes.T, self.connectivity[:, :3], c=c, lw=0.5)
        if z is not None:
            im = ax.tripcolor(*self.nodes.T, self.connectivity[:, :3], z, cmap=cmap,
                              shading=shading, vmin=vmin, vmax=vmax)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size=0.2, pad=0.1)
            cb = plt.colorbar(im, cax=cax)
            return fig, ax, cb

        return fig, ax
