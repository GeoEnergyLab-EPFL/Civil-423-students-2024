import matplotlib.pyplot as plt
import numpy as np
import copy


class MeshUtils:
    def build_edges(vertices):
        edges = np.array([[i, i + 1]
                          for i in range(len(vertices))]) % len(vertices)
        return edges

    def tri3_to_tri6(mesh3):
        # translated function from matlab
        # takes a linear triangular element mesh and converts it to quadratic

        # we first copy the initial mesh
        mesh6 = copy.copy(mesh3)
        # we take every node's connectivity
        sorted_nodes = mesh3.nodes[mesh3.connectivity.flatten(
            order='F')]

        ne = mesh3.number_els

        # this computes the in-between nodes to add to the linear triangle
        x1 = (sorted_nodes[:ne] + sorted_nodes[ne:2 * ne]) / 2
        x2 = (sorted_nodes[ne:2 * ne] + sorted_nodes[2 * ne:]) / 2
        x3 = (sorted_nodes[:ne] + sorted_nodes[2 * ne:]) / 2

        # we use the np.unique() function to map every point to its original position
        new_nodes, ic = np.unique(np.vstack((x1, x2, x3)),
                                  return_inverse=True, axis=0)
        ic += mesh3.number_nodes
        # we now add them to the mesh6 object
        mesh6.nodes = np.vstack((mesh3.nodes, new_nodes))
        mesh6.connectivity = np.hstack(
            (mesh3.connectivity, np.vstack((ic[:ne], ic[ne:2 * ne], ic[2 * ne:])).T))
        mesh6.number_nodes = len(mesh6.nodes)

        return mesh6

    def getBarycenter(mesh, element):

        nodes = mesh.connectivity[element]
        dim = len(mesh.nodes[0])
        barycenter = np.zeros(dim)
        for node in nodes:
            barycenter += mesh.nodes[node]
        barycenter /= len(nodes)

        return barycenter
