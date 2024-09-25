import sys
import os

import numpy as np
import Elements
import PoroElasticProperties as prop
import scipy as sp


def find_eltype(mesh):
    _, number_nodes = mesh.connectivity.shape
    eltype = 'linear' if number_nodes == 3 else 'quadratic' if number_nodes == 6 else 'undefined'
    return eltype


def project_flux(mesh, cond_vec, nodal_field, M=None, return_M=False):
    # Function to project the flux (derivatives of the nodal field) at the nodes of the mesh from the knowledge
    # of the solution at the nodes q== - Cond Grad nodal_field
    # inputs:
    # cond_vec : scalar or vector for conductivity
    # nodal_field : vector containing the field at the nodes of the whole mesh
    #
    # outputs:
    #   q :: an array of shape (dim, number of nodes)

    if len(nodal_field.shape) == 1:
        nodal_field = nodal_field[:, None]

    if np.isscalar(cond_vec):
        cond_vec = [cond_vec]

    # Step 1 : creation of the mesh matrix
    if M is None:
        rho_vec = np.full(len(cond_vec), 1.0)
        M = assemble_mass_matrix(mesh, rho_vec)

    eltype = find_eltype(mesh)

    # Step 2 : creating an empty array containing global nodal forces
    if mesh.simultype == '2D':
        nodal_force = np.zeros([mesh.dim, mesh.number_nodes])
    else:
        raise ValueError('Not implemented yet')

    # Step 3 : iterating through the mesh
    for el_id in range(mesh.number_els):
        # based on the material ID, we access the conductivity of the material
        mat_id = mesh.id[el_id]
        cond = cond_vec[mat_id]

        # Step 3.1 : access the node indices of the element
        # complete the code below
        node_indices = mesh.connectivity[el_id]

        # Step 3.2 : get the coordinates of the nodes and construct an element
        # complete the code below
        X = mesh.nodes[node_indices]
        
        element = Elements.Triangle(X, eltype, mesh.simultype)

        # Step 3.3 : assess nodal values corresponding to the element
        # complete the code below
        elt_heads =  nodal_field[node_indices]

        # Step 3.4 : compute nodal forces caused by the fluxes at the integration points
        # complete the code below
        nodal_force_per_element = element.project_element_flux(cond, elt_heads)

        # Step 3.5 : aggregate elemental forces in the global nodal forces array
        # complete the code below
        nodal_force[:, node_indices] += nodal_force_per_element

    # Step 4 : initiate an empty array for the nodal fluxes
    nodal_flux = np.zeros_like(nodal_force)

    # Step 5 : obtain nodal fluxes in each direction by solving the system mass_matrix * nodal_flux_vector = nodal_force_vector
    # Step 5.1 : get the inverse of the mass matrix
    Minv = sp.sparse.linalg.splu(M)
    for i in range(mesh.dim):
        # step 5.2 : Fast solve using the scipy method
        nodal_flux[i] = Minv.solve(nodal_force[i].T)

    if return_M:
        return nodal_flux, M
    else:
        return nodal_flux


def assemble_mass_matrix(mesh, rho_vec):
    # Function to assemble the mass matri of the system
    # inputs:
    # mesh : a mesh object giving the mesh of the problem
    # rho_vec : a scalar or vector containing the density of the materials in the domain
    #
    # outputs:
    #   M :: the mass matrix of the system of size (number nodes, number nodes)

    # We get the element type
    eltype = find_eltype(mesh)

    # If the provided material parameter are scalars we transform them in the correct
    # sized array
    if np.isscalar(rho_vec):
        rho_vec = [rho_vec]

    # creation of the empty matrix
    M = np.zeros([mesh.number_nodes, mesh.number_nodes])

    # we now loop over all the elements to obtain a the mass matrix
    for el_id in range(mesh.number_els):

        # we access the density of the material by its ID
        mat_id = mesh.id[el_id]
        rho_e = rho_vec[mat_id]

        # access the node indices of the element
        # complete the code below
        node_indices = mesh.connectivity[el_id]

        # get the coordinates of the nodes and construct an element
        # complete the code below
        X = mesh.nodes[node_indices]
        
        element = Elements.Triangle(X, eltype, mesh.simultype)

        # construct an elemental mass matrix
        # complete the code below
        M_el = element.element_mass_matrix(rho_e)

        # aggregate elemental matrices into the global mass matrix
        for i, ni in enumerate(node_indices):
            for j, nj in enumerate(node_indices):
                M[ni, nj] += M_el[i, j]
                # complete the code below
                # M[ , ] += M_el[ , ]

    return sp.sparse.csc_matrix(M)

def assemble_conductivity_matrix(mesh, cond):
    # Function to assemble the conductivity matrix of the system
    # inputs:
    #   - mesh : one of our mesh objects
    #   - cond : the scalar (for uniform permeabilit) or array containing the conductivity of the
    #            material(s)
    #
    # outputs:
    #   - C :: the assembled conductivity matrix for the complete system

    # we pre-define an empty matrix
    C = np.zeros((mesh.number_nodes, mesh.number_nodes))
    eltype = find_eltype(mesh)

    # we want to ensure the conductivity to be accessible by index
    if np.isscalar(cond):
        cond = [cond]

    # we loop over all the elements
    for el_id in range(mesh.number_els):

        # we access the conductivity of the element by its ID
        mat_id = mesh.id[el_id]
        cond_e = cond[mat_id]

        # access the node indices of the element
        # complete the code below
        node_indices = mesh.connectivity[el_id]

        # get the coordinates of the nodes and construct an element
        # complete the code below
        X = mesh.nodes[node_indices]

        # we define the element
        elt = Elements.Triangle(X, eltype, mesh.simultype)

        # construct the element conductivity matrix
        # complete the code below
        C_el = elt.element_conductivity_matrix(cond_e)

        # We assemble the element wise component into the global conductivity matrix
        for i, ni in enumerate(node_indices):
            for j, nj in enumerate(node_indices):
                # complete the code below
                # C[ , ] += C_el[ , ]
                C[ni, nj] += C_el[i, j]
    
    return sp.sparse.csc_matrix(C)