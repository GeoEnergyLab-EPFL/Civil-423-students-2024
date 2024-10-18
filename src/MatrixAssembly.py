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


def project_stress(mesh, E_vec, nu_vec, displacement, M=None, return_M=False):
    # Function to project the stress (derivatives of the nodal field) at the nodes of the mesh from the knowledge
    # of the solution at the nodes
    # inputs:
    # mesh : a mesh object giving the mesh of the problem
    # E_vec : scalar or vector containing the Young's modulus
    # nu_vec : scalar or vector containing Poisson's ratio
    # displacement : vector containing the field at the nodes displacement
    #
    # outputs:
    #   f :: an array of shape (n_stresses, number of nodes) containing the stresses at the element level
    #        where n_stresses is the number of stresses you have (3 for 2D, 4 for axisymmetric)

    # We get the element type
    eltype = find_eltype(mesh)

    # If the provided material parameters are scalars we transform them in the correct
    # sized array
    # Complete below
    # if xxx:
    #
    # if xxx:
    #
    if np.isscalar(E_vec):
        E_vec = [E_vec]
    if np.isscalar(nu_vec):
        nu_vec = [nu_vec]

    # Step 1 : creation of the mesh matrix
    # In case M (the Mass matrix) was already computed, to see for example stress evolution over time
    if M is None:
        M = assemble_mass_matrix(mesh, 1.0)

    # Step 2 : creating an empty array containing global nodal forces
    if mesh.simultype == '2D':
        nodal_force = np.zeros((3, mesh.number_nodes))
    elif mesh.simultype == 'axis':
        nodal_force = np.zeros((4, mesh.number_nodes))
    else:
        raise ValueError('Type not implemented yet')

    # Step 3 : iterating through the mesh
    for el_id in range(mesh.number_els):
        # based on the material ID, we access the material properties
        # Complete below
        # mat_id =
        # E =
        #  nu =
        mat_id = mesh.id[el_id]
        E = E_vec[mat_id]
        nu = nu_vec[mat_id]

        # we need to transform the properties to the bulk (k) and shear
        # modulus (g)
        # Complete below
        # k =
        # g =
        k = prop.bulk_modulus(E, nu)
        g = prop.shear_modulus(E, nu)

        # We want to obtain the elastic stiffness matrix
        D = elastic_isotropic_stiffness(k, g, mesh.simultype)

        # Step 3.1 : access the node indices of the element and degrees of freedom
        # Complete below
        # n_e =
        # n_dof =
        n_e = mesh.connectivity[el_id]
        n_dof = np.vstack([2 * n_e, 2 * n_e + 1]).reshape(-1, order='F')

        # Complete below
        # Step 3.2 : get the coordinates of the nodes and construct an element
        X = mesh.nodes[n_e]

        # Step 3.3 : assemble the element
        elt = Elements.Triangle(X, eltype, mesh.simultype)

        # Step 3.4 : assess nodal values corresponding to the element
        elt_displacement = displacement[n_dof]

        # Step 3.5 : compute nodal forces caused by the fluxes at the integration points
        nodal_force_per_elem = elt.project_element_stress(D, elt_displacement)

        # Step 3.6 : aggregate elemental forces in the global nodal forces array
        nodal_force[:, n_e] += nodal_force_per_elem.T

    # Step 4 : initiate an empty array for the nodal stresses
    f_out = np.zeros_like(nodal_force)

    # Step 5 : obtain nodal fluxes in each direction by solving the system mass_matrix * nodal_flux_vector = nodal_force_vector
    # Step 5.1 : get the inverse of the mass matrix
    Minv = sp.sparse.linalg.splu(M)
    for i in range(mesh.dim):
        # step 5.2 : Fast solve using the scipy method
        f_out[i] = Minv.solve(nodal_force[i].T)

    # Step 6 : return all necessary information
    if return_M:
        return f_out, M
    else:
        return f_out
    
def assemble_stiffness_matrix(mesh, E_vec, nu_vec):
    # First we initiate the stiffnes matrix as an epty matrix
    # The stiffness matrix is of shape 2 * mesh.number_nodes because of x and y displacement
    K = np.zeros((2 * mesh.number_nodes, 2 * mesh.number_nodes))

    # We then obtain the element type
    eltype = find_eltype(mesh)

    # In agreement with the possibility of having multiple materials
    # we ensure to have the correct vector of elastic parameters.
    if np.isscalar(E_vec):
        E_vec = [E_vec]
    if np.isscalar(nu_vec):
        nu_vec = [nu_vec]

    # We finally loop over all elements
    for el_id in range(mesh.number_els):
        # Getting the material type and the corresponding elastic parameters
        # Complete below
        # mat_id =
        # E =
        # nu =
        mat_id = mesh.id[el_id]
        E = E_vec[mat_id]
        nu = nu_vec[mat_id]

        # Note that for ease of coding the stiffness matrix is usually coded
        # using the bulk and shear modulus rather than young's modulus and
        # poisson's ratio. We transform here from one to the other.
        k = prop.bulk_modulus(E, nu)
        g = prop.shear_modulus(E, nu)

        # We get the elastic isotropic stiffness matrix for the element
        D = elastic_isotropic_stiffness(k, g, simultype=mesh.simultype)

        # We now access the nodes of the element and the corresponding
        # degrees of freedom (DOF)
        # Complete below
        # n_e =
        # n_dof =
        node_indices = mesh.connectivity[el_id]
        n_dof = np.vstack(
            [2 * node_indices, 2 * node_indices + 1]).reshape(-1, order='F')

        # we get the coordinates of the nodes
        # Complete below
        # X =
        X = mesh.nodes[node_indices]

        # We can now obtain the elements isoparametric representation
        elt = Elements.Triangle(X, eltype, mesh.simultype)

        # And can finally get the elements stiffness matrix.
        # This matrix depends on the elements elastic matrix
        K_el = elt.element_stiffness_matrix(D)

        # Finally, we put the components back into the global system at the
        # correctposition
        for i, ni in enumerate(n_dof):
            for j, nj in enumerate(n_dof):
                # Complete below
                # K[ , ] += K_el[ , ]
                K[ni, nj] += K_el[i, j]

    return sp.sparse.csc_matrix(K)

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

def assemble_tractions_over_line(mesh, node_list, traction):

    # we obtain the element type from the mesh
    eltype = find_eltype(mesh)

    # getting the indexes of the nodes we need to address
    il = np.isin(mesh.connectivity, node_list)

    # we want to find the number of nodes lying on one side of every (triangle) element
    if eltype == 'linear':
        n = 2
    elif eltype == 'quadratic':
        n = 3
    else:
        raise ValueError('Not implemented yet')

    # We identify all elements which have one side on the line
    elt_line = np.argwhere(il.sum(axis=1) == n)[:, 0]

    # we prepare the output force vectore as a list of two entries per node
    # corresponding to the x and y components of the vector
    f = np.zeros(2 * mesh.number_nodes)

    # We now loop over the elements with an edge on the boundary
    for i, e in enumerate(elt_line):
        # Number of the node on the line
        nn_l = il[e]

        # We get the equivalent global indices of the node
        global_nodes = mesh.connectivity[e, nn_l]
        global_dof = np.array([global_nodes * 2, global_nodes * 2 + 1]).T

        # and we get the coordinates of the node
        X = mesh.nodes[global_nodes]

        # Now we generate the line segment between the two nodes of the line segment
        seg_xi, seg_yi = np.argsort(X, axis=0).T
        segx, segy = X[seg_xi, 0], X[seg_yi, 1]

        # The element consists of two elements, one is the projection on the x-axis
        # the second is the projection on the y-axis
        elt_x = Elements.Segment(segx, eltype=eltype, simultype=mesh.simultype)
        elt_y = Elements.Segment(segy, eltype=eltype, simultype=mesh.simultype)

        # Now we calculate the corresponding forces using the perpendicular traction
        # to the element.
        fs = elt_y.neumann(traction[0])
        fn = elt_x.neumann(traction[1])

        # We can asselmble our global force vector
        f[global_dof[seg_yi, 0]] += fs
        f[global_dof[seg_xi, 1]] += fn

    # And finally return it.
    return f

def elastic_isotropic_stiffness(k, g, simultype='2D'):
    La = k + (4. / 3.) * g
    Lb = k - (2. / 3.) * g

    if simultype == '2D':
        D = np.array([[La, Lb, 0],
                      [Lb, La, 0],
                      [0, 0, g]])

    elif simultype == 'axis':
        D = np.array([[La, Lb, 0, Lb],
                      [Lb, La, 0, Lb],
                      [0, 0, g, 0],
                      [Lb, Lb, 0, La]])

    else:
        raise ValueError('Simulation type not implemented yet')

    return D