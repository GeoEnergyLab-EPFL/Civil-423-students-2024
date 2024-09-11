def undrained_bulk_modulus(k, b, M):
    """
    k: elastic bulk modulus [MPa]
    b: biot coefficient
    M: biot modulus [MPa]
    """
    return k + M*b**2


def poisson_ratio(g, k):
    """
    g is the shear modulus in MPa
    k_u is the bulk modulus
    """
    return (1 - 2 * g / (3 * k)) / (2 + 2 * g / (3 * k))


def young_modulus(g, k):
    """
    g is the shear modulus
    k is the bulk modulus
    """
    return 9 * k / (1 + 3 * k / g)


def bulk_modulus(E, nu):
    """
    E is the Young's modulus
    nu is Poisson's ratio
    """
    return E/(3*(1-2*nu))


def shear_modulus(E, nu):
    """
    E is the Young's modulus
    nu is Poisson's ratio
    """
    return E/(2*(1+nu))

