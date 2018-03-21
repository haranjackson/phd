from gpr.systems.conserved import flux_cons_ref, nonconservative_product_cons
from gpr.systems.conserved import source_cons_ref, system_cons
from gpr.systems.eigenvalues import max_abs_eigs


def flux(ret, Q, d, *args):
    """ Returns the flux terms of the system (in place)

    Parameters
    ----------
    ret : array
        The array into which the flux will be writen
    Q : array
        The vector of conserved variables for which the flux is being calculated
    d : int
        The direction in which the flux is to be calculated (d=0 for the x-axis,
        d=1 for the y-axis, etc.)
    *args : optional
        Any other parameters that are required by the system

    Notes
    -----
    This function calculates component d of :math:`F`, where

    .. math::

        \\frac{\partial Q}{\partial t} + \\nabla \cdot F + B \cdot  \\nabla Q = S

    """
    flux_cons_ref(ret, Q, d, *args)


def nonconservative_product(ret, x, Q, d, *args):
    """ Returns the product of the matrix of nonconservative terms, and a
    given vector x (in place)

    Parameters
    ----------
    ret : array
        The array into which the product of B(Q) and x will be written
    x : array
        The vector that is to be multiplied by B(Q)
    Q : array
        The vector of conserved variables for which the nonconservative matrix
        is being calculated
    d : int
        The direction in which the nonconservative terms are to be calculated
        (d=0 for the x-axis, d=1 for the y-axis, etc.)
    *args : optional
        Any other parameters that are required by the system

    Notes
    -----
    This function calculates component d of :math:`B`, where

    .. math::

        \\frac{\partial Q}{\partial t} + \\nabla \cdot F + B \cdot  \\nabla Q = S

    """
    nonconservative_product_cons(ret, x, Q, d, *args)


def source(ret, Q, *args):
    """ Returns the source terms of the system (in place)

    Parameters
    ----------
    ret : array
        The array into which the source terms will be writen
    Q : array
        The vector of conserved variables for which the source terms are being
        calculated
    *args : optional
        Any other parameters that are required by the system

    Notes
    -----
    This function calculates :math:`S`, where

    .. math::

        \\frac{\partial Q}{\partial t} + \\nabla \cdot F + B \cdot  \\nabla Q = S

    """
    source_cons_ref(ret, Q, *args)


def system_matrix(Q, d, *args):
    """ Returns the system matrix M

    Parameters
    ----------
    Q : array
        The vector of conserved variables for which the source terms are being
        calculated
    d : int
        The direction in which the system is to be considered (d=0 for the
        x-axis, d=1 for the y-axis, etc.)
    *args : optional
        Any other parameters that are required by the system

    Returns
    -------
    M : array
        The system matrix

    Notes
    -----
    This function calculates component d of :math:`M`, where

    .. math::

        M = \\frac{\partial F}{\partial Q} + B

    and

    .. math::

        \\frac{\partial Q}{\partial t} + \\nabla \cdot F + B \cdot  \\nabla Q = S

    """
    M = system_cons(Q, d, *args)
    return M


def max_eig(Q, d, *args):
    """ Returns the largest absolute size of the eigenvalues of the system

    Parameters
    ----------
    Q : array
        The vector of conserved variables for which the eigenvalues are being
        calculated
    d : int
        The direction in which the system is to be considered (d=0 for the
        x-axis, d=1 for the y-axis, etc.)
    *args : optional
        Any other parameters that are required by the system

    Returns
    -------
    λ : double
        The largest absolute size of the eigenvalues of the system

    Notes
    -----
    This function calculates the largest absolute size of the eigenvalues of
    component d of :math:`M`, where

    .. math::

        M = \\frac{\partial F}{\partial Q} + B

    and

    .. math::

        \\frac{\partial Q}{\partial t} + \\nabla \cdot F + B \cdot  \\nabla Q = S

    """
    λ = max_abs_eigs(Q, d, *args)
    return λ
