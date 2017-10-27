#include "types.h"


int uind(int i, int j, int ny)
{   // Returns the starting index of cell (i,j,k)

    return (i*ny+j)*V;
}

int uind(int i, int j, int k, int ny, int nz)
{   // Returns the starting index of cell (i,j,k)

    return ((i*ny+j)*nz+k)*V;
}

void boundaries(Vecr u, Vecr ub, int ndim, int nx, int ny, int nz,
                bool PERIODIC)
{   // If periodic is true, applies periodic boundary conditions,
    // else applies transmissive boundary conditions

    if (ndim==1)
    {
        ub.segment(V, nx*V) = u;

        if (PERIODIC)
        {
            ub.head<V>() = u.tail<V>();
            ub.tail<V>() = u.head<V>();
        }
        else
        {
            ub.head<V>() = u.head<V>();
            ub.tail<V>() = u.tail<V>();
        }
        return;
    }
    if (ndim==2)
    {
        for (int i=0; i<nx; i++)
            for (int j=0; j<ny; j++)
                ub.segment<V>(uind(i+1,j+1,ny+2)) = u.segment<V>(uind(i,j,ny));

        if (PERIODIC)
        {
            for (int i=0; i<nx; i++)
            {
                ub.segment<V>(uind(i+1,0,ny+2)) = u.segment<V>(uind(i,ny-1,ny));
                ub.segment<V>(uind(i+1,ny+1,ny+2)) = u.segment<V>(uind(i,0,ny));
            }
            for (int j=0; j<ny; j++)
            {
                ub.segment<V>(uind(0,j+1,ny+2)) = u.segment<V>(uind(nx-1,j,ny));
                ub.segment<V>(uind(nx+1,j+1,ny+2)) = u.segment<V>(uind(0,j,ny));
            }
        }
        else
        {
            for (int i=0; i<nx; i++)
            {
                ub.segment<V>(uind(i+1,0,ny+2)) = u.segment<V>(uind(i,0,ny));
                ub.segment<V>(uind(i+1,ny+1,ny+2)) = u.segment<V>(uind(i,ny-1,ny));
            }
            for (int j=0; j<ny; j++)
            {
                ub.segment<V>(uind(0,j+1,ny+2)) = u.segment<V>(uind(0,j,ny));
                ub.segment<V>(uind(nx+1,j+1,ny+2)) = u.segment<V>(uind(nx-1,j,ny));
            }
        }

        ub.segment<V>(uind(0,0,ny+2)) = (ub.segment<V>(uind(0,1,ny+2)) +
                                        ub.segment<V>(uind(1,0,ny+2))) / 2;
        ub.segment<V>(uind(0,ny+1,ny+2)) = (ub.segment<V>(uind(0,ny,ny+2)) +
                                           ub.segment<V>(uind(1,ny+1,ny+2))) / 2;
        ub.segment<V>(uind(nx+1,0,ny+2)) = (ub.segment<V>(uind(nx,0,ny+2)) +
                                           ub.segment<V>(uind(nx+1,1,ny+2))) / 2;
        ub.segment<V>(uind(nx+1,ny+1,ny+2)) = (ub.segment<V>(uind(nx,ny+1,ny+2)) +
                                              ub.segment<V>(uind(nx+1,ny,ny+2))) / 2;

        return;
    }
    if (ndim==3)
    {
        // TODO
    }
}

int extended_dimensions(int nx, int ny, int nz)
{
    if (nz>1)
        return (nx+2)*(ny+2)*(nz+2);
    else if (ny>1)
        return (nx+2)*(ny+2);
    else
        return nx+2;
}
