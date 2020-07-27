# Functions to generate the basis and organize
# the overlap, 1e-, and 2e- integrals.
import json
import Basis as ba
import numpy as np

def initialize(geometry,basis_name):
    #Read saved basis info
    file=f"BasisSets\\{basis_name.lower()}.txt"
    with open(file) as f:
        basis_dict=json.loads(f.read())

    #Create list of basis function
    bas_funcs=[]
    ##Read through all atoms to get element and coordinates
    for atom in geometry:
        elem,*coord=atom
        vals=basis_dict[elem]
        ##Get coefficients, exponents, and shell of each contracted orbital
        for c,e,shell in zip(vals['coeffs'],vals["exponents"],vals["shells"]):
            shell=_expandShell(shell)
            for s in shell:
                ##Generate basis function objects
                temp={"coeffs":c,"exponents":e,"shell":s}
                bas_funcs.append(ba.Basis_Function(coord,**temp))
    return bas_funcs

def _expandShell(shell):
    '''Read shell label, convert to momentum quantum
    numbers of all orbitals in that shell'''
    expansion={'S':[(0,0,0)],
               'P':[(1,0,0),(0,1,0),(0,0,1)],
               'D':[(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)]
              }
    expanded=expansion[shell]
    return expanded

def formS(basis_funcs):
    def temp(bra,ket):
        return bra.overlap(ket)
    return __formMat(basis_funcs,temp)

def formT(basis_funcs):
    def kinetic(bra,ket):
        return -.5*(bra.overlap(ket,deriv=(2,0,0))+
                    bra.overlap(ket,deriv=(0,2,0))+
                    bra.overlap(ket,deriv=(0,0,2)))

    return __formMat(basis_funcs,kinetic)

##Utility functions
def __formMat(basis_funcs,method):
    nbasis=len(basis_funcs)
    mat=np.zeros((nbasis,nbasis))

    for i,bra in enumerate(basis_funcs):
        for j,ket in enumerate(basis_funcs[:i+1]):
            mat[i,j]=method(bra,ket)
            mat[j,i]=mat[i,j]
    return mat