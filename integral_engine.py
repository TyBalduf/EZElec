# Functions to generate the basis and organize
# the overlap, 1e-, and 2e- integrals.
import json
import Basis as ba
import numpy as np

#Constants
ANG_TO_BOHR = 1.8897261245
##Dictionary of element atomic numbers
__ATOMIC_CHARGES={"H":1,"He":2,
                  "Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10}

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
        coord=[c*ANG_TO_BOHR for c in coord]
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

def getCharges(geom,chargeValues=__ATOMIC_CHARGES):
    charges=[chargeValues[g[0]] for g in geom]
    return charges


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

def formMu(basis_funcs):
    cart=((1,0,0),(0,1,0),(0,0,1))
    multi=3*[(True, True, True)]
    mu=__listMats(basis_funcs,cart,multi)
    return mu

def formP(basis_funcs):
    cart = ((1,0,0),(0,1,0),(0,0,1))
    P=__listMats(basis_funcs,cart,phase=-1)
    P=[-p for p in P]
    return P

def formL(basis_funcs):
    cart = ((0,1,1),(1,0,1),(1,1,0))
    multi =((False,False,True),(True,False,False),(False,True,False))
    first=__listMats(basis_funcs,cart,multi,phase=-1)

    multi = ((False, True, False), (False, False, True), (True, False, False))
    second=__listMats(basis_funcs,cart,multi,phase=-1)

    L=[l2-l1 for l1,l2 in zip(first,second)]
    return L

def formNucAttract(basis_funcs,geom):
    potentials=formPotential(basis_funcs,geom)
    charges=getCharges(geom)
    V=np.einsum('i,ijk->jk',charges,potentials)
    return V

def formPotential(basis_funcs,geom):
    nbasis=len(basis_funcs)
    vals=np.zeros((len(geom),nbasis,nbasis))

    for c,coord in enumerate(geom):
        cent=[val*ANG_TO_BOHR for val in coord[1:]]
        def attract(bra,ket):
            return bra.Coulomb_1e(ket,cent)
        vals[c]=__formMat(basis_funcs,attract)

    return vals

##Utility functions
def __formMat(basis_funcs,method,phase=1):
    nbasis=len(basis_funcs)
    mat=np.zeros((nbasis,nbasis))

    for i,bra in enumerate(basis_funcs):
        for j,ket in enumerate(basis_funcs[:i+1]):
            mat[i,j]=method(bra,ket)
            mat[j,i]=phase*mat[i,j]
    return mat

def __listMats(basis_funcs,cart,multi=3*[(False,False,False)],phase=1):
    mats=[]
    for c,m in zip(cart,multi):
        def method(bra,ket):
            return bra.overlap(ket,deriv=c,multi=m)
        mats.append(__formMat(basis_funcs,method,phase=phase))
    return mats
