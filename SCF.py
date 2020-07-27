# Performs RHF
import integral_engine as ie

'''
Molecular input here
made up example for now
geom is nested list of element/coordinates
bname is name of basis file to read
'''

geom=[['H',0.0,0.0,0.0],['H',0.0,0.0,0.75]]
bname='sto-3g'

#Generate list of basis functions
basis_funcs=ie.initialize(geom,bname)

#Generate integrals from the basis function list
S=ie.formS(basis_funcs)
print(S)

T=ie.formT(basis_funcs)
print(T)

'''
SCF loops here
'''