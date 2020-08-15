# Performs RHF
import integral_engine as ie

'''
Molecular input here
made up example for now
geom is nested list of element/coordinates
bname is name of basis file to read
'''

#geom=[['H',0.0,0.0,-0.375],['H',0.0,0.0,0.375],['H',0.0,-0.375,0.0],['H',0.0,0.375,0.0]]
geom=[['O',0.0,0.0,-1.375],['O',0.0,0.0,1.375]]
bname='sto-3g'

#Generate list of basis functions
basis_funcs=ie.initialize(geom,bname)

#Generate integrals from the basis function list
print("Overlap")
S=ie.formS(basis_funcs)
print(S)

print("Kinetic energy")
T=ie.formT(basis_funcs)
print(T)

print("dipoles")
Mu=ie.formMu(basis_funcs)
print(Mu)

print("Momentum")
P=ie.formP(basis_funcs)
print(P)

print("Angular momentum")
L=ie.formL(basis_funcs)
print(L)

print("Electron-Nuclear attraction")
V=ie.formNucAttract(basis_funcs,geom)
print(V)

print("Two-electron Integrals")
Pi=ie.form2e(basis_funcs)
print(Pi)

'''
SCF loops here
'''