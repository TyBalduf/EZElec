import ez_elec

#from ez_elec.Methods import SCF
#from ez_elec.Molecule import Molecule

#geom=[['H',0.0,0.0,-0.375],['H',0.0,0.0,0.375],['H',0.0,-0.375,0.0],['H',0.0,0.375,0.0]]
#geom=[['O',0.0,0.0,-1.375],['O',0.0,0.0,1.375]]
#geom=[['H',0.0,0.0,-0.375],['H',0.0,0.0,0.375]]
#geom=[['He',0.0,0.0,0.0],['H',0.0,0.0,0.774292095]]
geom="""
He 0.0 0.0 0.0
H 0.0 0.0 0.774292095
"""

params={'geom':geom,
        'basis':"sto-3g",
        "charge":1}

mol=ez_elec.Molecule(geom, charge=1)
print(mol.coords)

solution=ez_elec.SCF(**params)
print(solution.ints)
#P=[[0.95554*2, 0.03986*2],[0.03986*2,0.00166*2]]
#E,C=solution.solve()#guess=P)

#Test updating basis and using old as guess
solution.basis="sto-3g"
E,C=solution.solve()

