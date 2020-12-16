# Performs RHF
import integral_engine as ie

def genInts(geom,bname,properties=False):
    #Generate list of basis functions
    basis_funcs=ie.initialize(geom,bname)

    intDict={}
    #Generate integrals from the basis function list
    print("Overlap")
    intDict['S']=ie.formS(basis_funcs)
    print(intDict['S'])

    print("Kinetic energy")
    intDict['T']=ie.formT(basis_funcs)
    print(intDict['T'])

    print("Electron-Nuclear attraction")
    intDict['V']=ie.formNucAttract(basis_funcs,geom)
    print(intDict['V'])

    intDict['Pi']=ie.form2e(basis_funcs)
    print("Two-electron Integrals")
    print(intDict['Pi'])

    if properties:
        print("dipoles")
        intDict['Mu'] = ie.formMu(basis_funcs)
        print(intDict['Mu'])

        print("Momentum")
        intDict['P'] = ie.formP(basis_funcs)
        print(intDict['P'])

        print("Angular momentum")
        intDict['L'] = ie.formL(basis_funcs)
        print(intDict['L'])

    return intDict

def calc(geom=[['H',0.0,0.0,0.0]],bname='test'):
    Integrals=genInts(geom,bname)

    SCF(**Integrals)



def SCF(S,T,V,Pi,**kwargs):
    pass