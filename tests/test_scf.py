import pytest
import ez_elec
import numpy as np

class TestSCF():
    @pytest.fixture()
    def sample(self):
        geom = """
        He 0.0 0.0 0.0
        H 0.0 0.0 0.774292095
        """

        params = {'geom': geom,
                  'basis': "sto-3g",
                  "charge": 1}

        return ez_elec.SCF(**params)

    def test_overlap(self,sample):
        tempS=sample.ints["S"]
        refS=np.array([[1., 0.53681935],
                       [0.53681935, 1.]])

        np.testing.assert_allclose(tempS, refS)

    def test_kinetic(self,sample):
        tempT=sample.ints["T"]
        refT=np.array([[1.41176317, 0.19744319],
                       [0.19744319, 0.76003188]])

        np.testing.assert_allclose(tempT, refT)

    def test_potential(self,sample):
        tempV=sample.ints["V"]
        refV=np.array([[4.01004618, 1.6292717 ],
                       [1.6292717 , 2.49185755]])

        np.testing.assert_allclose(tempV, refV)

    def test_elecRepulsion(self,sample):
        tempPi = sample.ints["Pi"]
        refPi = np.array([[[[1.05571294, 0.44396499],
                             [0.44396499, 0.59080731]],
                            [[0.44396499, 0.22431934],
                             [0.22431934, 0.36741016]]],
                           [[[0.44396499, 0.22431934],
                             [0.22431934, 0.36741016]],
                            [[0.59080731, 0.36741016],
                             [0.36741016, 0.77460594]]]])

        np.testing.assert_allclose(tempPi, refPi)

