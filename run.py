from phasexp.molecule import Molecule
from phasexp.problem import run

# Alanine
#
#          H1                       HC
#          |                        |
#      H2――N0――H3  O11          HC――NH3――HC  OC
#          |      /                 |       /
#      H5――C4――C10              HA――CT1――CC
#          |      \                 |       \
#      H7――C6――H8  O12         HA1――CT3――HA2 OC
#          |                        |
#          H9                       HA3

def test_load_gro():
    path = "/home/go/Downloads/2d_res_vis/aa/"
    mol = Molecule.load_gro(
        path + "alanine.gro",
        path + "alanine.top",
        "/usr/local/gromacs/share/gromacs/top/charmm36-jul2022.ff/forcefield.itp"
    )
    assert len(mol.atoms) == 13
    assert mol.elements == [
        "N", "H", "H", "H", "C", "H", "C", "H", "H", "H", "C", "O", "O"]
    assert mol.bonds == [
        ( 0,  1), ( 0,  2), ( 0,  3), ( 0,  4), ( 4,  5), ( 4,  6),    
        ( 4, 10), ( 6,  7), ( 6,  8), ( 6,  9), (10, 11), (10, 12)]
    assert list(mol.graph.edges) == [tuple(pair) for pair in  mol.bonds]
    assert [list(triad) for triad in mol.angles] == [
        [0, 4, 5], [0, 4, 6], [0, 4, 10], [1, 0, 2], [1, 0, 3], [1, 0, 4], 
        [2, 0, 3], [2, 0, 4], [3, 0, 4], [4, 6, 7], [4, 6, 8], [4, 6, 9], 
        [4, 10, 11], [4, 10, 12], [5, 4, 6], [5, 4, 10], [6, 4, 10], 
        [7, 6, 8], [7, 6, 9], [8, 6, 9], [11, 10, 12] 
    ]
    assert [list(quad) for quad in mol.dihedrals] == [
        [0, 4, 6, 7], [0, 4, 6, 8], [0, 4, 6, 9], [0, 4, 10, 11], 
        [0, 4, 10, 12], [1, 0, 4, 5], [1, 0, 4, 6], [1, 0, 4, 10], 
        [2, 0, 4, 5], [2, 0, 4, 6], [2, 0, 4, 10], [3, 0, 4, 5], 
        [3, 0, 4, 6], [3, 0, 4, 10],  [5, 4, 6, 7], [5, 4, 6, 8], 
        [5, 4, 6, 9], [5, 4, 10, 11], [5, 4, 10, 12], [6, 4, 10, 11], 
        [6, 4, 10, 12], [7, 6, 4, 10], [8, 6, 4, 10], [9, 6, 4, 10] 
    ]
    assert 10 == mol.energy()
    
def test_run():
    path = "/home/go/Downloads/2d_res_vis/aa/"
    mol = Molecule.load_gro(
        path + "alanine.gro",
        path + "alanine.top"
    )
    result = run(mol)
    assert [14.0067, 1.00794, 1.00794, 1.00794, 12.0107, 1.00794, 12.0107, 
            1.00794, 1.00794, 1.00794, 12.0107, 15.9994, 15.9994] == result