from asyncore import write
from nepdes import NEPDes
import numpy as np
from ase.io import read, write

class Hehe:
    def __init__(self, element_list, r_cut=5., a_cut=3., 
                 r_nmax=10, a_nmax=8, r_basis_size=10, a_basis_size=8,
                 l_max=4, l4_max=2, l5_max=1) -> None:
        self.calc = NEPDes(len(element_list), element_list, r_cut, a_cut, r_nmax, a_nmax, 
                           r_basis_size, a_basis_size, l_max, l4_max, l5_max)
        self.type_dict = {e: i for i, e in enumerate(element_list)}

    def calculate(self, atoms):
        symbols = atoms.get_chemical_symbols()
        _type = [self.type_dict[k] for k in symbols]
        _box = atoms.cell.transpose(1, 0).reshape(-1).tolist()
        _position = atoms.get_positions().transpose(1, 0).reshape(-1).tolist()
        descriptor = np.array(self.calc.find_descriptor(_type, _box, _position))
        descriptor = descriptor.reshape(-1, len(atoms)).transpose(1, 0)
        return descriptor

c = Hehe(['Al', 'N'])
a = read('POSCAR')
d1 = c.calculate(a)

a.rotate(30, 'z', rotate_cell=True)
d2 = c.calculate(a)

a.rotate(40, '-x', rotate_cell=True)
a.positions += 3.
d3 = c.calculate(a)

np.testing.assert_almost_equal(d1, d2, 1e-6)
np.testing.assert_almost_equal(d1, d3, 1e-6)
print('ok')
