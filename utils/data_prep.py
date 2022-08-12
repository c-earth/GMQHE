##################################
# helper file for loading raw data
##################################

# import modules
import numpy as np
from math import ceil
from glob import glob
from pymatgen.core.structure import Structure

class TB_model:
    '''
    Tight-binding model class
    '''
    def __init__(self, structure, num_basis, num_point, H):
        '''
        Generate instance that store tight-binding model
        Arguments:
            structure   :   (pymatgen.core.structure.Structure)
                            crystal structure of the material
            num_basis   :   (int)
                            number of orbitals per unit cell considered in the model
            num_point   :   (int)
                            number of interacting neighbor cells for each unit cell
            H           :   (3-layer nested dict containing num_basis x num_basis numpy.array)
                            H[i][j][k] represent the hopping term from the 
                            current unit cell (n_a, n_b, n_c) to the unit cell
                            at (n_a + i, n_b + j, n_c + k)
        '''
        self.structure = structure
        self.num_basis = num_basis
        self.num_point = num_point
        self.H = H

    @staticmethod
    def H_from_str(hr, num_basis, num_point):
        '''
        Convert 2-layer nested list to 3-layer nested dict
        Arguments:
            hr          :   (2-layer nested list)
                            tight-binding Hamiltonian data
            num_basis   :   (int)
                            number of orbitals per unit cell considered in the model
            num_point   :   (int)
                            number of interacting neighbor cells for each unit cell
        Outputs:
            (3-layer nested dict)
        '''
        H_size = num_basis ** 2
        H = dict()
        for pnt in range(num_point):
            hr_pnt = hr[H_size * pnt: H_size * (pnt + 1)]
            H_pnt = np.array([line[-2:] for line in hr_pnt])
            H_pnt = H_pnt[:, 0] + H_pnt[:, 1] * 1j
            H_pnt = H_pnt.reshape((num_basis, num_basis), order = 'F')
            R = hr_pnt[0][:3]
            H[R[0]] = H.get(R[0], dict())
            H[R[0]][R[1]] = H[R[0]].get(R[1], dict())
            H[R[0]][R[1]][R[2]] = H_pnt
        return H

    @staticmethod
    def eval_lines(lines):
        '''
        Evaluate every element in each line
        Arguments:
            lines   :   (list of str)
        Outputs:
            (2-layer nested list)
        '''
        return [[eval(value)for value in line.split()] for line in lines]

    @staticmethod
    def eval_hr(hr):
        '''
        Evaluate tight-binding Hamiltonian from str
        Arguments:
            hr  :   (list of str)
                    each element represents a tight-binding model's element
        Outputs:
            (int)                   :   umber of orbitals per unit cell considered in the model
            (int)                   :   number of interacting neighbor cells for each unit cell
            (2-layer nested list)   :   each row represents a tight-binding model's element
        '''
        num_basis = eval(hr[1].split()[0])
        num_point = eval(hr[2].split()[0])
        num_deg_lin = int(ceil(num_point/15))
        return num_basis, num_point, TB_model.eval_lines(hr[3 + num_deg_lin:])

    @classmethod
    def from_data(cls, cif, hr, num_basis, num_point):
        '''
        Generate TB_model instance from data
        Argements:
            cif         :   (str)
                            lattice structure data
            hr          :   (2-layer nested list)
                            tight-binding Hamiltonian data
            num_basis   :   (int)
                            number of orbitals per unit cell considered in the model
            num_point   :   (int)
                            number of interacting neighbor cells for each unit cell
        Outputs:
            (TB_model)
        '''
        structure = Structure.from_str(cif, fmt = 'cif')
        H = cls.H_from_str(hr, num_basis, num_point)
        return cls(structure, num_basis, num_point, H)

    @classmethod
    def from_files(cls, cif_file, hr_file):
        '''
        Generate TB_model instance from data files
        Argements:
            cif_file    :   (str)
                            path to the cif file containing lattice structure data
            hr_file     :   (str)
                            path to the dat file containing tight-binding Hamiltonian data
        Outputs:
            (TB_model)
        '''
        with open(cif_file, 'r', encoding='utf-8') as f:
            cif = f.read()
        with open(hr_file, 'r', encoding='utf-8') as f:
            hr = f.readlines()
        num_basis, num_point, hr = cls.eval_hr(hr)
        return cls.from_data(cif, hr, num_basis, num_point)

    @classmethod
    def from_data_folder(cls, data_folder):
        '''
        Generate TB_model instance from data folder
        Argements:
            data_folder :   (str)
                            path to the folder containing data files
                            1. cif file containing material lattice structure data
                            2. dat filr containing tight-binding Hamiltonian data
        Outputs:
            (TB_model)
        '''
        cif_file = glob(data_folder + '/*.cif')[0]
        hr_file = glob(data_folder + '/*hr.dat')[0]
        return cls.from_files(cif_file, hr_file)

    @classmethod
    def from_folder(cls, folder):
        '''
        Generate dict of TB_model instances for material data folders in the folder
        Arguments:
            folder  :   (str)
                        path to the folder containing data folders
        Outputs:
            (dict)  :   material (str) -> TB_model
        '''
        data_folders = glob(folder + '/*/')
        materials = [data_folder[len(folder)+1:-1] for data_folder in data_folders]
        return {material: cls.from_data_folder(data_folder) for material, data_folder in zip(materials, data_folders)}

    def __repr__(self):
        '''
        Class representation
        Outputs:
            (str)   :   With following informations,
                        1. Number of orbitals per unit cell considered in the model
                        2. Number of interacting neighbor cells for each unit cell
                        3. pymatgen.core.structure.Structure
        '''
        return f'number of bases = {self.num_basis} \n' +\
               f'number of points {self.num_point} \n' +\
               self.structure.__repr__()