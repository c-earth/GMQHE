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
    '''
    def __init__(self, structure, H):
        self.structure = structure
        self.H = H

    @staticmethod
    def H_from_str(hr, num_wan, num_pnt):
        H_size = num_wan ** 2
        H = dict()
        for pnt in range(num_pnt):
            hr_pnt = hr[H_size * pnt: H_size * (pnt + 1)]
            H_pnt = np.array([line[-2:] for line in hr_pnt])
            H_pnt = H_pnt[:, 0] + H_pnt[:, 1] * 1j
            H_pnt = H_pnt.reshape((num_wan, num_wan), order = 'F')
            R = hr_pnt[0][:3]
            H[R[0]] = H.get(R[0], dict())
            H[R[0]][R[1]] = H[R[0]].get(R[1], dict())
            H[R[0]][R[1]][R[2]] = H_pnt
        return H

    @staticmethod
    def eval_lines(lines):
        return [[eval(value)for value in line.split()] for line in lines]

    @staticmethod
    def eval_hr(hr):
        num_wan = eval(hr[1].split()[0])
        num_pnt = eval(hr[2].split()[0])
        num_deg_lin = int(ceil(num_pnt/15))
        return num_wan, num_pnt, TB_model.eval_lines(hr[3 + num_deg_lin:])

    @classmethod
    def from_strs(cls, cif, hr, wcenter, num_wan, num_pnt):
        structure = Structure.from_str(cif, fmt = 'cif')
        H = cls.H_from_str(hr, num_wan, num_pnt)
        return cls(structure, H)

    @classmethod
    def from_files(cls, cif_file, hr_file, wcenter_file):
        with open(cif_file, 'r', encoding='utf-8') as f:
            cif = f.read()
        with open(hr_file, 'r', encoding='utf-8') as f:
            hr = f.readlines()
        with open(wcenter_file, 'r', encoding='utf-8') as f:
            wcenter = f.readlines()
        num_wan, num_pnt, hr = cls.eval_hr(hr)
        wcenter = cls.eval_lines(wcenter)
        return cls.from_strs(cif, hr, wcenter, num_wan, num_pnt)

    @classmethod
    def from_data_folder(cls, data_folder):
        cif_file = glob(data_folder + '/*.cif')[0]
        hr_file = glob(data_folder + '/*hr.dat')[0]
        wcenter_file = glob(data_folder + '/*wcenter.dat')[0]
        return cls.from_files(cif_file, hr_file, wcenter_file)

    @classmethod
    def from_folder(cls, folder):
        material_files = glob(folder + '/*/')
        materials = [material_file[len(folder)+1:-1] for material_file in material_files]
        return {material: cls.from_data_folder(material_file) for material, material_file in zip(materials, material_files)}

    def __repr__(self):
        dim_x = len(self.H)
        dim_y = len(self.H[0])
        dim_z = len(self.H[0][0])
        dim_wan = self.H[0][0][0].shape
        return f'x cells = {dim_x} \n y cells = {dim_y} \n z cells = {dim_z} \n matrix size = {dim_wan} \n' + self.structure.__repr__()