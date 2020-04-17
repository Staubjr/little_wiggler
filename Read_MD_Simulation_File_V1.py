# Created on 2/29/2020 by Staubjr (11:00)

import sys
import numpy as np
import physvis as vis
import matplotlib.pyplot as plt
import os

class atom_builder:
    all_atoms = []

    def __init__(self, atom_number, atom_type, element, x, y, z):
        self.atom_number = atom_number
        self.atom_type = atom_type
        self.element = element
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.pos = np.array([self.x, self.y, self.z])
        atom_builder.all_atoms.append(self)
        visual(self)

class bond_builder:

    all_bonds = []

    def __init__(self, atom1, atom2, number_of_bonds):
        self.atom1 = atom1
        self.atom2 = atom2
        self.number_of_bonds = number_of_bonds
        bond_builder.all_bonds.append(self)
        visual(self)

class visual:

    all_visual_atoms = []

    all_visual_bonds = []

    atom_colors = { 'carbon'   : (0.5, 0.5, 0.5) ,
                    'hydrogen' : (1., 0., 0.),
                    'oxygen'   : (0., 0., 1.),
                    'nitrogen' : (1., 0.5, 0.) }

    atom_radii = { 'carbon'   : 0.25,
                   'hydrogen' : 0.1,
                   'oxygen'   : 0.35,
                   'nitrogen' : 0.30 }

    bond_color = { 'single' : (0.4, 0.7, 0.9),
                   'double' : (50/255. , 64/255. , 150/255.) } 

    
    def __init__(self, object):

        if object.__class__.__name__ == 'atom_builder':
            self.atom_object = object
            
            self.visual =  vis.sphere(pos = self.atom_object.pos, radius = visual.atom_radii[self.atom_object.element],
                              color = visual.atom_colors[self.atom_object.element] )
            visual.all_visual_atoms.append(self)

        if object.__class__.__name__ == 'bond_builder':
            self.bond_object = object
            
            if self.bond_object.number_of_bonds == 1:
                self.bond_object.bonds = 'single'
            if self.bond_object.number_of_bonds == 2:
                self.bond_object.bonds = 'double'
            
            self.visual = vis.cylinder(radius = self.bond_object.number_of_bonds * 0.05, pos = self.bond_object.atom1.pos,
                                axis = self.bond_object.atom2.pos - self.bond_object.atom1.pos,
                                color = visual.bond_color[self.bond_object.bonds] )

            visual.all_visual_bonds.append(self)
        

def read_txt_file(txt_file):

    file_name = str(txt_file)
    file = open(file_name, 'r')

    rows = file.readlines()
    atom_positions = []
    bonds = []

    for line in rows:
        if line[0:5] == str('#Atom'):
            atom_position_start = rows.index(line) + 1

        if line[0:6] == str('#Bonds'):
            atom_position_stop = rows.index(line) - 2
            bonds_start = rows.index(line) + 4

        bonds_stop = len(rows)

    for line in rows[atom_position_start: atom_position_stop + 1]:
        values = line.split('\t')
        last_value = values[5].strip('\n')
        values.pop(5)
        values.append(last_value)
        atom = atom_builder(values[0], values[1], values[2], values[3], values[4], values[5])


    for line in rows[bonds_start: bonds_stop + 1]:
        values = line.split('\t')
        atoms_in_bond = values[0].split('-')
        number_of_bonds = int(values[1].strip('\n'))
        formatted_values = []
        formatted_values.append(atoms_in_bond[0])
        formatted_values.append(atoms_in_bond[1])
        formatted_values.append(number_of_bonds)
        
        atom_1_number = formatted_values[0]
        atom_2_number = formatted_values[1]
        number_of_bonds = formatted_values[2]
        
        for atom in atom_builder.all_atoms:
            if atom.atom_number == atom_1_number:
                atom_1 = atom
            if atom.atom_number == atom_2_number:
                atom_2 = atom

        bond = bond_builder(atom_1, atom_2, number_of_bonds)


def main():

    file_name = str(sys.argv[1])
    
    read_txt_file(file_name)
    
    display = True

    disp = vis.scene()
    disp.foreground = [1,1,1]

    while display == True:
        vis.rate(30)

if __name__ == '__main__':
    main()
    
