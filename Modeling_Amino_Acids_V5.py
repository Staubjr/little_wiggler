#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:16:57 2019

@author: jacobstaub
"""

import numpy as np
import math 
import matplotlib.pyplot as plt
import sys
import random
import physvis as vis
import random
from scipy.integrate import ode
import os
from scipy.spatial.transform import Rotation as R

""" Note about the units used here:
    
    To match the par and top files from CHARM files
    these are the following unit scales we are working in:
        
    Distance: Angstroms
    Mass: AMU
    Energy: eV
    Time: sqrt( AMU * Angstroms^2 / eV ) =~ 1E-14 sec
    Angles: Radians

    This results in this unit of energy:

    E' = (1E-2)/n * Joules
    E' = 1.602E-21 * eV """

Time_Unit = ( (1.66054E-27) * (1E-10)**2 / 1.602E-19 ) ** (1/2)
energy_conversion_factor = (2.611E22/6.022E23)

Simulation_config_path = '/home/staubj/Capstone_Files/Txt_CHARM_Files/MD_Simulation_Final_Data/'
Resdiue_config_path = '/home/staubj/Capstone_Files/Txt_CHARM_Files/Residue_Equilibrium_Position_Txt_Files/'
CHARMM_config_path = '/home/staubj/Capstone_Files/Txt_CHARM_Files/CHARRM_FILES/'

def mag(vector):
    ''' Returns the magnitude of a vector '''
    mag_vec = math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
    return mag_vec

def unit_vector(vector):
    ''' Returns the unit vector of a vector '''
    mag_vec = mag(vector)
    uv = vector/mag_vec
    return uv

def cross_product(vector1, vector2):
    ''' Takes the cross product of 2 1X3 matricies '''
    result_x = (vector1[1]*vector2[2] - vector1[2]*vector2[1])
    result_y = (vector1[2]*vector2[0] - vector1[0]*vector2[2])
    result_z = (vector1[0]*vector2[1] - vector1[1]*vector2[0])
    result = np.array([result_x, result_y, result_z])
    return result

def spherical_coordinate_conversion_matrix(vector):
    '''Converts a vector from Cartesian to Spherical coordinates'''
    r = ( (vector[0]**2) + (vector[1]**2) + (vector[2]**2) )**(1/2)
    theta = math.atan( ( (vector[0]**2) + (vector[1]**2) )**(1/2)/vector[2] )
    phi = math.atan(vector[1]/vector[0])
    atom_pos_spherical = np.array([r, phi, theta])
    return atom_pos_spherical

def cartesian_coordinate_conversion_matrix(r, theta, phi):
    '''Converts a vector from Spherical to Cartesian coordinates'''
    x = r*math.sin(phi)*math.cos(theta)
    y = r*math.sin(phi)*math.sin(theta)
    z = r*math.cos(phi)
    atom_pos_cartesian = np.array([x, y,z])
    return atom_pos_cartesian
           
def ders(t, y):

    vals[:] = y
    
    for mol in molecule.all_molecules:
        mol.clear_all_counted()

    index = len(y)//7
        
    dvals = np.empty((len(y)))
    dvals[0:3*index] = y[3*index:6*index]
    
    """ Update all the forces """

    for at in atom.all_atoms:
        at.force[:] = 0.0

    for mol in molecule.all_molecules:
        mol.get_all_forces()

    for lennard_pair in lennard_jones.all_lennard_jones_pairs:
        lennard_pair.atom1_force_counted = False
        lennard_pair.atom2_force_counted = False
    
    for lennard_pair in lennard_jones.all_lennard_jones_pairs:
        if lennard_pair.atom1_force_counted == False and lennard_pair.atom2_force_counted == False:
            lennard_pair.lennard_jones_force()
            
    for sparky_pair in electrostatic.all_electrostatic_pairs:
        sparky_pair.atom1_force_counted = False
        sparky_pair.atom2_force_counted = False
    
    for sparky_pair in electrostatic.all_electrostatic_pairs:
        if sparky_pair.atom1_force_counted == False and sparky_pair.atom2_force_counted == False:
            sparky_pair.electrostatic_force()


    
    """ Get the accelerations in dvals """
    vindex = 3*len(dvals)//7
    windex = 6*len(dvals)//7
    
    for mol in molecule.all_molecules:
        for at in mol.atoms:

            outside = False
            
            distance_vec = at.pos
            distance_mag = mag(distance_vec)
            distance_unit_vec = distance_vec / distance_mag
            
            if distance_mag - simulation.boundary > 0.0:
                outside = True
                
            if outside == False:
                dvals[vindex:vindex+3] = at.force/at.mass - y[vindex:vindex+3] * molecule.dampening_coef/at.mass
                dvals[windex] = - molecule.dampening_coef * np.square(y[vindex:vindex+3]).sum()

            if outside == True:
                
                dvals[vindex:vindex+3] = ( at.force/at.mass - y[vindex:vindex+3] * molecule.dampening_coef/at.mass 
                                                            - abs(distance_mag - simulation.boundary) * simulation.stiffness_coefficient/at.mass
                                                                                           * distance_unit_vec )

                dvals[windex] = ( - molecule.dampening_coef * np.square(y[vindex:vindex+3]).sum() 
                                  - abs(distance_mag - simulation.boundary) * simulation.stiffness_coefficient *
                                  ( distance_unit_vec[0] * y[vindex]     +
                                    distance_unit_vec[1] * y[vindex + 1] +
                                    distance_unit_vec[2] * y[vindex + 2] ) )
                
            vindex += 3
            windex += 1

    return dvals
    

#############################################################################################################################

class simulation():
    """ Simulation class to contain parameters and run variables about the things for the simulation

    This class does the following things:
    
    1.) update_energies: Calculates the instananeous potential energy and kinetic energy
                         of the system as well as the work done by the viscous fluid and
                         'squishy sphere' boundary if True. Values for instantaneous 
                         potential, kinetic energy, and total energy are appended to lists
                         of values over time. Resets instantaneous class attributes for 
                         potential energy, kinetic energy, and work at the end.

    2.) visualize: If called, creates an instance of the visual classes for all bonds and 
                   atoms in the molecules in the class list molecules.

    3.) graph_energy_conservation: Uses matplotlib to create a plot with three subplots:
                                   total energy and work, total energy plus work, and potential
                                   and kinetic energy with repsect to time. If save image == True,
                                   creates SVG file with hardcoded title in current directory.

    4.) get_potential: After reseting class attribute potential_energy to 0, calculates the potential
                       in the molecules in class list molecules due to lennard-jones pairs, electrostatic
                       pairs, bonds, bond angles, and dihedrals. 

    5.) get_kinetic_energy: After reseting class attriute kinetic_energy to 0, calculates the kinetic
                            energy of all of the molecules in the class list molecules.

    6.) get_work: For all atoms in molecules in class list molecule, add all the works done by the atoms
                  in the time step to the class attribute work. Also, add the value to the class 
                  attribute total_work.

    7.) hydrate: Randomly adds water to the simulation. The function only if by adding the water to the 
                 simulation, the potential does not increase by a large hardcoded value.  This function 
                 randomly makes water molecules inside the sphere and randomly rotates them usnig Scipy.

    8.) print_final_positions: Print final positions of all atoms in all molecules in class list molecule. 
                               Also, print the atoms bonded to each other by atom index as well as element
                               type. 
    """

    boundary = 0
    radius = 0
    stiffness_coefficient = 0
    
    def __init__(self, squishy_sphere = False, radius = 100.0, stiffness = 1E5):
        
        if squishy_sphere == True:

            if radius == 100.0:
                raise Exception('If you want a boundary, you need to enter a radius!')

            if stiffness == 0.0:
                raise Exception('If you want a boundary, you need to determine a stiffness!')
                                           
            simulation.boundary = radius
            simulation.radius = radius
            simulation.stiffness_coefficient = stiffness
            
        if squishy_sphere == False:

            simulation.boundary = 1E6
            simulation.radius = 1E6
            simulation.stiffness_coefficient = 0.0

        self.initialize_CHARMM_values()

        self.molecules = []
        self.potential_energy = 0.0
        self.kinetic_energy = 0.0
        self.work = 0.0
        self.total_work = 0.0

        self.times = []
        self.kinetic_energies = []
        self.potential_energies = []
        self.total_energies = []
        self.works = []
        self.total_works = []
        self.energy_plus_total_work = []


    def initialize_CHARMM_values(self):

        os.chdir(CHARMM_config_path)

        file = open('Formatted_CHARMM_File.txt', 'r')
        lines = file.readlines()

        for line in lines:
            if line[1:6] == 'BONDS':
                bond_start_line = lines.index(line) + 2

            if line [1:12] == 'BOND_ANGLES':
                bond_end_line = lines.index(line) - 2
                bond_angle_start_line = lines.index(line) + 2

            if line[1:10] == 'Dihedrals':
                bond_angle_end_line = lines.index(line) - 2
                dihedral_start_line = lines.index(line) + 2

            if line[1:8] == 'Lennard':
                dihedral_end_line = lines.index(line) - 2
                lennard_jones_start_line = lines.index(line) + 2

            lennard_jones_end_line = len(lines)

        for line in lines[bond_start_line: bond_end_line]:
        
            values = line.split('\t')
            index = str(values[0] + '-' + values[1])

            bond.k_bond[index] = float(values[2]) * energy_conversion_factor
            bond.E0_bond[index] = float(values[3])

        for line in lines[bond_angle_start_line: bond_angle_end_line]:

            values = line.split('\t')
            index = str(values[0] + '-' + values[1] + '-' + values[2])

            bond_angle.k_bond_angle[index] = float(values[3]) * energy_conversion_factor
            bond_angle.bond_angle[index] = float(values[4]) * math.pi/180

        for line in lines[dihedral_start_line: dihedral_end_line]:

            values = line.split('\t')
            index = str(values[0] + '-' + values[1] + '-' + values[2] + '-' + values[3])

            dihedral.k_chi[index] = float(values[4]) * energy_conversion_factor
            dihedral.n[index] = int(values[5])
            dihedral.delta[index] = float(values[6])

        for line in lines[lennard_jones_start_line: lennard_jones_end_line]:

            values = line.split('\t')
            index = str(values[0])

            lennard_jones.epsilon_i[index] = float(values[1]) * energy_conversion_factor
            lennard_jones.r_min_i[index] = float(values[2])        

        
    def update_energies(self):
        
        self.get_potential()
        self.get_kinetic_energy()        
        self.get_work()
        
        self.total_energy = self.kinetic_energy + self.potential_energy

        self.potential_energies.append(self.potential_energy)
        self.kinetic_energies.append(self.kinetic_energy)
        self.total_energies.append(self.total_energy)
        
        self.works.append(-self.work)
        self.total_works.append(-self.total_work)
        self.energy_plus_total_work.append(self.total_energy - self.work)
        
        self.potential_energy = 0.0
        self.kinetic_energy = 0.0
        self.work = 0.0


    def visualize(self):

        for molecule in self.molecules:

            for atom in molecule.atoms:
                visual(atom)
                
            for bond in molecule.bonds:
                visual(bond)
        

    def graph_energy_conservation(self, save_image = False):
        
        fig, ax = plt.subplots( 1, 3, sharex = True, sharey = True, figsize = (16, 9))
        ax[0].plot(self.times, self.total_energies, label = 'Total Energy', color = 'r')
        ax[0].plot(self.times, self.works, label = 'Total_Work', color = 'b')
        ax[2].plot(self.times, self.potential_energies, label = 'Potential Energy', color = 'm')
        ax[2].plot(self.times, self.kinetic_energies, label = 'Kinetic Energy', color = 'c')
        ax[1].plot(self.times, self.energy_plus_total_work, label = 'Total Energy + Work', color = '#4F0066')
        ax[2].legend()
        ax[0].legend()
        ax[2].set_ylabel('Energy (eV)')
        ax[0].set_ylabel('Energy (eV)')
        ax[2].set_xlabel('Time (Jake Seconds)')
        ax[1].set_ylabel('Energy (eV)')
        ax[0].set_title('Work and Energy')
        ax[1].set_title('Total Energy Plus Work')
        ax[2].set_title('Potential and Kinetic Energy')
        fig.show()
        plt.show()

        if save_image == True:
            fig.savefig('Alanine_Stabilization_Energy_Diagram.svg', type = '.svg' )

    def get_potential(self):
        
        self.potential_energy = 0.0

        for lennard_pair in lennard_jones.all_lennard_jones_pairs:
            lennard_pair.atom1_potential_counted = False
            lennard_pair.atom2_potential_counted = False

        for sparky_pair in electrostatic.all_electrostatic_pairs:
            sparky_pair.atom1_potnetial_counted = False
            sparky_pair.atom2_potential_counted = False

        for molecule in self.molecules:

            for bond in molecule.bonds:
                self.potential_energy += bond.bond_potential()

            for bond_angle in molecule.bond_angles:
                self.potential_energy += bond_angle.bond_angle_potential()

            for dihedral in molecule.dihedrals:
                self.potential_energy += dihedral.dihedral_potential()

            for lennard_pair in lennard_jones.all_lennard_jones_pairs:
                if lennard_pair.atom1_potential_counted == False and lennard_pair.atom2_potential_counted == False:
                    self.potential_energy += lennard_pair.lennard_jones_potential()

            for sparky_pair in electrostatic.all_electrostatic_pairs:
                if sparky_pair.atom1_potential_counted == False and sparky_pair.atom2_potential_counted == False:
                    self.potential_energy += sparky_pair.electrostatic_potential()


    def get_kinetic_energy(self):

        self.kinetic_energy = 0.0
        
        for molecule in self.molecules:
            for atom in molecule.atoms:                
                self.kinetic_energy += 0.5 * atom.mass * mag(atom.vel)**2

                
    def get_work(self):
        for molecule in self.molecules:
            for atom in molecule.atoms:                
                self.work += atom.work[0]
                self.total_work += atom.work[0]

                
    def hydrate(self, number_of_water):

        water_theta = 104.5/(2 * math.pi)
        l0 = 0.96

        waters_added = 0
        attempt_to_add_water = 0

        while waters_added < number_of_water:

            x = random.uniform(-1,1)
            y = random.uniform(-1,1)
            z = random.uniform(-1,1)
            
            random_pos = np.array([x,y,z])
            mag_r = mag(random_pos)
            unit_pos = random_pos / mag_r

            random_position = unit_pos * random.uniform(0.1, 0.80 * simulation.boundary)
            
            water_one = molecule()
            O1 = atom( atom_number = 3, element = 'oxygen'  , element_type = 'OT', x0 = random_position[0], y0 = random_position[1], z0 = random_position[2], test = True )
            H1 = atom( atom_number = 1, element = 'hydrogen', element_type = 'HT', test = True )
            H2 = atom( atom_number = 1, element = 'hydrogen', element_type = 'HT', test = True )

            H2.pos = np.array([math.sin(water_theta/2)*l0, math.cos(water_theta/2)*l0, 0])
            H1.pos = np.array([-math.sin(water_theta/2)*l0, math.cos(water_theta/2)*l0, 0])

            rotation_theta = random.uniform(0, 2 * math.pi)
            random_vec = np.random.random(3)
            mag_v = mag(random_vec)
            rotation_vector = random_vec / mag_v

            r = R.from_rotvec( rotation_theta * rotation_vector )
            r.as_matrix()
            
            H1.pos = r.apply(H1.pos)
            H2.pos = r.apply(H2.pos)

            H1.pos += O1.pos
            H2.pos += O1.pos

            self.get_potential()
            initial_potential = self.potential_energy
            self.potential_energy = 0.0

            for pair in testing_potentials.all_test_lennard_jones_pairs:
                self.potential_energy += lennard_jones.lennard_jones_potential(pair)

            for pair in testing_potentials.all_test_electrostatic_pairs:
                self.potential_energy += electrostatic.electrostatic_potential(pair)


            # print(initial_potential, abs(self.potential_energy - initial_potential))
            # sys.exit(20)
            
            if abs(self.potential_energy - initial_potential) < 1.25 * initial_potential: #should I actually hard code this?
                
                water_one.add_atom(H1)
                water_one.add_atom(H2)
                water_one.add_atom(O1)

                for water_atom in water_one.atoms:
                    for specific_atom in atom.all_atoms:
                        lennard_jones(water_atom, specific_atom)

                for water_atom in water_one.atoms:
                    visual(water_atom)
                    atom.all_atoms.append(water_atom)
                
                water_one.bond_atoms(H1, O1, 1)
                water_one.bond_atoms(H2, O1, 1)
                
                self.molecules.append(water_one)
                
                waters_added += 1
                attempt_to_add_water += 1

            else:

                atoms_to_remove = [O1, H1, H2]
                testing_potentials.remove_atoms(atoms_to_remove)
                
                attempt_to_add_water += 1

            print(" Hydration Status : {:.1f}% | Attempt # : {}".format(waters_added/number_of_water * 100, attempt_to_add_water))

            initial_potential = 0.0
            self.potential_energy = 0.0

    def print_final_positions(self, name):

        os.chdir(Simulation_config_path)
        file = open('{}.txt'.format(str(name)), 'w')

        file.write('#### Final MD Simulation Positions ####')
        file.write('#\n#\n')
        file.write('#Atom_Number:\tAtom_type:\tAtom_Element:\tPosition[x,y,z]:\n')
    
        for molecule in self.molecules:
            for atom in molecule.atoms:

                file.write(str(atom.atom_number))
                file.write('\t')
                file.write(str(atom.element_type))
                file.write('\t')
                file.write(str(atom.element))
                file.write('\t')
                file.write(str(atom.pos[0]))
                file.write('\t')
                file.write(str(atom.pos[1]))
                file.write('\t')
                file.write(str(atom.pos[2]))
                file.write('\n')

        file.write('\n#Bonds Formed:\n\n')

        file.write('# Atom #1 - Atom #2: Bond Order:\n\n')

        for molecule in self.molecules:
            for bond in molecule.bonds:
                file.write(str(bond.atom1.atom_number))
                file.write('-')
                file.write(str(bond.atom2.atom_number))
                file.write('\t')
                file.write(str(bond.number_of_bonds))
                file.write('\n')

        file.close()


class testing_potentials:
    """ 

    """
    
    Time_Unit = ( (1.66054E-27) * (1E-10)**2 / 1.602E-18 ) ** (1/2)
    Special_Epsilon = (6.2415E18)**2 * (1/Time_Unit)**2 / ( (1E-10)**3 * (1.66054E-27) ) 
    K = 1 / ( 4 * math.pi * Special_Epsilon)

    all_test_atoms = []
    all_test_lennard_jones_pairs = []
    all_test_electrostatic_pairs = []

    epsilon_i = { 'CC33A' : -0.0780 * (2.611E22/6.022E23) ,
                  'HCA3A' : -0.0240 * (2.611E22/6.022E23) ,
                  'HT'    : -0.046  * (2.611E22/6.022E23) ,
                  'HCA2A' : -0.0350 * (2.611E22/6.022E23) ,
                  'CC32A' : -0.0560 * (2.611E22/6.022E23) ,
                  'OT'    : -0.1521 * (2.611E22/6.022E23) ,
                  'CT1'   : -0.2000 * (2.611E22/6.022E23) ,
                  'CT3'   : -0.0800 * (2.611E22/6.022E23) ,
                  'NH1'   : -0.2000 * (2.611E22/6.022E23) ,
                  'C'     : -0.1100 * (2.611E22/6.022E23) ,
                  'H'     : -0.0460 * (2.611E22/6.022E23) ,
                  'HB'    : -0.0220 * (2.611E22/6.022E23) ,
                  'O'     : -0.1200 * (2.611E22/6.022E23) ,
                  'HA'    : -0.0220 * (2.611E22/6.022E23) }

    r_min_i = { 'CC33A' : 2.0400 ,
                'HCA3A' : 1.3400 ,
                'HT'    : 0.2245 ,
                'HCA2A' : 1.3400 ,
                'CC32A' : 2.0100 ,
                'OT'    : 1.7682 ,
                'CT1'   : 2.2750 ,
                'CT3'   : 2.0600 ,
                'NH1'   : 1.8500 ,
                'C'     : 2.0000 ,
                'H'     : 0.2245 ,
                'HB'    : 1.3200 ,
                'O'     : 1.7000 ,
                'HA'    : 1.3200 }

    charge = { 'CT3' : -0.27,
               'HA'  :  0.09,
               'C'   :  0.51,
               'O'   : -0.51,
               'NH1' : -0.47,
               'H'   :  0.31,
               'CT1' :  0.07,
               'HB'  :  0.09,
               'OT'  : -0.834,
               'HT'  :  0.417 }


    def __init__(self, atom_1, atom_2, interaction_type = 'lennard-jones'):

        self.interaction_type = interaction_type
        self.atom1 = atom_1
        self.atom2 = atom_2

        if self.interaction_type == 'lennard-jones':
            self.atom1_potential_counted = False
            self.atom1_force_counted = False
            self.atom2_potential_counted = False
            self.atom2_force_counted = False
        
            self.Epsilon = (2.6114E22/6.022E23) * math.sqrt( lennard_jones.epsilon_i[atom_1.element_type] *
                                                         lennard_jones.epsilon_i[atom_2.element_type] )

        
            self.R_min = ( lennard_jones.r_min_i[atom_1.element_type]/2 +
                       lennard_jones.r_min_i[atom_2.element_type]/2 )
        
            testing_potentials.all_test_lennard_jones_pairs.append(self)


        if self.interaction_type == 'electrostatic':

            self.charge1 = electrostatic.charge[self.atom1.element_type]
            self.charge2 = electrostatic.charge[self.atom2.element_type]
            self.atom1_force_counted = False
            self.atom1_potential_counted = False
            self.atom2_force_counted = False
            self.atom2_potenial_counted = False
            testing_potentials.all_test_electrostatic_pairs.append(self)

    def remove_atoms(list_of_atoms):

        for atom_to_delete in list_of_atoms:
            
            for pair in testing_potentials.all_test_lennard_jones_pairs:
                if pair.atom1 or pair.atom2 == atom_to_delete:
                    testing_potentials.all_test_lennard_jones_pairs.remove(pair)

            for pair in testing_potentials.all_test_electrostatic_pairs:
                if pair.atom1 or pair.atom2 == atom_to_delete:
                    testing_potentials.all_test_electrostatic_pairs.remove(pair)

            testing_potentials.all_test_atoms.remove(atom_to_delete)
            
class atom:

    all_atoms = []
    
    masses = { 'hydrogen' : 1,
               'oxygen'   : 16,
               'carbon'   : 12,
               'nitrogen' : 10 }
    
    potential_bonds = { 'hydrogen' : 1,
                        'oxygen'   : 2,
                        'carbon'   : 4,
                        'nitrogen' : 3 }
    
    def __init__(self, element, element_type = 'None Assigned',  charge = None, molecule = None, atom_number = 0,
                       x0 = 0., y0 = 0., z0 = 0., vx0 = 0., vy0 = 0., vz0 = 0., test = False):
        
        self._pos = np.array([x0, y0, z0])
        self._vel = np.array([vx0, vy0, vz0])
        self._work = np.zeros(1)
        self.element = element
        self.mass = atom.masses[self.element]
        self.atom_number = atom_number
        self.molecule = molecule
        self.element_type = element_type
        self.charge = charge
        self.force = np.zeros(3)
        self.potential_bonds = atom.potential_bonds[self.element]
        self.bonds = []
        self.test = test

        if len(atom.all_atoms) > 1:    
            for specific_atom in atom.all_atoms:
                if self.test == False:
                    lennard_jones(self, specific_atom)
                    electrostatic(self, specific_atom)
                    testing_potentials(atom_1 = self, atom_2 = specific_atom, interaction_type = 'lennard-jones') 
                    testing_potentials(atom_1 = self, atom_2 = specific_atom, interaction_type = 'electrostatic')            
                    
                if self.test == True:
                    testing_potentials(atom_1 = self, atom_2 = specific_atom, interaction_type = 'lennard-jones') 
                    testing_potentials(atom_1 = self, atom_2 = specific_atom, interaction_type = 'electrostatic')

        # if self.test == False and self.visualize == True:
        #     visual(self)
            
        if self.test == False:
            atom.all_atoms.append(self)
            
        testing_potentials.all_test_atoms.append(self)
        
    def move_data_to_buffers(self, posbuffer, velbuffer, workbuffer):

        pos = self._pos.copy()
        vel = self._vel.copy()
        work = self._work.copy()
        
        self._pos = np.frombuffer(posbuffer)
        self._vel = np.frombuffer(velbuffer)
        self._work = np.frombuffer(workbuffer)

        self._pos[:] = pos
        self._vel[:] = vel
        self._work[:] = work
               
    @property
    def pos(self):
        return self._pos
    
    @pos.setter
    def pos(self, value):
        self._pos[:] = value
        
    @property
    def vel(self):
        return self._vel
    
    @vel.setter
    def vel(self, value):
        self._vel[:] = value

    @property
    def work(self):
        return self._work

    @work.setter
    def work(self, value):
        self._work[:] = value
        
        
class molecule():

    dampening_coef = 0.7  # I typically use 1.0 for 'good' results
    
    all_molecules = []

    atom_number = 0
    
    def __init__(self):
        self.atoms = []
        self.bonds = []
        self.bond_angles = []
        self.dihedrals = []
        self.array_packing_offset = 0
        molecule.all_molecules.append(self)
        
    def add_atom(self, atom):
        self.atoms.append(atom)
        atom.molecule = self
        
    def bond_atoms(self, atom1, atom2, multiplicity):
        
        if not atom1 in self.atoms:
            raise Exception("I ain't got {}, atom number {}. Try molecule {}".format(atom2.element, atom2.atom_number, atom2.molecule))
        if not atom2 in self.atoms:
            raise Exception("I ain't got {}, atom number {}. Try molecule {}".format(atom2.element, atom2.atom_number, atom2.molecule))
        
        new_bond = bond(atom1, atom2, number_of_bonds = multiplicity)

        new_dihedrals_maybe = False
        
        for other_bond in self.bonds:
            
            if atom1 == other_bond.atom1 or atom1 == other_bond.atom2:
                new_bond_angle = bond_angle(other_bond, new_bond)
                new_dihedrals_maybe = True
                self.bond_angles.append(new_bond_angle)
                
            if atom2 == other_bond.atom1 or atom2 == other_bond.atom2:
                new_bond_angle = bond_angle(other_bond, new_bond)
                new_dihedrals_maybe = True
                self.bond_angles.append(new_bond_angle)

            if new_dihedrals_maybe == True:
                for other_bond_angle in self.bond_angles:
                    bond_angle_1 = new_bond_angle.ordered_list_of_atoms()
                    bond_angle_2 = other_bond_angle.ordered_list_of_atoms()
                    center_atom_1 = bond_angle_1[1]
                    center_atom_2 = bond_angle_2[1]
                    if center_atom_1 != center_atom_2:
                        
                        if new_bond_angle.bond1 == other_bond_angle.bond1 or new_bond_angle.bond1 == other_bond_angle.bond2:
                            new_dihedral = dihedral(other_bond_angle, new_bond_angle)
                            self.dihedrals.append(new_dihedral)
                    
                        elif new_bond_angle.bond2 == other_bond_angle.bond1 or new_bond_angle.bond2 == other_bond_angle.bond2:
                            new_dihedral = dihedral(other_bond_angle, new_bond_angle)
                            self.dihedrals.append(new_dihedral)

                new_dihedrals_maybe = False
                                                    
        self.bonds.append(new_bond)
            
    def clear_all_counted(self):
        for bon in self.bonds:
            bon.counted = False
        for bangle in self.bond_angles:
            bangle.counted = False
        for dangle in self.dihedrals:
            dangle.counted = False
        
            
    def get_all_forces(self):
        """ Takes a molecule and calculates all bond and bond angle
        forces for all of the atoms in the molecule. """

        """ Set all the forces on all of the atoms to 0 """
        
        for at in self.atoms:
            at.force[:] = 0.

        """ Bond Forces """
        
        for specific_bond in self.bonds:
            if specific_bond.counted == False:

                specific_bond.bond_force()

        """ Bond Angle Forces """
        
        for specific_bond_angle in self.bond_angles:
            if specific_bond_angle.counted == False:

                specific_bond_angle.bond_angle_force()

        """ Dihedral Forces """
                
        for specific_dihedral in self.dihedrals:
            if specific_dihedral.counted == False:
                
                specific_dihedral.dihedral_force()
                
        """ Lennard Jones Forces """
                
        # for specific_atom in self.atoms:

        #     for specific_lennard_jones_pair in lennard_jones.all_lennard_jones_pairs:

        #         if specific_atom == specific_lennard_jones_pair.atom1 and specific_lennard_jones_pair.atom1_force_counted == False:
        #             forces = specific_lennard_jones_pair.lennard_jones_force()
        #             specific_lennard_jones_pair.atom1.force += forces[0]
        #             specific_lennard_jones_pair.atom1_force_counted = True
                    
        #         if specific_atom == specific_lennard_jones_pair.atom2 and specific_lennard_jones_pair.atom2_force_counted == False:
        #             forces = specific_lennard_jones_pair.lennard_jones_force()
        #             specific_lennard_jones_pair.atom2.force += forces[1]
        #             specific_lennard_jones_pair.atom2_force_counted = True

        """ Electrostatic Potential Forces """
        
        # for specific_atom in self.atoms:
        #     for specific_electrostatic_pair in electrostatic.all_electrostatic_pairs:

        #         if specific_atom == specific_electrostatic_pair.atom1:
        #             forces = electrostatic.electrostatic_force(specific_electrostatic_pair)
        #             specific_electrostatic_pair.atom1.force += forces[0]

        #         if specific_atom == specific_electrostatic_pair.atom2:
        #             forces = electrostatic.electrostatic_force(specific_electrostatic_pair)
        #             specific_electrostatic_pair.atom2.force += forces[1]

                    
class bond:
    
    k_bond = {} #(eV/A^2)
              
    E0_bond = {} #angstroms
     
    def __init__(self, atom1, atom2, counted = False, number_of_bonds = 1):
        self.atom1 = atom1
        self.atom2 = atom2

        if self.atom1.potential_bonds < len(self.atom1.bonds):
            print('A {} is going to be bonded to {} other atoms.'.format(self.atom1.element, len(self.atom1.bonds + 1)))

        if self.atom2.potential_bonds < len(self.atom2.bonds):
            print('A {} is going to be bonded to {} other atoms.'.format(self.atom2.element, len(self.atom2.bonds + 1)))
            
        self.counted = counted
        self.number_of_bonds = number_of_bonds
        
        atom1.bonds.append(self)
        atom2.bonds.append(self)

        try:
            self.k_bond = bond.k_bond["{}-{}".format(self.atom1.element_type, self.atom2.element_type)]
        except:
            self.k_bond = bond.k_bond["{}-{}".format(self.atom2.element_type, self.atom1.element_type)]
        finally:
            if self.k_bond == None:
                sys.stderr.write('Unable to find a bond spring coefficient for atoms : {}-{}, atom numbers : {}-{}, and element types : {}-{}'.format(self.atom1.element, self.atom2.element,
                         self.atom1.atom_number, self.atom2.atom_number,
                         self. atom1.element_type, self.atom2.element_type))

        try:
            self.E0_bond = bond.E0_bond["{}-{}".format(self.atom1.element_type, self.atom2.element_type)]
        except:
            self.E0_bond = bond.E0_bond["{}-{}".format(self.atom2.element_type, self.atom1.element_type)]
        finally:
            if self.k_bond == None:
                sys.stderr.write('Unable to find a equilibrium bond lengths for atoms : {}-{}, atom numbers : {}-{}, and element types : {}-{}'.format(self.atom1.element, self.atom2.element,
                          self.atom1.atom_number, self.atom2.atom_number,
                          self.atom1.element_type, self.atom2.element_type))

        # if self.atom1.visualize == True and self.atom2.visualize == True:
        #     visual(self)
    
    def bond_potential(self):
        """ Calculates and returns bond potential """
            
        distance = mag(self.atom2.pos - self.atom1.pos)
        potential =  0.5 * self.k_bond * (distance - self.E0_bond)**2
        
        return potential
    
    def bond_force(self):
        """ Calculates the spring bond potential and returns dU """
                                                                              
        r_vec = (self.atom2.pos - self.atom1.pos)
        r_hat = r_vec/mag(r_vec)
        dU = self.k_bond * (mag(r_vec) - self.E0_bond)
        Force_1 = dU * r_hat
        Force_2 = -1 * Force_1

        self.atom1.force += Force_1
        self.atom2.force += Force_2

        self.atom1.counted = True
        self.atom2.counted = True
    
class bond_angle:
    
    k_bond_angle = {}  #(eV/rad^2)
    
    bond_angle = {} #radians
        
    def __init__(self, bond1, bond2, counted = False):
        self.bond1 = bond1
        self.bond2 = bond2
        self.counted = counted
              
        if bond1.atom1 != bond2.atom1 and bond1.atom1 != bond2.atom2 and bond1.atom2 != bond2.atom1 and bond1.atom2 != bond2.atom2:
            raise Exception('You fuckin\' dooftard. These bonds do not have a common atom!')

        typedex_list = bond_angle.ordered_list_of_atoms(self)

        self.start_atom = typedex_list[0]
        self.middle_atom = typedex_list[1]
        self.end_atom = typedex_list[2]
                  
        self.typedex = "{}-{}-{}".format(self.start_atom.element_type,  
                                         self.middle_atom.element_type,
                                         self.end_atom.element_type)

        self.reverse_typedex = "{}-{}-{}".format(self.end_atom.element_type,  
                                                 self.middle_atom.element_type,
                                                 self.start_atom.element_type)


        if self.typedex in bond_angle.k_bond_angle:
            self.k_bond_angle = bond_angle.k_bond_angle[self.typedex]
        elif self.reverse_typedex in bond_angle.k_bond_angle:
            if self.typedex != self.reverse_typedex:
                    self.k_bond_angle = bond_angle.k_bond_angle[self.reverse_typedex]
        else:
            sys.stderr.write('Unable to find a bond angle spring coefficient for atoms : {}-{}-{}, atom numbers : {}-{}-{}, and element types : {}-{}-{}\n\n'.format(self.start_atom.element, self.middle_atom.element, self.end_atom.element,
                                    self.start_atom.atom_number, self.middle_atom.atom_number, self.end_atom.atom_number,
                                    self.start_atom.element_type, self.middle_atom.element_type, self.end_atom.element_type))

                

        if self.typedex in bond_angle.bond_angle:
            self.bond_angle0 = bond_angle.bond_angle[self.typedex]
        elif self.reverse_typedex in bond_angle.bond_angle:
            if self.typedex != self.reverse_typedex:
                self.bond_angle0 = bond_angle.bond_angle[self.reverse_typedex]
        else:
            sys.stderr.write('Unable to find a bond angle equilibrium angle for atoms : {}-{}-{}, atom numbers : {}-{}-{}, and element types : {}-{}-{}\n\n'.format(self.start_atom.element, self.middle_atom.element, self.end_atom.element,
                                   self.start_atom.atom_number, self.middle_atom.atom_number, self.end_atom.atom_number,
                                   self.start_atom.element_type, self.middle_atom.element_type, self.end_atom.element_type))

            sys.exit(20)
         
                                                
    def ordered_list_of_atoms(self):
        """ Determines which atom in a pair of bonds
        ( so a bond angle) is the central atom.  This is used
        to eliminate the possibility of redundnacy and overcounting
        forces """
        
        if self.bond1.atom1 == self.bond2.atom1:
            center_atom = self.bond1.atom1
            atom1 = self.bond1.atom2
            atom3 = self.bond2.atom2
                     
        elif self.bond1.atom1 == self.bond2.atom2:      
            center_atom = self.bond1.atom1
            atom1 = self.bond1.atom2
            atom3 = self.bond2.atom1
                     
        elif self.bond1.atom2 == self.bond2.atom1:
            center_atom = self.bond1.atom2
            atom1 = self.bond1.atom1
            atom3 = self.bond2.atom2
                     
        elif self.bond1.atom2 == self.bond2.atom2:
            center_atom = self.bond1.atom2
            atom1 = self.bond1.atom1
            atom3 = self.bond2.atom1
                     
        else:
            raise Exception('No central atom for two bonds that don\'t share an atom!')

        atom_order_list = [atom1, center_atom, atom3]
        
        return atom_order_list
    
    def bond_angle_potential(self):
        """ Calculates and returns bond angle potential """

        bond1_vec = self.start_atom.pos - self.middle_atom.pos
        bond2_vec = self.end_atom.pos - self.middle_atom.pos

        try:
            tmpshit = mag( cross_product(bond1_vec, bond2_vec) ) / ( mag(bond1_vec) * mag(bond2_vec))
            theta = math.asin( tmpshit )
            
        except ValueError as err:
            
            if tmpshit > 1 : tmpshit = 1
            elif tmpshit < -1 : tmpshit = -1
            else:
                sys.stderr.write('This should never happen... \n')
                raise Exception(err)
            theta = math.asin( tmpshit )
            
        try:
            tmpshit = (bond1_vec * bond2_vec).sum() / ( mag(bond1_vec) * mag(bond2_vec) )
            theta = math.acos( tmpshit )
            
        except ValueError as err:
            if tmpshit > 1 : tmpshit = 1
            if tmpshit < -1 : tmpshit = -1
            else:
                raise Exception ('HOLY FUCK {} {} \n'.format(bond1_vec, bond2_vec))
            theta = math.acos( tmpshit )
        
        potential = ( 0.5 * self.k_bond_angle * (theta - self.bond_angle0)**2 )
        
        return potential
    
    def bond_angle_force(self):
        """ Calculates the bond angle potential and returns 
            (Force_1, Force_2, F_center_atom (the force on the center
             atom) """
        
        """ Calculate the bonds linking the center atom to the outer atoms """
        
        bond1_vec = self.start_atom.pos - self.middle_atom.pos
        bond2_vec = self.end_atom.pos - self.middle_atom.pos

        try:
            tmpshit = mag( cross_product(bond1_vec, bond2_vec) ) / ( mag(bond1_vec) * mag(bond2_vec))
            theta = math.asin( tmpshit )
            
        except ValueError as err:
            
            if tmpshit > 1 : tmpshit = 1
            elif tmpshit < -1 : tmpshit = -1
            else:
                sys.stderr.write('This should never happen... \n')
                raise Exception(err)
            theta = math.asin( tmpshit )
            
        try:
            tmpshit = (bond1_vec * bond2_vec).sum() / ( mag(bond1_vec) * mag(bond2_vec) )
            theta = math.acos( tmpshit )
            
        except ValueError as err:
            if tmpshit > 1 : tmpshit = 1
            if tmpshit < -1 : tmpshit = -1
            else:
                raise Exception ('HOLY FUCK {} {} \n'.format(bond1_vec, bond2_vec))
            theta = math.acos( tmpshit )
            
        """ Calculate the potential with respect to the theta """
        dV_dtheta =  self.k_bond_angle * (theta - self.bond_angle0)
        
        """ Calculate the torques based on -dU/dTheta """
        Torque_1_val = -1*dV_dtheta
        Torque_2_val = -1*Torque_1_val
        
        """ Get the forces on the atoms and central atom """
        mag1 = mag(bond1_vec)
        mag2 = mag(bond2_vec)

        b1b2mag = mag(cross_product(bond1_vec, bond2_vec))

        if b1b2mag < 1e-12:
            Force_1 = np.zeros(3)
            Force_2 = np.zeros(3)
            F_center_atom = np.zeros(3)

        else:
            a_hat = cross_product(bond1_vec, bond2_vec) / b1b2mag
            F1_hat = cross_product(bond1_vec, a_hat) / mag(cross_product(bond1_vec, a_hat))
            F1_val = Torque_1_val / mag1
            Force_1 = F1_hat*F1_val
        
            F2_hat = cross_product(bond2_vec, a_hat) / mag(cross_product(bond2_vec, a_hat))
            F2_val = Torque_2_val / mag2
            Force_2 = F2_hat*F2_val
        
            F_center_atom = -1 * (Force_1 + Force_2)


        # sys.stderr.write("Force_1={}, Force_2={}, F_center_atom={}\n".format(Force_1, Force_2, F_center_atom))
        if np.any(np.isnan(Force_1)):
            import pdb; pdb.set_trace()
        
        self.start_atom.force += Force_1
        self.middle_atom.force += F_center_atom
        self.end_atom.force += Force_2

        self.start_atom.counted = True
        self.middle_atom.counted = True
        self.end_atom.counted = True

class dihedral:

    k_chi = {}

    delta = {}

    n = {}
    
    def __init__(self, bond_angle1, bond_angle2, counted = False):
        self.bond_angle1 = bond_angle1
        self.bond_angle2 = bond_angle2

        self.delete_all_references_to_this_variable = 0
        
        if bond_angle1.middle_atom == bond_angle2.start_atom:
            if bond_angle1.end_atom == bond_angle2.middle_atom:
                self.first_atom = bond_angle1.start_atom
                self.second_atom = bond_angle1.middle_atom
                self.third_atom = bond_angle2.middle_atom
                self.fourth_atom = bond_angle2.end_atom

            if bond_angle1.start_atom == bond_angle2.middle_atom:
                self.first_atom = bond_angle1.end_atom
                self.second_atom = bond_angle1.middle_atom
                self.third_atom = bond_angle2.middle_atom
                self.fourth_atom = bond_angle2.end_atom

        if bond_angle1.middle_atom == bond_angle2.end_atom:
            if bond_angle1.end_atom == bond_angle2.middle_atom:
                self.first_atom = bond_angle1.start_atom
                self.second_atom = bond_angle1.middle_atom
                self.third_atom = bond_angle2.middle_atom
                self.fourth_atom = bond_angle2.start_atom

            if bond_angle1.start_atom == bond_angle2.middle_atom:
                self.first_atom = bond_angle1.end_atom
                self.second_atom = bond.angle1.middle_atom
                self.third_atom = bond_angle2.middle_atom
                self.fourth_atom = bond_angle2.start_atom
                
        
        self.shared_atoms = [self.second_atom, self.third_atom]
        self.counted = counted
        
        self.typedex = "{}-{}-{}-{}".format(self.first_atom.element_type,
                                            self.second_atom.element_type,
                                            self.third_atom.element_type,
                                            self.fourth_atom.element_type)

        self.reverse_typedex = "{}-{}-{}-{}".format(self.fourth_atom.element_type,
                                            self.third_atom.element_type,
                                            self.second_atom.element_type,
                                            self.first_atom.element_type)


        if bond_angle1.bond1 != bond_angle2.bond1 and bond_angle1.bond1 != bond_angle2.bond2 and bond_angle1.bond2 != bond_angle2.bond1 and bond_angle1.bond2 != bond_angle2.bond2:
            raise Exception('You really fucked up. These bond angles do not share a common bond!')

        if self.first_atom.element == 'hydrogen' or self.second_atom.element == 'hydrogen' or self.third_atom.element == 'hydrogen' or self.fourth_atom.element == 'hydrogen':
            hydrogen_present = True
        else:
            hydrogen_present = False

        no_n = False
        no_delta = False
        no_chi = False
        
        if self.typedex in dihedral.n:
            self.n = dihedral.n[self.typedex]
        elif self.reverse_typedex in dihedral.n:
            if self.typedex != self.reverse_typedex:
                self.n = dihedral.n[self.reverse_typedex]
        else:
            self.n = 1
            no_n = True
            
            # if hydrogen_present == False:
            #     sys.stderr.write('Could not find a n for the dihedral {} or {}, defaulted to 1\n'.format(self.typedex, self.reverse_typedex))
                

        if self.typedex in dihedral.delta:
            self.delta = dihedral.delta[self.typedex]
        elif self.reverse_typedex in self.delta:
            if self.typedex != self.reverse_typedex:
                self.delta = dihedral.delta[self.reverse_typedex]
        else:
            self.delta = 0.
            no_delta = True
            
            # if hydrogen_present == False:
            #     sys.stderr.write('Could not find a delta for the dihedral {} or {}, defaulted to 0\n'.format(self.typedex, self.reverse_typedex))

        if self.typedex in dihedral.k_chi:
            self.k_chi = dihedral.k_chi[self.typedex]
        elif self.reverse_typedex in dihedral.k_chi:
            if self.typedex != self.reverse_typedex:
                self.k_chi = dihedral.k_chi[self.reverse_typedex]
        else:
            self.k_chi = 0.
            no_chi = True
            
            # if hydrogen_present == False:
            #     sys.stderr.write('COuld not find a k_chi for the dihedral {} or {}, defaulted to 0\n'.format(self.typedex, self.reverse_typedex))

        if hydrogen_present == False:
            if no_n == True and no_delta == True and no_chi == True:
                sys.stderr.write('Could not find a n, delta, or chi for the dihedral {} or {}, defaulted to 1\n'.format(self.typedex, self.reverse_typedex))

        if no_n == True and no_delta == False:
            sys.exit(20)

        if no_n == True and no_chi == False:
            sys.exit(20)

        if no_delta == True and no_n == False:
            sys.exit(20)

        if no_chi == True and no_n == False:
            sys.exit(20)

        if no_delta == True and no_chi == False:
            sys.exit(20)

        if no_chi == True and no_delta == False:
            sys.exit(20)
    

    def dihedral_potential(self):
        
        ''' Calculates the potential from the dihedrals '''
        
        a_vec = self.first_atom.pos - self.second_atom.pos
        b_vec = self.second_atom.pos - self.third_atom.pos
        c_vec = self.fourth_atom.pos - self.third_atom.pos

        a_mag = mag(a_vec)
        b_mag = mag(b_vec)
        c_mag = mag(c_vec)

        a_unit_vec = a_vec / a_mag
        b_unit_vec = b_vec / b_mag
        c_unit_vec = c_vec / c_mag

        a_perp = a_vec - b_unit_vec * ( a_vec[0] * b_unit_vec[0] +
                                        a_vec[1] * b_unit_vec[1] +
                                        a_vec[2] * b_unit_vec[2] )

        c_perp = c_vec - b_unit_vec * ( c_vec[0] * b_unit_vec[0] +
                                        c_vec[1] * b_unit_vec[1] +
                                        c_vec[2] * b_unit_vec[2] )


        acos_argument = ((a_perp[0] * c_perp[0] +
                          a_perp[1] * c_perp[1] +
                          a_perp[2] * c_perp[2])

                         /

                         ( mag(a_perp) * mag(c_perp) ) )

        if acos_argument < -1.0:
            acos_argument = -1.0

        if acos_argument > 1.0:
            acos_argument = 1.0
        
        chi = math.acos( acos_argument )

        cross = cross_product(a_perp, c_perp)

        if ( cross[0] * b_vec[0] +
             cross[1] * b_vec[1] +
             cross[2] * b_vec[2]) > 0:
            chi = chi
        else:
            chi = -1*chi

        potential =  self.k_chi * ( 1 + math.cos( self.n * chi - self.delta ) )

        return potential

    def dihedral_force(self):
        ''' Calculates the forces on all atoms in the dihedrals.

            Returns a list of the forces in this order:
                      
                    1.) Start Atom
                    2.) Central Bond Atom 1
                    3.) Central Bond Atom 2
                    4.) End Atom                                 '''

        if self.k_chi == 0.:
            self.first_atom.counted = True
            self.second_atom.counted = True
            self.third_atom.counted = True
            self.fourth_atom.counted = True
            return
        
        a_vec = self.first_atom.pos - self.second_atom.pos
        b_vec = self.second_atom.pos - self.third_atom.pos
        c_vec = self.fourth_atom.pos - self.third_atom.pos

        a_mag = mag(a_vec)
        b_mag = mag(b_vec)
        c_mag = mag(c_vec)

        a_unit_vec = a_vec / a_mag
        b_unit_vec = b_vec / b_mag
        c_unit_vec = c_vec / c_mag

        a_perp = a_vec - b_unit_vec * ( a_vec[0] * b_unit_vec[0] +
                                        a_vec[1] * b_unit_vec[1] +
                                        a_vec[2] * b_unit_vec[2] )

        c_perp = c_vec - b_unit_vec * ( c_vec[0] * b_unit_vec[0] +
                                        c_vec[1] * b_unit_vec[1] +
                                        c_vec[2] * b_unit_vec[2] )


        acos_argument = ((a_perp[0] * c_perp[0] +
                          a_perp[1] * c_perp[1] +
                          a_perp[2] * c_perp[2])

                         /

                         ( mag(a_perp) * mag(c_perp) ) )

        if acos_argument < -1.0:
            acos_argument = -1.0

        if acos_argument > 1.0:
            acos_argument = 1.0
        
        chi = math.acos( acos_argument )

        cross = cross_product(a_perp, c_perp)

        if ( cross[0] * b_vec[0] +
             cross[1] * b_vec[1] +
             cross[2] * b_vec[2]) > 0:
            chi = chi
        else:
            chi = -1*chi

        dU_dChi = -1*self.n * self.k_chi * math.sin( self.n * chi - self.delta )
        
        if math.fabs(dU_dChi) < 1E-7:
            dU_dChi = 0.
            F_a_vec = 0.
            Summed_Forces = 0.
            Summed_Forces = 0.
            F_c_vec = 0.
            

        # ''' Note: This gives us the values of dU, but now we need the direction
        #           and magnitude of the torque. The magnitude of the torque is based
        #           on the dU value and the length of bond from the outer atom to
        #           the nearest atom in the central bond. To get the direction of
        #           the torque, we can use the same procedure that we did when we 
        #           get the bond angle. First, we need to get the cross product
        #           between the two perpendicular vectors (a_perp and c_perp). 
        #           Then we can take this cross product, get the hat vector, and cross
        #           it with the hat of each perpendicular vector. The next step is
        #           is to multiply this by the value of force and return this value. '''

        else:
            torque_A_val = dU_dChi
            torque_C_val = torque_A_val
            torque_A_hat = b_unit_vec
            torque_C_hat = -b_unit_vec
            torque_A_vec = torque_A_val * torque_A_hat
            torque_C_vec = torque_C_val * torque_C_hat

            F_a_hat = cross_product(torque_A_vec, a_vec) / ( mag(torque_A_vec) * mag(a_vec))
            F_a_val = math.fabs(torque_A_val) / mag(a_perp)
            F_a_vec = F_a_val * F_a_hat

            F_c_hat = cross_product(torque_C_vec, c_vec) / ( mag(torque_C_vec) * mag(c_vec))
            F_c_val = math.fabs(torque_C_val) / mag(c_perp)
            F_c_vec = F_c_val * F_c_hat

            ''' Note: The list of forces that I am returning are for the first atom,
                  second atom, third atom, and fourth atom. Also, the way the middle
                  atom forces are being calculated is just through the summing the forces
                  on the start and end atoms. Then we are applying  these forces to the
                  middle two atoms. '''

        self.first_atom.force += F_a_vec
        self.second_atom.force += -F_a_vec
        self.third_atom.force += -F_c_vec
        self.fourth_atom.force += F_c_vec
        
        self.first_atom.counted = True
        self.second_atom.counted = True
        self.third_atom.counted = True
        self.fourth_atom.counted = True

class lennard_jones:

    all_lennard_jones_pairs = []

    epsilon_i = {}

    r_min_i = {}

    def __init__(self, atom_1, atom_2):
        
        self.atom1 = atom_1
        self.atom1_potential_counted = False
        self.atom1_force_counted = False
        self.atom2 = atom_2
        self.atom2_potential_counted = False
        self.atom2_force_counted = False
        
        self.Epsilon = (2.6114E22/6.022E23) * math.sqrt( lennard_jones.epsilon_i[atom_1.element_type] *
                                                         lennard_jones.epsilon_i[atom_2.element_type] )

        
        self.R_min = ( lennard_jones.r_min_i[atom_1.element_type]/2 +
                       lennard_jones.r_min_i[atom_2.element_type]/2 )
        
        lennard_jones.all_lennard_jones_pairs.append(self)

    def lennard_jones_potential(self):
        
        r_vec = self.atom2.pos - self.atom1.pos
        mag_r = mag(r_vec)
        potential = self.Epsilon * ( (self.R_min/mag_r)**12 - 2 * (self.R_min/mag_r)**6)
        
        self.atom1_potential_counted = True
        self.atom2_potentail_counted = True
        
        return potential

    def lennard_jones_force(self):
        
        r_vec = self.atom1.pos - self.atom2.pos
        mag_r = mag(r_vec)
        r_hat = r_vec/mag_r
        
        dU_dr = -12 * self.Epsilon * ( ( (self.R_min ** 12) / (mag_r ** 13) ) -
                                      ( (self.R_min ** 6)  / (mag_r ** 7) ) )
        Force_1 = - dU_dr * r_hat
        Force_2 = - Force_1

        Force_list = [Force_1, Force_2]
        
        self.atom1.force += Force_1
        self.atom2.force += Force_2

        self.atom1_potential_counted = True
        self.atom2_potential_counted = True

        return Force_list


class electrostatic:
    
    Time_Unit = ( (1.66054E-27) * (1E-10)**2 / 1.602E-18 ) ** (1/2)
    Special_Epsilon = (6.2415E18)**2 * (1/Time_Unit)**2 / ( (1E-10)**3 * (1.66054E-27) ) 
    K = 1 / ( 4 * math.pi * Special_Epsilon)
    
    all_electrostatic_pairs = []

    charge = { 'C'   :  0.51,
               'CA'  : -0.115,
               'CC'  :  0.55,
               'CP1' :  0.02,
               'CP2' : -0.18,
               'CP3' :  0.00,
               'CPH1': -0.05,
               'CPH2':  0.25,
               'CPT' : 0.13,
               'CT3' : -0.27,
               'CT2' :  0.20,
               'CT1' :  0.07,
               'CY'  : -0.03,
               'H'   :  0.31,
               'HA'  :  0.09,
               'HB'  :  0.09,
               'HC'  :  0.44,
               'HP'  :  0.115,
               'HR1' :  0.13,
               'HR2' :  0.18,
               'HR3' :  0.10,
               'HS'  :  0.16,
               'HT'  :  0.417,
               'N'   : -0.29,
               'NC2' : -0.80,
               'NH1' : -0.47,
               'NH2' : -0.63,
               'NR1' : -0.68,
               'NR2' : -0.70,
               'NY'  : -0.61,
               'O'   : -0.51,
               'OC'  : -0.76,
               'OH1' : -0.66,
               'OT'  : -0.834,
               'S'   : -0.23 }

    def __init__(self, atom1, atom2):
        self.atom1 = atom1
        self.atom2 = atom2

        if self.atom1.charge == None:
            self.charge1 = electrostatic.charge[atom1.element_type]
        else:
            self.charge1 = atom1.charge

        if self.atom2.charge == None:
            self.charge2 = electrostatic.charge[atom2.element_type]
        else:
            self.charge2 = atom2.charge
            
        self.atom1_force_counted = False
        self.atom1_potential_counted = False
        self.atom2_force_counted = False
        self.atom2_potenial_counted = False
        electrostatic.all_electrostatic_pairs.append(self)

    def electrostatic_potential(self):
        
        r_vec = self.atom1.pos - self.atom2.pos
        mag_r = mag(r_vec)
        potential = electrostatic.K * self.charge1 * self.charge2 / ( mag_r )
        
        return potential

    def electrostatic_force(self):

        r_vec = self.atom1.pos - self.atom2.pos
        mag_r = mag(r_vec)
        r_hat = r_vec / mag_r
        dU_dr = -electrostatic.K * self.charge1 * self.charge2 / ( (mag_r)**2 )

        Force_1 = dU_dr * r_hat
        Force_2 = - Force_1

        Force_list = [Force_1, Force_2]

        return Force_list
        
               
    

class protein:

    amino_acid_directory = { 'A' : 'Alanine',
                             'R' : 'Arginine',
                             'N' : 'Asparagine',
                             'D' : 'Aspartic_Acid',
                             'C' : 'Cysteine',
                             'Q' : 'Glutamine',
                             'E' : 'Glutamic_Acid',
                             'G' : 'Glycine',
                             'H' : 'Histidine',
                             'I' : 'Isoleucine',
                             'L' : 'Leucine',
                             'K' : 'Lysine',
                             'M' : 'Methionine',
                             'F' : 'Phenylalanine',
                             'P' : 'Proline',
                             'S' : 'Serine',
                             'T' : 'Threonine',
                             'W' : 'Tryptophan',
                             'Y' : 'Tyrosine',
                             'V' : 'Valine' }

    def __init__(self, length):
        self.molecule = molecule()
        self.length = length
        self.amino_acids = []
        self.current_atom_number = 0        
        self.last_carbonyl_carbon = None
        self.rotation_theta = 0
        self.dtheta = math.pi/4
        
        # for i in range(self.length):
        #     self.amino_acids.append('No residue listed')

    def add_amino_acid(self, residue):
        specific_amino_acid = amino_acid(protein = self, position = len(self.amino_acids))

        if residue == 'Alanine':
            specific_amino_acid.residue_name = 'Alanine'
            specific_amino_acid.add_alanine()

        if residue == 'Arginine':
            specific_amino_acid.residue_name = 'Arginine'
            specific_amino_acid.add_arginine()

        if residue == 'Aspargine':
            specific_amino_acid.add_aspargine(self)

        if residue == 'Aspartic_Acid':
            specific_amino_acid.add_aspartic_acid(self)

        if residue == 'Cysteine':
            specific_amino_acid.add_cysteine(self)

        if residue == 'Glutamine':
            specific_amino_acid.add_glutamine(self)

        if residue == 'Glutaminic_Acid':
            specific_amino_acid.add_glutamic_acid(self)

        if residue == 'Glycine':
            specific_amino_acid.add_glycine(self)

        if residue == 'Histidine':
            specific_amino_acid.add_histidine(self)

        if residue == 'Isoleucine':
            specific_amino_acid.add_isoleucine(self)

        if residue == 'Leucine':
            specific_amino_acid.add_leucine(self)

        if residue == 'Lysine':
            specific_amino_acid.add_lysine(self)

        if residue == 'Methionine':
            specific_amino_acid.add_methionine(self)

        if residue == 'Phenylalanine':
            specific_amino_acid.add_phenylalanine(self)

        if residue == 'Proline':
            specific_amino_acid.add_proline(self)

        if residue == 'Serine':
            specific_amino_acid.add_serine(self)

        if residue == 'Threonine':
            specific_amino_acid.add_threonine(self)

        if residue == 'Tryptophan':
            specific_amino_acid.add_tryptophan(self)

        if residue == 'Tyrosine':
            specific_amino_acid.add_tyrosine(self)

        if residue == 'Valine':
            specific_amino_acid.add_valine(self)

        self.amino_acids.append(specific_amino_acid)
            

class amino_acid:

    def __init__(self, protein, position, residue = 'Glycine'):
        self.residue_name = residue
        self.protein = protein
        self.position = position
                    

    def save_residue_positions(self, t, tf):

        print_positions = True
        vel_min = 1E-2 # I typically use 1E-2 for the best results

        time = t

        for atom in self.protein.molecule.atoms:
            if mag(atom.vel) > vel_min:
                print_positions = False

        os.chdir(Resdiue_config_path)

        if print_positions == True:
            file = open('{}_Positions.txt'.format(str(self.residue_name)), 'w')

            file.write('# {} Residue Atom Positions Relative to Amine Nitrogen'.format(str(self.residue_name)))
            file.write('#\n#\n')
            file.write('##################################################################')
            file.write('#\n#\n')

            file.write('#Atom_Number:\tAtom_Type:\tAtom_Element:\tAtom_Charge:\tPosition[x,y,z]\n')

            for atom in self.protein.molecule.atoms:
                if atom.element_type == 'NH1':
                    amine_nitrogen_pos = atom.pos

            for atom in self.protein.molecule.atoms:
                position = atom.pos - amine_nitrogen_pos
                file.write(str(atom.atom_number))
                file.write('\t')
                file.write(str(atom.element_type))
                file.write('\t')
                file.write(str(atom.element))
                file.write('\t')
                file.write(str(atom.charge))
                file.write('\t')
                file.write(str(position[0]))
                file.write('\t')
                file.write(str(position[1]))
                file.write('\t')
                file.write(str(position[2]))                    
                file.write('\n')

            file.write('\n#Bonds in {}: \n\n'.format(str(self.residue_name)))
            file.write('# Atom #1 - Atom #2: Bond Order:\n\n')

            for bond in self.protein.molecule.bonds:

                file.write('{}'.format(str(bond.atom1.atom_number)))
                file.write('-')
                file.write('{}'.format(str(bond.atom2.atom_number)))
                file.write('\t')
                file.write(str(bond.number_of_bonds))
                file.write('\n')

                
            file.close()

            time = tf

        return time

    def read_residue_txt_files_for_atoms_and_bonds(self, txt_file):
        
        os.chdir(Resdiue_config_path)
    
        file_name = str(txt_file)
        file = open(file_name, 'r')

        rows = file.readlines()

        for line in rows:
            if line[0:5] == str('#Atom'):
                atom_position_start = rows.index(line) + 1

            if line[0:6] == str('#Bonds'):
                atom_position_stop = rows.index(line) - 2
                bonds_start = rows.index(line) + 4

        bonds_stop = len(rows)
        
        if self.protein.last_carbonyl_carbon != None:
            position_shift = self.protein.last_carbonyl_carbon.pos + np.array([0., 1., 0.])

        else:
            position_shift = np.zeros(3)

        rotation_vector = np.array([0., 1., 0.])
        rotation_theta = self.protein.rotation_theta

        r = R.from_rotvec( rotation_theta * rotation_vector )
        r.as_matrix()

        position_shift = r.apply(position_shift)

        for line in rows[atom_position_start: atom_position_stop + 1]:
            values = line.split('\t')
            last_value = values[6].strip('\n')
            values.pop(6)
            values.append(last_value)

            atom_pos = np.array([float(values[4]), float(values[5]), float(values[6])])
            atom_pos = r.apply(atom_pos)
            
            specific_atom = atom(atom_number = int(values[0]) + self.protein.current_atom_number , element_type = str(values[1]), element = str(values[2]), charge = float(values[3]), x0 = atom_pos[0] + position_shift[0], y0 = atom_pos[1] + position_shift[1], z0 = atom_pos[2] + position_shift[2])
            
            self.protein.molecule.add_atom(specific_atom)

            if specific_atom.element_type == str('C'):
                new_carbonyl_carbon = specific_atom

            if specific_atom.element_type == str('NH1'):
                new_amine_nitrogen = specific_atom

                
        for line in rows[bonds_start: bonds_stop + 1]:
            values = line.split('\t')
            atoms_in_bond = values[0].split('-')
            number_of_bonds = int(values[1].strip('\n'))
            formatted_values = []
            formatted_values.append(atoms_in_bond[0])
            formatted_values.append(atoms_in_bond[1])
            formatted_values.append(number_of_bonds)
        
            atom_1_number = int(formatted_values[0]) + self.protein.current_atom_number
            atom_2_number = int(formatted_values[1]) + self.protein.current_atom_number
            number_of_bonds = formatted_values[2]

            for this_atom in self.protein.molecule.atoms:
                
                if this_atom.atom_number == atom_1_number:
                    atom_1 = this_atom
                if this_atom.atom_number == atom_2_number:
                    atom_2 = this_atom

            self.protein.molecule.bond_atoms(atom_1, atom_2, number_of_bonds)

        self.protein.current_atom_number += atom_position_stop + 1 - atom_position_start

        if len(self.protein.amino_acids) > 0:
            self.protein.molecule.bond_atoms(new_amine_nitrogen, self.protein.last_carbonyl_carbon, 1)

        self.protein.last_carbonyl_carbon = new_carbonyl_carbon

        self.protein.rotation_theta += self.protein.dtheta


    def add_alanine(self):

        file_name = str('Alanine_Positions.txt')

        self.read_residue_txt_files_for_atoms_and_bonds(file_name)

    def add_arginine(self):

        file_name = str('Arginine_Positions.txt')

        self.read_residue_txt_files_for_atoms_and_bonds(file_name)


    def add_aspargine(self):
        pass

    def add_aspartic_acid(self):
        pass

    def add_cysteine(self):
        pass

    def add_glutamine(self):
        pass

    def add_glutamic_acid(self):
        pass

    def add_glycine(self):
        pass

    def add_histidine(self):
        pass

    def add_isoleucine(self):
        pass

    def add_leucine(self):
        pass

    def add_lysine(self):
        pass

    def add_methionine(self):
        pass

    def add_phenylalaine(self):
        pass

    def add_proline(self):
        pass

    def add_serine(self):
        pass

    def add_threonine(self):
        pass

    def add_tryptophan(self):
        pass

    def add_tyrosine(self):
        pass

    def add_valine(self):
        pass


class hydrocarbons():

    def __init__(self, molecule_type, name):
        if molecule_type == 'ethane':
            name = hydrocarbons.make_ethane()

    def make_ethane(self):
        s70 = math.sin(70/180 * math.pi)
        c70 = math.cos(70/180 * math.pi)
        s120 = math.sin(120/180 * math.pi)
        c120 = math.cos(120/180 * math.pi)
        s20 = math.sin(20/180 * math.pi)
        c20 = math.cos(20/180 * math.pi)

        l0 = 1.530
        l1 = 1.111
        h_bond = l1 * (1/2)**(1/2)

        h1pos = np.array( [ -l1*c70, -l1*s70, 0. ] )
        h2pos = np.array( [ -l1*c70, -l1*s70*c120, l1*s70*s120 ] )
        h3pos = np.array( [ -l1*c70, -l1*s70*c120, -l1*s70*s120 ] )

        crotang = math.cos( 20./180. * math.pi)
        srotang = math.sin( 20./180. * math.pi)
        rotmat = np.array( [ [ 1.,  0.,      0.      ],
                         [ 0., crotang, -srotang,],
                         [ 0., srotang,  crotang ]
                         ] )
        h1pos = np.matmul(rotmat, h1pos)
        h2pos = np.matmul(rotmat, h2pos)
        h3pos = np.matmul(rotmat, h3pos)

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

        if object.__class__.__name__ == 'atom':
            self.atom_object = object
            self.visual =  vis.sphere(pos = object.pos, radius = visual.atom_radii[object.element],
                              color = visual.atom_colors[object.element] )
            visual.all_visual_atoms.append(self)

        if object.__class__.__name__ == 'bond':
            self.bond_object = object
            
            if self.bond_object.number_of_bonds == 1:
                self.bond_object.bonds = 'single'
            if self.bond_object.number_of_bonds == 2:
                self.bond_object.bonds = 'double'
            
            self.visual = vis.cylinder(radius = self.bond_object.number_of_bonds * 0.05, pos = self.bond_object.atom1.pos,
                                axis = self.bond_object.atom2.pos - self.bond_object.atom1.pos,
                                color = visual.bond_color[self.bond_object.bonds] )

            visual.all_visual_bonds.append(self)


    def update_visual():

        for specific_atom_visual in visual.all_visual_atoms:
            specific_atom_visual.visual.pos = specific_atom_visual.atom_object.pos

        for specific_bond_visual in visual.all_visual_bonds:
            specific_bond_visual.visual.pos = specific_bond_visual.bond_object.atom1.pos
            specific_bond_visual.visual.axis = (specific_bond_visual.bond_object.atom2.pos -
                                                specific_bond_visual.bond_object.atom1.pos )
                                                                    
            
def main():

    global vals

    my_MD_simulation = simulation(squishy_sphere = True, radius = 15.0, stiffness = 1E3)

    protein_length = 12
    
    protein_one = protein( length = protein_length )

    for residue in range(0, protein_length):
        protein_one.add_amino_acid(residue = 'Alanine')

    my_MD_simulation.molecules.append(protein_one.molecule)

    my_MD_simulation.hydrate(number_of_water = 150)

    my_MD_simulation.visualize()
    
#############################################################################################################################
    
    n_atoms = 0
    
    for mol in molecule.all_molecules:
        n_atoms += len(mol.atoms)

    vals = np.empty(7*n_atoms)
    
    integrator = ode(ders)
    integrator.set_integrator("vode", rtol = 1E-6)

    pindex = 0
    windex = 0
    for mol in molecule.all_molecules:
        for at in mol.atoms:
            at.move_data_to_buffers( vals[pindex:pindex+3],
                                     vals[3*n_atoms+pindex : 3*n_atoms+pindex+3],
                                     vals[6*n_atoms+windex : 6*n_atoms+windex+1] )
            pindex += 3
            windex += 1
            
    t = 0 
    tf = 500.0
    t_int = 10.
    t_checker = 1
    dt = 1E-2

    integrator.set_initial_value(vals)

    done = False
    
    while done == False:

        print(t)
        
        integrator.integrate(t + dt)
        vals[:] = integrator.y
        
        my_MD_simulation.times.append(t)
        my_MD_simulation.update_energies()

        visual.update_visual()

        # t = protein_one.amino_acids[0].save_residue_positions(t, tf)

        if t >= t_int:

            my_MD_simulation.print_final_positions(name = str('{:.0f}'.format(t)))
            
            done_moving = True
            vel_min = 1E-2

            for mol in my_MD_simulation.molecules:
                for at in mol.atoms:
                    if mag(at.vel) > vel_min:
                        done_moving = False

            if done_moving == True:
                done = True

            else:
                t_int += 10.
                
        t += dt

        vis.rate(30) 

    my_MD_simulation.print_final_positions(name = str('This_is_a_test_image'))
    # my_MD_simulation.graph_energy_conservation( save_image = True)
    
if __name__ == '__main__':
    main()
