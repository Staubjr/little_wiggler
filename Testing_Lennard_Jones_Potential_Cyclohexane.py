#/usr/bin/env python3
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
from scipy.integrate import ode

""" Note about the units used here:
    
    To match the par and top files from CHARM files
    these are the following unit scales we are working in:
        
    Distance: Angstroms
    Mass: AMU
    Energy: eV
    Time: sqrt( AMU * Angstroms^2 / eV ) ~ 1E-14 sec
    Angles: Radians

    This results in this unit of energy:

    E' = (1E-2)/n * Joules
    E' = 1.602E-21 * eV """

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
    global vals

    for mol in molecule.all_molecules:
        mol.clear_all_counted()
    
    vals[:] = y
    dvals = np.empty((len(vals)))
    dvals[0:len(dvals)//2] = vals[len(vals)//2:len(vals)]
    
    """ Update all the forces """
    for mol in molecule.all_molecules:
        mol.get_all_forces()
    
    """ Get the accelerations in dvals """
    index = len(dvals)//2
    for mol in molecule.all_molecules:
        for at in mol.atoms:
            dvals[index:index+3] = at.force/at.mass - y[index:index+3] * molecule.dampening_coef/at.mass
            index += 3

    # dh0 = mol.dihedrals[3]
    # a1 = dh0.first_atom
    # a2 = dh0.second_atom
    # posvec = a1.pos - a2.pos
    # dex = 0
    # for mol in molecule.all_molecules:
    #     for at in mol.atoms:
    #         if at == a2:
    #             break
    #         dex += 1
    # dex = 3*dex + len(dvals)//2
    # acc = np.array( [dvals[dex], dvals[dex+1], dvals[dex+2]] )
    # Print("dh0 1-2 length: {:.4g}, acc: {:.4g}, length dot acc: {:.4g}, f: {:.4g}, length dot f: {:.4g}, norm(f dot acc): {:.4g}"
    #       .format( mag(posvec), mag(acc), (posvec*acc).sum() , 
    #                mag(a2.force), (posvec*a2.force).sum(), (a2.force*acc).sum()/mag(a2.force)/mag(acc) ) )
        
    return dvals
    

def runge_kutta(t, vals, dt):
    """ Fourth-order Runge Kutta algorithm to update Vals """
    
    C1 = ders(t,vals) * dt
    C2 = ders(t+dt/2,vals+C1/2) * dt
    C3 = ders(t+dt/2,vals+C2/2) * dt
    C4 = ders(t+dt,vals+C3) * dt
    vals += (1/6)*(C1 + 2*C2 + 2*C3 + C4)

    return vals

class atom:

    all_atoms = []
    
    masses = { 'hydrogen' : 1,
               'oxygen'   : 16,
               'carbon'   : 12,
               'nitrogen' : 10 }
    
    sigs = { "hydrogen-hydrogen" : 7.4,
            "oxygen-oxygen"      : 1.503,
            "hydrogen-oxygen"    : 9.6 }
    
    epsilons = { "hydrogen-hydrogen" : 4.519,
                "hydrogen-oxygen"    : 3.793,
                "oxygen-oxygen"      : 1.503 }
    
    potential_bonds = { 'hydrogen' : 1,
                        'oxygen'   : 2,
                        'carbon'   : 4,
                        'nitrogen' : 3 }

    charges = { 'ON' : -0.438,
                'NN' : 0.22   }
    
    def __init__(self, element_type, molecule = None, potential = 0, atom_number = 0, element = 'hydrogen',
                       x0 = 0., y0 = 0., z0 = 0., vx0 = 0., vy0 = 0., vz0 = 0. ):
        
        self._pos = np.array([x0, y0, z0])
        self._vel = np.array([vx0, vy0, vz0])
        self.element = element
        self.mass = atom.masses[self.element]
        self.atom_number = atom_number
        self.molecule = molecule
        self.element_type = element_type
        # self.charge = atom.charges[self.element_type]
        self.potential = potential
        self.force = np.zeros(3)
        self.bonds = []

        if len(atom.all_atoms) > 0:
            for specific_atom in atom.all_atoms:
                lennard_jones(self, specific_atom)

        visual(self)
        atom.all_atoms.append(self)
        
    def move_data_to_buffers(self, posbuffer, velbuffer):
        pos = self._pos.copy()
        vel = self._vel.copy()
        self._pos = np.frombuffer(posbuffer)
        self._vel = np.frombuffer(velbuffer)
        self._pos[:] = pos
        self._vel[:] = vel
               
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
        
class molecule:

    dampening_coef = 0.9
    
    all_molecules = []
    
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
        
    def bond_atoms(self, atom1, atom2):
        if not atom1 in self.atoms:
            raise Exception("I ain't got {}, atom number {}. Try molecule {}".format(atom2.element, atom2.atom_number, atom2.molecule))
        if not atom2 in self.atoms:
            raise Exception("I ain't got {}, atom number {}. Try molecule {}".format(atom2.element, atom2.atom_number, atom2.molecule))

        new_bond = bond(atom1, atom2)

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
        
    def get_potential(self):
        for bonded_atoms in self.bonds:
            dU = bonded_atoms.bond_potential()
            self.bonds.atom1.potential += dU
            self.bonds.atom1.potential += -1 * dU
        for angled_bonds in self.bond_angles:
            dV_dtheta = self.bond_angle.bond_angle_potential()
            self.bond_angles.atom1 = dV_dtheta
            
    def clear_all_counted(self):
        for bon in self.bonds:
            bon.counted = False
        for bangle in self.bond_angles:
            bangle.counted = False
        for dangle in self.dihedrals:
            dangle.counted = False
        
            
    def get_all_forces(self):
        """ Takes a molecule and calculates all bond and bond angle
        forces for all of the atoms in the molecule

        11/25/2019: Jake, you can definitely clean this code up.
        You wrote paramters in for each of the classes (bonds, bond
        angle, and dihedrlas) for the order of the atoms. Def not computationally
        efficient to be doing this as you are now. FIX THIS! """
        
        for at in self.atoms:
            at.force[:] = 0.
            
        for specific_bond in self.bonds:
            if specific_bond.counted == False:

                specific_bond.bond_force()

        for specific_bond_angle in self.bond_angles:
            if specific_bond_angle.counted == False:

                specific_bond_angle.bond_angle_force()

        for specific_dihedral in self.dihedrals:
            if specific_dihedral.counted == False:
                
                specific_dihedral.dihedral_force()

        for specific_atom in self.atoms:
            for specific_lennard_jones_pair in lennard_jones.all_lennard_jones_pairs:

                if specific_atom == specific_lennard_jones_pair.atom1:
                    forces = lennard_jones.lennard_jones_force(specific_lennard_jones_pair)
                    specific_lennard_jones_pair.atom1.force += forces[0]
                    
                if specific_atom == specific_lennard_jones_pair.atom2:
                    forces = lennard_jones.lennard_jones_force(specific_lennard_jones_pair)
                    specific_lennard_jones_pair.atom2.force += forces[1]                

class bond:
    
    k_bond = { "HT-OT" : 450 * (2.611E22/6.022E23),
               "CC33A-HCA3A" : 322.00 * (2.611E22/6.022E23),
               "CC33A-CC33A" : 222.50 * (2.611E22/6.022E23),
               "CC32A-CC33A" : 222.50 * (2.611E22/6.022E23),
               "CC32A-HCA2A" : 309.00 * (2.611E22/6.022E23),
               "CC32A-CC32A" : 222.50 * (2.611E22/6.022E23) } #(eV/A^2)
              
    E0_bond = { "HT-OT" : 0.9572,
                "CC33A-HCA3A" : 1.111,
                "CC33A-CC33A" : 1.530,
                "CC32A-CC33A" : 1.528,
                "CC32A-HCA2A" : 1.111,
                "CC32A-CC32A" : 1.530 } #angstroms
     
    def __init__(self, atom1, atom2, counted = False, multiplicity = 1):
        self.atom1 = atom1
        self.atom2 = atom2
        self.counted = counted
        self.multiplicity = multiplicity
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
            
        visual(self)
    
    def bond_potential(self):
        """ Calculates and returns bond potential """
            
        distance = self.atom2.pos - self.atom1.pos
        potential =  (1/2) * self.k_bond * (distance - self.E0_bond)**2 
        
        return potential
    
    def bond_force(self):
        """ Calculates the spring bond potential and returns dU """
        
        r_vec = (self.atom2.pos - self.atom1.pos)
        r_hat = r_vec/mag(r_vec)
        dU = 2 * self.k_bond * (mag(r_vec) - self.E0_bond)
        Force_1 = dU * r_hat
        Force_2 = -1 * Force_1

        self.atom1.force += Force_1
        self.atom2.force += Force_2

        self.atom1.counted = True
        self.atom2.counted = True
    
class bond_angle:
    
    k_bond_angle = { "HT-OT-HT"          : 55.000 * (2.611E22/6.022E23),
                     "HCA3A-CC33A-HCA3A" : 35.50  * (2.611E22/6.022E23),
                     "HCA3A-CC33A-CC33A" : 37.500 * (2.611E22/6.022E23),
                     "HCA2A-CC32A-CC33A" : 34.600 * (2.611E22/6.022E23),
                     "CC33A-CC32A-CC33A" : 53.350 * (2.611E22/6.022E23),
                     "HCA3A-CC33A-CC32A" : 34.600 * (2.611E22/6.022E23),
                     "HCA2A-CC32A-HCA2A" : 35.50  * (2.611E22/6.022E23),
                     "HCA2A-CC32A-CC32A" : 26.500 * (2.611E22/6.022E23),
                     "CC32A-CC32A-CC32A" : 58.350 * (2.611E22/6.022E23) }  #(eV/rad^2)
    
    bond_angle = { "HT-OT-HT"          : (104.5200/360) * math.pi*2,
                   "HCA3A-CC33A-HCA3A" : (108.40/360)   * math.pi*2,
                   "HCA3A-CC33A-CC33A" : (110.10/360)   * math.pi*2,
                   "HCA2A-CC32A-CC33A" : (110.10/360)   * math.pi*2,
                   "CC33A-CC32A-CC33A" : (114.00/360)   * math.pi*2,
                   "HCA3A-CC33A-CC32A" : (110.10/360)   * math.pi*2,
                   "HCA2A-CC32A-HCA2A" : (109.00/360)   * math.pi*2,
                   "HCA2A-CC32A-CC32A" : (110.10/360)   * math.pi*2,
                   "CC32A-CC32A-CC32A" : (113.60/360)   * math.pi*2 } #radians
        
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
            print('here')
            sys.stderr.write('Unable to find a bond angle spring coefficient for atoms : {}-{}-{}, atom numbers : {}-{}-{}, and element types : {}-{}-{}'.format(self.start_atom.element, self.middle_atom.element, self.end_atom.element,
                                    self.start_atom.atom_number, self.middle_atom.atom_number, self.end_atom.atom_number,
                                    self.start_atom.element_type, self.middle_atom.element_type, self.end_atom.element_type))

                

        if self.typedex in bond_angle.bond_angle:
            self.bond_angle0 = bond_angle.bond_angle[self.typedex]
        elif self.reverse_typedex in bond_angle.bond_angle:
            if self.typedex != self.reverse_typedex:
                self.bond_angle0 = bond_angle.bond_angle[self.reverse_typedex]
        else:
            sys.stderr.write('Unable to find a bond angle equilibrium angle for atoms : {}-{}-{}, atom numbers : {}-{}-{}, and element types : {}-{}-{}'.format(self.start_atom.element, self.middle_atom.element, self.end_atom.element,
                                   self.start_atom.atom_number, self.middle_atom.atom_number, self.end_atom.atom_number,
                                   self.start_atom.element_type, self.middle_atom.element_type, self.end_atom.element_type))

         
                                                
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
        
        theta = math.asin( mag( cross_product(bond1_vec, bond2_vec) ) / ( mag(bond1_vec) * mag(bond2_vec)) )
        
        potential = ( self.k_bond_angle * (theta - self.bond_angle0)**2 )
        
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
                raise err
            theta = math.asin( tmpshit )
            
        try:
            theta = math.acos( (bond1_vec * bond2_vec).sum() / ( mag(bond1_vec) * mag(bond2_vec) ) )
            
        except ValueError as err:
            raise Exception ('HOLY FUCK {} {} \n'.format(bond1_vec, bond2_vec))
        """ Calculate the potential with respect to the theta """
        dV_dtheta =  self.k_bond_angle * (theta - self.bond_angle0)
        
        """ Calculate the torques based on -dU/dTheta """
        Torque_1_val = -1*dV_dtheta
        Torque_2_val = -1*Torque_1_val
        
        """ Get the forces on the atoms and central atom """
        mag1 = mag(bond1_vec)
        mag2 = mag(bond2_vec)
        
        a_hat = cross_product(bond1_vec, bond2_vec) / mag(cross_product(bond1_vec, bond2_vec))
        F1_hat = cross_product(bond1_vec, a_hat) / mag(cross_product(bond1_vec, a_hat))
        F1_val = Torque_1_val / mag1
        Force_1 = F1_hat*F1_val
        
        F2_hat = cross_product(bond2_vec, a_hat) / mag(cross_product(bond2_vec, a_hat))
        F2_val = Torque_2_val / mag2
        Force_2 = F2_hat*F2_val
        
        F_center_atom = -1 * (Force_1 + Force_2)

        self.start_atom.force += Force_1
        self.middle_atom.force += F_center_atom
        self.end_atom.force += Force_2

        self.start_atom.counted = True
        self.middle_atom.counted = True
        self.end_atom.counted = True

class dihedral:

    k_chi = {'HCA3A-CC33A-CC33A-HCA3A' : 0.15250 * (2.611E22/6.022E23),
             'HCA3A-CC33A-CC32A-HCA2A' : 0.1600  * (2.611E22/6.022E23),
             'HCA3A-CC33A-CC32A-CC33A' : 0.1600  * (2.611E22/6.022E23),
             'HCA2A-CC32A-CC32A-HCA2A' : 0.19000 * (2.611E22/6.022E23),
             'HCA2A-CC32A-CC32A-CC32A' : 0.19000 * (2.611E22/6.022E23),
             'CC32A-CC32A-CC32A-CC32A' : 0.11251 * (2.611E22/6.022E23) }

    delta = {'HCA3A-CC33A-CC33A-HCA3A' : 0.0,
             'HCA3A-CC33A-CC32A-HCA2A' : 0.0,
             'HCA3A-CC33A-CC32A-CC33A' : 0.0,
             'HCA2A-CC32A-CC32A-HCA2A' : 0.0,
             'HCA2A-CC32A-CC32A-CC32A' : 0.0,
             'CC32A-CC32A-CC32A-CC32A' : 0.0 }

    n = {'HCA3A-CC33A-CC33A-HCA3A' : 3,
         'HCA3A-CC33A-CC32A-HCA2A' : 3,
         'HCA3A-CC33A-CC32A-CC33A' : 3,
         'HCA2A-CC32A-CC32A-HCA2A' : 3,
         'HCA2A-CC32A-CC32A-CC32A' : 3,
         'CC32A-CC32A-CC32A-CC32A' : 5 }
    
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
        
        if self.typedex in dihedral.n:
            self.n = dihedral.n[self.typedex]
        elif self.reverse_typedex in dihedral.n:
            if self.typedex != self.reverse_typedex:
                self.n = dihedral.n[self.reverse_typedex]
        else:
            sys.stderr.write('Could not find a n for the dihedral {} or {}'.format(self.typedex, self.reverse_typedex))
            sys.exit(20)

        if self.typedex in dihedral.delta:
            self.delta = dihedral.delta[self.typedex]
        elif self.reverse_typedex in self.delta:
            if self.typedex != self.reverse_typedex:
                self.delta = dihedral.delta[self.reverse_typedex]
        else:
            sys.stderr.write('Could not find a delta for the dihedral {} or {}'.format(self.typedex, self.reverse_typedex))
            sys.exit(20)

        if self.typedex in dihedral.k_chi:
            self.k_chi = dihedral.k_chi[self.typedex]
        elif self.reverse_typedex in dihedral.k_chi:
            if self.typedex != self.reverse_typedex:
                self.k_chi = dihedral.k_chi[self.reverse_typedex]
        else:
            sys.stderr.write('COuld not find a k_chi for the dihedral {} or {}'.format(self.typedex, self.reverse_typedex))
            sys.exit(20)
    

    def dihedral_potential(self):
        ''' Calculates the potential from the dihedrals '''

        a_vec = self.first_atom.pos - self.second_atom.pos
        b_vec = self.second_atom.pos - self.third_atom.pos
        c_vec = self.third_atom.pos - self.fourth_atom.pos

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

        chi = math.acos( (a_perp[0] * c_perp[0] +
                          a_perp[1] * c_perp[1] +
                          a_perp[2] * c_perp[2])

                         /

                         ( mag(a_perp) * mag(c_perp) ) )

        
        dihedral_potential =  self.k_chi * ( 1 + math.cos( self.n * chi - self.delta ) )

    def dihedral_force(self):
        ''' Calculates the forces on all atoms in the dihedrals.

            Returns a list of the forces in this order:
                      
                    1.) Start Atom
                    2.) Central Bond Atom 1
                    3.) Central Bond Atom 2
                    4.) End Atom                                 '''

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

    epsilon_i = { 'CC33A' : -0.0780 * (2.611E22/6.022E23) ,
                  'HCA3A' : -0.0240 * (2.611E22/6.022E23) ,
                  'HT'    : -0.046  * (2.611E22/6.022E23) ,
                  'HCA2A' : -0.0350 * (2.611E22/6.022E23) ,
                  'CC32A' : -0.0560 * (2.611E22/6.022E23) ,
                  'OT'    : -0.1521 * (2.611E22/6.022E23) }

    r_min_i = { 'CC33A' : 2.0400 ,
                'HCA3A' : 1.3400 ,
                'HT'    : 0.2245 ,
                'HCA2A' : 1.3400 ,
                'CC32A' : 2.0100 ,
                'OT'    : 1.7682 }

    def __init__(self, atom_1, atom_2):
        self.atom1 = atom_1
        self.atom2 = atom_2
        
        self.Epsilon = math.sqrt( lennard_jones.epsilon_i[atom_1.element_type] *
                                  lennard_jones.epsilon_i[atom_2.element_type] )
        
        self.R_min = ( lennard_jones.r_min_i[atom_1.element_type]/2 +
                       lennard_jones.r_min_i[atom_2.element_type]/2 )
        
        lennard_jones.all_lennard_jones_pairs.append(self)

    def lennard_jones_potential(self):
        r_vec = self.atom1.pos - self.atom2.pos
        mag_r = mag(r_vec)
        potential = self.Epsilon * ( (self.R_min/mag_r)**12 - 2 * (self.R_min/mag_r)**6)
        return potential

    def lennard_jones_force(self):
        r_vec = self.atom1.pos - self.atom2.pos
        mag_r = mag(r_vec)
        dU_dr = 12 * self.Epsilon * ( ( (self.R_min) ** 12 / (mag_r )** 13 ) -
                                       ( (self.R_min) ** 6  / (mag_r) ** 7 )  )
        Force_1 = dU_dr * r_vec
        Force_2 = - Force_1

        Force_list = [Force_1, Force_2]

        return Force_list


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

    
    def __init__(self, object):

        if object.__class__.__name__ == 'atom':
            self.atom_object = object
            self.visual =  vis.sphere(pos = object.pos, radius = visual.atom_radii[object.element],
                              color = visual.atom_colors[object.element] )
            visual.all_visual_atoms.append(self)

        if object.__class__.__name__ == 'bond':
            self.bond_object = object
            self.visual = vis.cylinder(radius = 0.05, pos = object.atom1.pos,
                                axis = object.atom2.pos - object.atom1.pos,
                                color = (0.4, 0.7, 0.9) )

            visual.all_visual_bonds.append(self)

    def update_visual():

        for specific_atom_visual in visual.all_visual_atoms:
            specific_atom_visual.visual.pos = specific_atom_visual.atom_object.pos

        for specific_bond_visual in visual.all_visual_bonds:
            specific_bond_visual.visual.pos = specific_bond_visual.bond_object.atom1.pos
            specific_bond_visual.visual.axis = (specific_bond_visual.bond_object.atom2.pos -
                                                specific_bond_visual.bond_object.atom1.pos )
                                                                    
            
        
    
def main():

    Time_Unit = ( (1.66054E-27 * 1E-20) / (1.60218E-19) )**(1/2)
    Epsilon = ( ( 1.60218E-19)**2 * (Time_Unit)**2 ) / ( (1E-10)**3 * (1.66054E-27) )
    print('Time Units (seconds): {} and Epsilon: {}'.format(Time_Unit, Epsilon))

    
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

    
    global vals
    
    c_1 = atom(element = 'carbon', element_type = 'CC32A', atom_number = 1, y0 =  l0/2)
    c_2 = atom(element = 'carbon', element_type = 'CC32A', atom_number = 2, y0 =  l0 * s20, x0 =  l0 * c20)
    c_3 = atom(element = 'carbon', element_type = 'CC32A', atom_number = 3, y0 = -l0 * s20, x0 =  l0 * c20)
    c_4 = atom(element = 'carbon', element_type = 'CC32A', atom_number = 4, y0 = -l0/2)
    c_5 = atom(element = 'carbon', element_type = 'CC32A', atom_number = 5, y0 = -l0 * s20, x0 = -l0 * c20)
    c_6 = atom(element = 'carbon', element_type = 'CC32A', atom_number = 6, y0 =  l0 * s20, x0 = -l0 * c20)
    
    h_1 = atom(atom_number = 7, element = 'hydrogen', element_type = 'HCA2A', x0 = c_1.pos[0] - h_bond - 0.01, y0 = c_1.pos[1] + h_bond, z0 = .5)
    h_2 = atom(atom_number = 8, element = 'hydrogen', element_type = 'HCA2A', x0 = c_1.pos[0] + h_bond, y0 = c_1.pos[1] + h_bond, z0 = -0.5)
    
    h_3 = atom(atom_number = 9, element = 'hydrogen', element_type = 'HCA2A', x0 = c_2.pos[0] + h_bond, y0 = c_2.pos[1] + h_bond)
    h_4 = atom(atom_number = 10, element = 'hydrogen', element_type = 'HCA2A', x0 = c_2.pos[0] + l1, y0 = c_2.pos[1])
    
    h_5 = atom(atom_number = 11, element = 'hydrogen', element_type = 'HCA2A', x0 = c_3.pos[0] + l1, y0 = c_3.pos[1])
    h_6 = atom(atom_number = 12, element = 'hydrogen', element_type = 'HCA2A', x0 = c_3.pos[0] + h_bond, y0 = c_3.pos[1] - h_bond)
    
    h_7 = atom(atom_number = 13, element = 'hydrogen', element_type = 'HCA2A', x0 = c_4.pos[0] + h_bond, y0 = c_4.pos[1] - h_bond)
    h_8 = atom(atom_number = 14, element = 'hydrogen', element_type = 'HCA2A', x0 = c_4.pos[0] - h_bond, y0 = c_4.pos[1] - h_bond)

    h_9 = atom(atom_number = 15, element = 'hydrogen', element_type = 'HCA2A', x0 = c_5.pos[0] - l1, y0 = c_5.pos[1])
    h_10 = atom(atom_number = 16, element = 'hydrogen', element_type = 'HCA2A', x0 = c_5.pos[0] - h_bond, y0 = c_5.pos[1] - h_bond)

    h_11 = atom(atom_number = 17, element = 'hydrogen', element_type = 'HCA2A', x0 = c_6.pos[0] - h_bond, y0 = c_6.pos[1] + h_bond)
    h_12 = atom(atom_number = 18, element = 'hydrogen', element_type = 'HCA2A', x0 = c_6.pos[0] - l1, y0 = c_6.pos[1])


    cyclohexane_1 = molecule()
    
    cyclohexane_1.add_atom(h_1)
    cyclohexane_1.add_atom(h_2) 
    cyclohexane_1.add_atom(h_3)
    cyclohexane_1.add_atom(h_4)
    cyclohexane_1.add_atom(h_5)
    cyclohexane_1.add_atom(h_6)
    cyclohexane_1.add_atom(h_7)
    cyclohexane_1.add_atom(h_8)
    cyclohexane_1.add_atom(h_9)
    cyclohexane_1.add_atom(h_10)
    cyclohexane_1.add_atom(h_11)
    cyclohexane_1.add_atom(h_12)
    
    cyclohexane_1.add_atom(c_1)
    cyclohexane_1.add_atom(c_2)
    cyclohexane_1.add_atom(c_3)
    cyclohexane_1.add_atom(c_4)
    cyclohexane_1.add_atom(c_5)
    cyclohexane_1.add_atom(c_6)
    

    cyclohexane_1.bond_atoms(h_1 , c_1)
    cyclohexane_1.bond_atoms(h_2 , c_1)
    cyclohexane_1.bond_atoms(h_3 , c_2)    
    cyclohexane_1.bond_atoms(h_4 , c_2)
    cyclohexane_1.bond_atoms(h_5 , c_3)
    cyclohexane_1.bond_atoms(h_6 , c_3)
    cyclohexane_1.bond_atoms(h_7 , c_4)
    cyclohexane_1.bond_atoms(h_8 , c_4)
    cyclohexane_1.bond_atoms(h_9 , c_5)
    cyclohexane_1.bond_atoms(h_10, c_5)
    cyclohexane_1.bond_atoms(h_11, c_6)
    cyclohexane_1.bond_atoms(h_12, c_6)
    
    cyclohexane_1.bond_atoms(c_1, c_2)
    cyclohexane_1.bond_atoms(c_2, c_3)
    cyclohexane_1.bond_atoms(c_3, c_4)
    cyclohexane_1.bond_atoms(c_4, c_5)
    cyclohexane_1.bond_atoms(c_5, c_6)
    cyclohexane_1.bond_atoms(c_6, c_1)

    for dangle in cyclohexane_1.dihedrals:
        print(dangle.first_atom.atom_number, '-',
              dangle.second_atom.atom_number, '-',
              dangle.third_atom.atom_number, '-',
              dangle.fourth_atom.atom_number)

    print('\n', len(cyclohexane_1.dihedrals))

    # Something is v wrong with my dihedral counter.
    # It appears as though our worst fears have been confirmed ...
    # the cyclic molecule is causing me to wayyyyy over count the
    # number of dihedrals present.  I am pretty sure that
    # cyclohexane should not have 121 dihedrals in it.  Jake,
    # implement a method to fix this overcounting and you should
    # also fix that ValueError that you are getting in the bond
    # force function with the arcsin.  When theta is 90 you are
    # getting this error.



############################################################################################
    
    index = 0
    n_atoms = 0
    
    for mol in molecule.all_molecules:
        n_atoms += len(mol.atoms)
    
    vals = np.empty(6*n_atoms)
    
    integrator = ode(ders)
    integrator.set_integrator("vode")
    
    
    for mol in molecule.all_molecules:
        for at in mol.atoms:
            at.move_data_to_buffers( vals[index:index+3],
                                     vals[3*n_atoms+index : 3*n_atoms+index+3] )
            index += 3    
    t = 0 
    tf = 500    
    dt = 1E-1

    t_stall = 5
    
    integrator.set_initial_value(vals)

    have_already_printed = False

    
    while t < tf:
        
        integrator.integrate(t + dt)
        vals[:] = integrator.y
        
        visual.update_visual()

        while t <= t_stall:
            t += dt

        vels = [ mag(atom.vel) for atom in cyclohexane_1.atoms ]
        maxvel = max(vels)
        
        if not have_already_printed and maxvel <= 1E-5:
            print(maxvel)
            for bangle in cyclohexane_1.bond_angles:
                r1 = bangle.end_atom.pos - bangle.middle_atom.pos
                r2 = bangle.start_atom.pos - bangle.middle_atom.pos
                the_angle_bruh = math.acos( (r1 * r2).sum() / mag(r1) / mag(r2) ) * 180./math.pi
                print("{:20s} : {:.2f}°".format(bangle.typedex , the_angle_bruh))
            have_already_printed = True
                
                
        
        t += dt
        
        vis.rate(30)
                         
if __name__ == '__main__': 
    main()
