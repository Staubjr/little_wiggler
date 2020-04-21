import sys
import os
import numpy as np
import math

global file_config_path

file_config_path = str('CHARMM_FILES/')

def read_CHARRM_file():

    os.chdir(file_config_path)

    file = open('charmm_readable_parameter_file.txt', 'r')
    second_file = open('second_charmm_readable_parameter_file.txt', 'r')
    lines = file.readlines()
    second_lines = second_file.readlines()

    curated_lines = []
    curated_lines_2 = []
    bond_lines = []
    bond_angle_lines = []
    dihedral_lines = []
    lennard_jones_lines = []
    electrostatic_lines = []

    i = 0
    
    for a_line in lines:

        delete_line = False
        
        specific_line = a_line.strip()

        if len(specific_line) == 0:
            delete_line = True

        if delete_line == False:
            if specific_line[0] == '!' or specific_line[0] == '*':
                delete_line = True
                      
        if delete_line == False:

            exclamation_point = 0

            for index in range(0, len(specific_line)):
                if specific_line[index] == '!':
                    exclamation_point = index

            value = len(specific_line) - exclamation_point

            if exclamation_point != 0:
                fixed_line = specific_line[:-value]

            else:
                fixed_line = specific_line.strip('\n')

            curated_lines.append(fixed_line)

    for a_line in second_lines:

        delete_line = False
        
        specific_line = a_line.strip()

        if len(specific_line) == 0:
            delete_line = True

        if delete_line == False:
            if specific_line[0] == '!' or specific_line[0] == '*':
                delete_line = True
                      
        if delete_line == False:

            exclamation_point = 0

            for index in range(0, len(specific_line)):
                if specific_line[index] == '!':
                    exclamation_point = index

            value = len(specific_line) - exclamation_point

            if exclamation_point != 0:
                fixed_line = specific_line[:-value]

            else:
                fixed_line = specific_line.strip('\n')

            curated_lines_2.append(fixed_line)
            
    bond_start_line = 0
    bond_end_line = 0
    bond_angle_start_line = 0
    bond_angle_end_line = 0
    dihedral_start_line = 0
    dihedral_end_line = 0
    lennard_jones_start_line = 0
    lennard_jones_end_line = 0
    electrostatic_start_line = 0
    electrostatic_end_line = 0

    bond_values = []
    bond_angle_values = []
    dihedral_values = []
    lennard_jones_values = []

    write_file = open('Formatted_CHARMM_File.txt', 'w')
    
    
    for specific_line in curated_lines:
        if specific_line[0:5] == str('BONDS'):
            bond_start_line = curated_lines.index(specific_line) + 1

        if specific_line[0:6] == str('ANGLES'):
            bond_angle_start_line = curated_lines.index(specific_line) + 1
            bond_end_line = bond_angle_start_line - 1

        if specific_line[0:9] == str('DIHEDRALS'):
            dihedral_start_line = curated_lines.index(specific_line) + 1
            bond_angle_end_line = dihedral_start_line - 1

        if specific_line[0:8] == str('IMPROPER'):
            dihedral_end_line = curated_lines.index(specific_line)

        if specific_line[0:9] == str('NONBONDED'):
            lennard_jones_start_line = curated_lines.index(specific_line) + 1

        if specific_line[0:5] == str('HBOND'):
            lennard_jones_end_line = curated_lines.index(specific_line)-1

    for specific_line in curated_lines_2:
        if specific_line[0:5] == str('BONDS'):
            bond_start_line_2 = curated_lines_2.index(specific_line) + 1

        if specific_line[0:6] == str('ANGLES'):
            bond_angle_start_line_2 = curated_lines_2.index(specific_line) + 1
            bond_end_line_2 = bond_angle_start_line_2 - 1

        if specific_line[0:9] == str('DIHEDRALS'):
            dihedral_start_line_2 = curated_lines_2.index(specific_line) + 1
            bond_angle_end_line_2 = dihedral_start_line_2 - 1

        if specific_line[0:9] == str('NONBONDED'):
            dihedral_end_line_2 = curated_lines_2.index(specific_line) - 1
            lennard_jones_start_line_2 = curated_lines_2.index(specific_line) + 1

        if specific_line[0:5] == str('END'):
            lennard_jones_end_line_2 = curated_lines_2.index(specific_line)-1

    # print(bond_start_line_2,
    #       bond_end_line_2,
    #       bond_angle_start_line_2,
    #       bond_angle_end_line_2,
    #       dihedral_start_line_2,
    #       dihedral_end_line_2,
    #       lennard_jones_start_line_2,
    #       lennard_jones_end_line_2)

            
    for line in curated_lines[bond_start_line: bond_end_line]:
        values = line.split(' ')
        fixed_values = []
        
        for value in values:
            if value != str(''):
                value = value.replace(" ", "")
                fixed_values.append(value)

        bond_values.append(fixed_values)

    for line in curated_lines[bond_angle_start_line: bond_angle_end_line]:
        values = line.split(' ')
        fixed_values = []

        for value in values:
            if value != str(''):
                value = value.replace(" ", "")
                fixed_values.append(value)

        bond_angle_values.append(fixed_values[0:5])
                
    for line in curated_lines[dihedral_start_line: dihedral_end_line]:
        values = line.split(' ')
        fixed_values = []

        for value in values:
            if value != str(''):
                value = value.replace(" ", "")
                fixed_values.append(value)

        dihedral_values.append(fixed_values[0:7])

    del curated_lines[lennard_jones_start_line]
        
    for line in curated_lines[lennard_jones_start_line: lennard_jones_end_line]:
        values = line.split(' ')
        fixed_values = []
        
        for value in values:
            if value != str(''):
                value = value.replace(" ", "")
                fixed_values.append(value)

        if len(fixed_values) > 4:
            del fixed_values[4:len(fixed_values)]
            
        fixed_values.pop(1)

        lennard_jones_values.append(fixed_values)

        
    for line in curated_lines_2[bond_start_line_2: bond_end_line_2]:
        values = line.split(' ')
        fixed_values = []
        
        for value in values:
            if value != str(''):
                value = value.replace(" ", "")
                fixed_values.append(value)

        bond_values.append(fixed_values)

    for line in curated_lines_2[bond_angle_start_line_2: bond_angle_end_line_2]:
        values = line.split(' ')
        fixed_values = []

        for value in values:
            if value != str(''):
                value = value.replace(" ", "")
                fixed_values.append(value)

        bond_angle_values.append(fixed_values[0:5])
                
    for line in curated_lines_2[dihedral_start_line_2: dihedral_end_line_2]:
        values = line.split(' ')
        fixed_values = []

        for value in values:
            if value != str(''):
                value = value.replace(" ", "")
                fixed_values.append(value)

        dihedral_values.append(fixed_values[0:7])

    del curated_lines_2[lennard_jones_start_line_2]
        
    for line in curated_lines_2[lennard_jones_start_line_2: lennard_jones_end_line_2]:
        values = line.split(' ')
        fixed_values = []
        
        for value in values:
            if value != str(''):
                value = value.replace(" ", "")
                fixed_values.append(value)

        if len(fixed_values) > 4:
            del fixed_values[4:len(fixed_values)]
            
        fixed_values.pop(1)

        lennard_jones_values.append(fixed_values)
        
    write_file.write('#BONDS\n')
    write_file.write('#ATOM#1\tATOM#2\tKb(Kcal/mol/A**2)\tBOND_LENGTH(A)\n')

    for bond in bond_values:
        for value in bond:
            write_file.write(str(value))
            write_file.write('\t')
        write_file.write('\n')

    write_file.write('#\n#\n#BOND_ANGLES\n')
    write_file.write('#ATOM#1\tATOM#2\t#ATOM#3\tKtheta(Kcal/mol/rad**2)\tTHETA0(A)\n')

    for bangle in bond_angle_values:
        for value in bangle:
            write_file.write(str(value))
            write_file.write('\t')
        write_file.write('\n')

    write_file.write('#\n#\n#Dihedrals\n')
    write_file.write('#ATOM#1\tATOM#2\tATOM#3\tATOM#4\tKchi(Kcal/mol)\tn\tdelta(degrees)\n')


    written_dihedrals = []
    
    for dihedral in dihedral_values:
        
        test_dihedral = [dihedral[0], dihedral[1], dihedral[2], dihedral[3]]
        write_dihedral = True

        for finished_dihedral in written_dihedrals:
            if test_dihedral == finished_dihedral:
                write_dihedral = False

        if write_dihedral == True:
            for value in dihedral:
                write_file.write(str(value))
                write_file.write('\t')
            write_file.write('\n')
            
        written_dihedrals.append(test_dihedral)

    write_file.write('#\n#\n#Lennard-Jones\n')
    write_file.write('#ATOM#1\tEpsilon(Kcal/mol)\tRmin/2\n')

    for lenny in lennard_jones_values:
        for value in lenny:
            write_file.write(str(value))
            write_file.write('\t')
        write_file.write('\n')


        
    write_file.close()
        

def main():

    read_CHARRM_file()

if __name__ == '__main__':
    main()
