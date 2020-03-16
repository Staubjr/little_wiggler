import sys
import os
import numpy as np
import math

global file_config_path

file_config_path = str('/home/staubj/Capstone_Files/Txt_CHARM_Files/CHARRM_FILES/')

def unfuck_string(string):
    string.replace

def read_CHARRM_file():

    os.chdir(file_config_path)

    file = open('charrm_readable_parameter_file.txt', 'r')
    lines = file.readlines()

    # line_index = 3130
    # for index in range(0, len(lines[line_index])):
    #     print(index, lines[line_index][index])

    # sys.exit(20)

    curated_lines = []
    bond_lines = []
    bond_angle_lines = []
    dihedral_lines = []
    lennard_jones_lines = []
    electrostatic_lines = []
    
    for specific_line in lines:

        length = len(specific_line)
        blank_line = True
        delete_line = False

        if specific_line[0] == '!' or specific_line[0] == '*':
            delete_line = True

        if delete_line == False:
            
            if specific_line[0:12] == '           !':
                delete_line = True

            if specific_line[0:17] == '                !':
                delete_line = True

            if specific_line[0:3] == '\t\t!':
                delete_line = True

            if specific_line[0:20] == str('                   !'):
                delete_line = True

            if specific_line[0:37] == '                                    !':
                delete_line = True

            if specific_line[0:2] == '\t!':
                delete_line = True

            if specific_line[0:9] == '        !':
                delete_line = True
                    
        if delete_line == False:
            
            for index in range(0, length):
                if specific_line[index] != ' ':
                    blank_line = False

        if blank_line == True:
            if delete_line == False:
                delete_line == True

        if specific_line == str('\n'):
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

        if delete_line == False:
            curated_lines.append(fixed_line)

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

    write_file = open('Formatted_CAHRMM_File.txt', 'w')
    
    
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

    for dihedral in dihedral_values:
        for value in dihedral:
            write_file.write(str(value))
            write_file.write('\t')
        write_file.write('\n')

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
