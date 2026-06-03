"""Module for storing numeric constants"""
import math

# ENERGY CONVERSIONS
# convert au to electron volts
au2ev    = 27.2113845
ev2au    = 1. / au2ev
# convert au to cm-1
au2cm    = 219474.63068
cm2au    = 1. / au2cm

# DISTANCE CONVERSIONS
# bohr to angstrom
bohr2ang = 0.529177249
# angstrom to bohr
ang2bohr  = 1. / bohr2ang

#CHARGE CONVERSIONS
au2debye  = 2.541580

# MASS conversions
amu2au = 1836.152673

#MISC CONSTANTS
fine_str = 1. / 137.0359895
c_spd    = 1. / fine_str
# conversion factor for degree K to au
kB       = 3.16681520371153e-6

#TRIG CONSTANTS
rad2deg  = 180./math.pi
deg2rad  = math.pi/180.

# TIME CONSTANTS
fs2au  = 41.334
au2fs  = 1./fs2au
