from enum import Enum

bohr2ang           = 0.5291772105638411
HARTREE_TO_EV      = 27.211386024367243                                  # equal to ase.units.Hartree
EV_TO_JOULE        = 1.6021766208e-19                                    # equal to ase.units._e (electron charge)
JOULE_TO_KCAL      = 1 / 4184.                                           # exact
HARTREE_TO_JOULE   = HARTREE_TO_EV * EV_TO_JOULE
AVOGADROS_NUMBER   = 6.022140857e+23                                     # equal to ase.units._Nav
HARTREE_TO_KCALMOL = HARTREE_TO_JOULE * JOULE_TO_KCAL * AVOGADROS_NUMBER # equal to ase value of 627.5094738898777
KCAL_MOL_to_AU     = 1 / HARTREE_TO_KCALMOL
KCAL_MOL_A_to_AU   = KCAL_MOL_to_AU * bohr2ang
KCAL_MOL_A2_to_AU  = KCAL_MOL_A_to_AU * bohr2ang
A_to_AU            = 1/bohr2ang


class Units(Enum):
    A = 0
    AU = 1
    KCAL = 2