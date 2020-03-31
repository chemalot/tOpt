from enum import Enum

bohr2ang          = 0.529177210
KCAL_MOL_to_AU    = 1/627.509
KCAL_MOL_A_to_AU  = KCAL_MOL_to_AU * bohr2ang
KCAL_MOL_A2_to_AU = KCAL_MOL_A_to_AU * bohr2ang
A_to_AU = 1/bohr2ang


class Units(Enum):
    A = 0
    AU = 1
    KCAL = 2