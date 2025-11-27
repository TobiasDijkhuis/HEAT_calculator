import numpy as np

from HEAT_calculator.utils import calculate_principal_moments_of_inertia, read_xyz

atoms, coordinates, comment = read_xyz(
    "compare_ATcT_reduced_precision/CO2_1/CO2_1_Compound_2.xyz"
)
moments, axes = calculate_principal_moments_of_inertia(atoms, coordinates)
