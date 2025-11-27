from time import time

import matplotlib.pyplot as plt

from HEAT_calculator.species import (
    calculate_dct_species,
    calculate_reference_species,
    get_elements_in_species,
)

if __name__ == "__main__":
    # Formation enthalpies at 0 K taken from ATcT ver. 1.220,
    # in kcal/mol
    species_atct = {
        "H2O": -57.0994264,
        "OH": 8.909178,
        "CO": -27.1986138,
        "CO2": -93.9557839,
        "SO": 1.4221,
        "SO2": -70.317878,
        "CH3OH": -45.40153,
        "CH3O": 6.92639,
        "CH2OH": -2.416348,
        "HCO": 9.89173,
        "HOCO": -43.293499,  # Trans HOCO
        "H2": 0.0,  # By definition
        "N2": 0.0,  # By definition
        "NH2": 45.15535,
        "CH3": 35.821941,
        "CH2NH2": 38.10468,
        "(CH3)2": -16.34799,
        "OH-": -33.2373327,
    }

    # Species to calculate.
    # In reality, we know the ground state multiplities of many of these of course,
    # but this is just to benchmark, and to make sure that we get the correct ones.
    smiles_dct = {
        "H2O": ("O", 0, None),
        "OH": ("[O][H]", 0, None),
        "CO": ("[C+]#[O+]", 0, None),
        "CO2": ("O=C=O", 0, None),
        "SO": ("[S]#O", 0, None),
        "SO2": ("O=S=O", 0, None),
        "CH3OH": ("CO", 0, None),
        "CH3O": ("C[O]", 0, None),
        "CH2OH": ("[CH2]O", 0, None),
        "HCO": ("[CH]=O", 0, None),
        "HOCO": ("O[C]=O", 0, None),
        "H2": ("[H][H]", 0, None),
        "N2": ("N#N", 0, None),
        "NH2": ("[NH2]", 0, None),
        "CH3": ("[CH3]", 0, None),
        "CH2NH2": ("[CH2]N", 0, None),
        "(CH3)2": ("CC", 0, None),
        "OH-": ("[OH]-", -1, None),
    }

    method = "G2-MP2-SVP"
    directory = "compare_ATcT_reduced_precision"
    force = False
    njobs = 2

    time_start = time()
    ground_states = calculate_dct_species(
        smiles_dct,
        directory=directory,
        method=method,
        reduce_coordinate_precision=True,
        force=force,
        njobs=njobs,
    )

    atoms = get_elements_in_species(ground_states.values())
    calculated_atoms = calculate_reference_species(
        atoms,
        method=method,
        directory=directory,
        force=force,
        njobs=njobs,
    )

    time_end = time()
    print(
        f"Using {njobs} jobs, calculating the species took {time_end - time_start:.2f} seconds"
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    plt.figure()
    running_sum_abs_deviation = 0.0
    for spec in ground_states:
        formation_enthalpy = ground_states[spec].calculate_enthalpy_of_formation(
            calculated_atoms
        )
        if ground_states[spec].num_electrons % 2 == 0:
            marker = "o"
            color = colors[0]
        else:
            marker = "X"
            color = colors[1]
        plt.scatter(
            species_atct[spec],
            formation_enthalpy,
            c=color,
            marker=marker,
            linewidth=0.05,
            edgecolor="k",
        )
        running_sum_abs_deviation += abs(formation_enthalpy - species_atct[spec])

    mean_absolute_deviaton = running_sum_abs_deviation / len(ground_states)
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()

    plt.gca().set_aspect("equal")
    plt.plot(
        [-1e10, 1e10], [-1e10, 1e10], c="gray", ls="dashed", alpha=0.5, zorder=0.99
    )
    plt.xlim(ylim)
    plt.ylim(ylim)

    plt.xlabel(r"Literature $\Delta H_{\mathrm{form}}$ (kcal mol$^{-1}$)")
    plt.ylabel(r"Calculated $\Delta H_{\mathrm{form}}$ (kcal mol$^{-1}$)")

    plt.text(
        0.05,
        0.95,
        f"Mean absolute deviation:\n{mean_absolute_deviaton:.2f} kcal mol$^{{-1}}$",
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
    )

    plt.tight_layout()

    plt.savefig("comparison_ATcT_G2.pdf")
    plt.show()
