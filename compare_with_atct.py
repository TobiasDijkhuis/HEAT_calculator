import matplotlib.pyplot as plt

from species import (Species, get_ground_state_species,
                     get_possible_multiplicities, get_reference_species)

method = "G2-MP2-SVP"
directory = "compare_ATcT_reduced_precision"
force = False

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

calculated_atoms = get_reference_species(
    ["H", "O", "C", "S", "N"],
    method=method,
    directory=directory,
    force=force,
)

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
    "H2": 0.0,
    "N2": 0.0,
    "NH2": 45.15535,
    "CH3": 35.821941,
    "CH2NH2": 38.10468,
    "(CH3)2": -16.34799,
    "OH-": -33.2373327,
}

smiles_dct = {
    "H2O": "O",
    "OH": "[O][H]",
    "CO": "[C+]#[O+]",
    "CO2": "O=C=O",
    "SO": "[S]#O",
    "SO2": "O=S=O",
    "CH3OH": "CO",
    "CH3O": "C[O]",
    "CH2OH": "[CH2]O",
    "HCO": "[CH]=O",
    "HOCO": "O[C]=O",
    "H2": "[H][H]",
    "N2": "N#N",
    "NH2": "[NH2]",
    "CH3": "[CH3]",
    "CH2NH2": "[CH2]N",
    "(CH3)2": "CC",
    "OH-": "[OH]-",
}

ground_states = {}
for spec, smiles in smiles_dct.items():
    possibilities = get_possible_multiplicities(spec, smiles)
    for state in possibilities:
        state.write_input_files(
            method=method, directory=directory, reduce_coordinate_precision=True
        )
        state.calculate_energy(force=force)
    ground_states[spec] = get_ground_state_species(possibilities, spec)

plt.figure()
for spec in species_atct:
    formation_enthalpy = ground_states[spec].calculate_enthalpy_of_formation(
        calculated_atoms
    )
    if ground_states[spec].num_electrons % 2 == 0:
        marker = "o"
        color = colors[0]
    else:
        marker = "X"
        color = colors[1]
    plt.scatter(species_atct[spec], formation_enthalpy, c=color, marker=marker)


xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()

plt.gca().set_aspect("equal")
plt.plot([-1e10, 1e10], [-1e10, 1e10], c="gray", ls="dashed", alpha=0.5, zorder=0.99)
plt.xlim(xlim)
plt.ylim(xlim)

plt.xlabel(r"Literature $\Delta H_{\mathrm{form}}$ (kcal mol$^{-1}$)")
plt.ylabel(r"Calculated $\Delta H_{\mathrm{form}}$ (kcal mol$^{-1}$)")

plt.tight_layout()
plt.savefig(f"comparison_ATcT_G2.pdf")
plt.show()
