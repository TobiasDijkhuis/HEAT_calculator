from python_tools import format_float_as_twocol_tex
from species import calculate_dct_species, get_reference_species

if __name__ == "__main__":
    force = False
    directory = "new_values_for_network"
    reference_species = get_reference_species(
        ["H", "O", "C", "S", "N"], directory=directory, force=force
    )

    species_dct = {
        "HCSH": ("[H]=[C]S", 0, None),
        "NSH": ("[N][S][H]", 0, None),
        "HNSH": ("[H][N]S", 0, None),
        "NSH2": ("[N]=[SH2]", 0, None),
        "H2NS": ("N[S]", 0, None),
        "CH2SH2": ("C=[SH2]", 0, None),
        "HSO": ("S[O]", 0, None),
        "HNSH2": ("N=[SH2]", 0, None),
        "OCSH": ("O=[C]S", 0, None),
        "OCHS": ("O=C[S]", 0, None),
        "HOCS": ("O[C]=S", 0, None),
        "HSO2": ("O=S[O]", 0, None),
        "HOSO": ("O[S]=O", 0, None),
        "HS2": ("S#[S]", 0, None),
    }

    calculated_species = calculate_dct_species(
        species_dct, force=force, directory=directory
    )

    sorted_species = dict(
        sorted(calculated_species.items(), key=lambda item: item[1].mass)
    )

    print(
        r"Species & SMILES & Multiplicity & $\Delta H_{\mathrm{form}}$ (kcal mol$^{-1}) \\"
    )
    for name, spec in sorted_species.items():
        enthalpy_of_formation = spec.calculate_enthalpy_of_formation(reference_species)
        print(
            f"{spec.format_name_as_tex()} & {spec.format_smiles_as_tex()} & {spec.multiplicity} & {format_float_as_twocol_tex(enthalpy_of_formation)} \\\\"
        )
