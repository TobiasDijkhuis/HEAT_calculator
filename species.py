from __future__ import annotations

from dataclasses import dataclass
from uclchem.makerates.species import Species as UCLCHEMSpecies
from pathlib import Path
from collections import Counter
from subprocess import run
from typing import Literal
from time import time
import sys
import os

from utils import (
    InvalidMultiplicityError,
    IncorrectGeneratedXYZ,
    available_methods,
    read_final_energy_from_compound,
    set_file_executable,
    verify_type,
)
from data import (
    atomic_masses,
    atomic_numbers,
    experimental_formation_0K,
    HARTREE_TO_KCALPERMOL,
)


@dataclass
class Species:
    name: str
    smiles: str
    charge: int = 0
    multiplicity: int = 0

    def __post_init__(self) -> None:
        verify_type(self.charge, int, "charge")

        self.split_name = self.name.split("_")[0]

        self.constituents = self._find_constituents()
        self.mass = self._calculate_mass()
        self.num_electrons = self._calculate_num_electrons()

        self._verify_multiplicity()

    def write_input_files(
        self,
        directory: str | Path | None,
        method: Literal[available_methods] = "G2-MP2-SVP",
        reduce_coordinate_precision: bool = True,
    ) -> None:
        method = method.upper()
        if "ATOM" in method:
            raise ValueError()
        if method not in available_methods:
            raise ValueError()
        if self.is_atomic():
            method += "-ATOM"

        if directory is not None:
            self.directory = Path(directory) / self.name
        else:
            self.directory = Path(self.name)
        if not self.directory.is_dir():
            self.directory.mkdir(parents=True)

        self._generate_xyz()
        if reduce_coordinate_precision:
            self._reduce_coordinate_precision()

        input = f"""# Automatically generated ORCA input at TIME HERE
# Calculate high accuracy energies for the use of calculating thermodynamic values
!compound[{method}]
* xyzfile {self.charge} {self.multiplicity} {self.name}.xyz
%compound[{method}]
    with
        molecule = "{self.name}.xyz"
RunEnd"""
        with open(self.directory / f"{self.name}.inp", "w") as file:
            file.write(input)

    def _generate_xyz(self) -> None:
        with open(self.directory / f"{self.name}.smi", "w") as file:
            file.write(self.smiles)
        command = f"obabel --title {self.name} -ismi {self.directory/self.name}.smi -oxyz -O {self.directory/self.name}.xyz -h --gen3d --best"
        result = run(command.split(), text=True, capture_output=True)
        if result.stderr and not result.stderr == "1 molecule converted\n":
            raise RuntimeError(result.stderr)

        self._verify_generated_xyz()

    def _verify_generated_xyz(self) -> None:
        with open(self.directory / f"{self.name}.xyz", "r") as file:
            lines = file.readlines()
        num_atoms = int(lines[0].strip())
        if not self.num_atoms == num_atoms:
            raise IncorrectGeneratedXYZ(
                f"Number of atoms in generated xyz file ({num_atoms}) does not match the number of atoms inferred from the molecular formula ({self.num_atoms}) of {self.split_name}"
            )
        atoms = [line.split()[0] for line in lines[2:]]
        if not sorted(atoms) == sorted(self.constituents):
            raise IncorrectGeneratedXYZ()

    def _reduce_coordinate_precision(self) -> None:
        """This can help with ORCA detecting symmetries and keeping
        geometries more constrained during optimization"""
        with open(self.directory / f"{self.name}.xyz") as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            if i < 2:
                lines[i] = line.strip()
                continue
            line_split = line.split()
            lines[i] = (
                f"  {line_split[0]}  {'  '.join([f'{float(val):.3f}' for val in line_split[1:]])}"
            )
        with open(self.directory / f"{self.name}.xyz", "w") as file:
            file.write("\n".join(lines))

    def calculate_energy(self, force: bool = False) -> None:
        if (
            self.split_name == "H"
            and self.smiles == "[H]"
            and self.charge == 0
            and self.multiplicity == 2
        ):
            self.energy = -0.5 * HARTREE_TO_KCALPERMOL
            return

        optimization_failed_path = self.directory / ".optimization_failed"
        if not force and optimization_failed_path.is_file():
            print(
                f"Tried to run calculation for {self.name} previously, but optimization failed."
            )
            self.energy = None
            return
        compound_path = self.directory / f"{self.name}_compound_detailed.txt"
        if not force and compound_path.is_file():
            self.energy = (
                read_final_energy_from_compound(compound_path) * HARTREE_TO_KCALPERMOL
            )
            return

        command = f"#!/usr/bin/bash\n\n/opt/orca-6.1.0/orca {self.name}.inp > {self.name}.out\n\nrm *tmp*\nrm *gbw\nrm *densities*"
        init_dir = os.getcwd()
        os.chdir(self.directory)
        with open("run.sh", "w") as file:
            file.write(command)
        set_file_executable("run.sh")

        print(f"Running calculation of {self}")
        time_start = time()
        result = run("./run.sh", text=True, capture_output=True)
        print(result.stderr)
        time_end = time()
        print(f"  Took {time_end - time_start:.2f} seconds")
        os.chdir(init_dir)

        if not compound_path.is_file():
            print("  Calculation failed.")
            self.energy = None
            with open(self.directory / f"{self.name}.out") as file:
                lines = file.readlines()
            for line in lines[::-1]:
                if (
                    "The optimization has not yet converged - more geometry cycles are needed"
                    in line
                ):
                    print(
                        "  Optimization required more steps. Making file to indicate this for future runs"
                    )
                    optimization_failed_path.touch()
                    break
            return
        self.energy = (
            read_final_energy_from_compound(compound_path) * HARTREE_TO_KCALPERMOL
        )

    def calculate_enthalpy_of_formation(
        self,
        calculated_reference_atoms: dict[str, Species],
        experimental_reference_atoms: dict[str, float] = experimental_formation_0K,
    ) -> float:
        formation_enthalpy = self.energy
        for atom in self.constituents:
            formation_enthalpy += (
                experimental_reference_atoms[atom]
                - calculated_reference_atoms[atom].energy
            )
        return formation_enthalpy

    def _find_constituents(self) -> list[str]:
        spec = UCLCHEMSpecies([self.split_name] + [0] * 11)
        constituents = spec.find_constituents(quiet=True)
        if isinstance(constituents, Counter):
            constituents = list(constituents.elements())
        return [el.capitalize() for el in constituents]

    @property
    def num_atoms(self) -> int:
        return len(self.constituents)

    def is_atomic(self) -> bool:
        return self.num_atoms == 1

    def _calculate_num_electrons(self) -> int:
        num_electrons = 0
        for atom in self.constituents:
            num_electrons += atomic_numbers[atom]
        num_electrons -= self.charge
        return num_electrons

    def _calculate_mass(self) -> float:
        mass = 0.0
        for atom in self.constituents:
            mass += atomic_masses[atom]
        return mass

    def _verify_multiplicity(self) -> None:
        verify_type(self.multiplicity, int, "multiplicity")

        if (self.multiplicity - 1.0) / 2.0 >= self.num_electrons:
            raise InvalidMultiplicityError()
        if self.num_electrons % 2 == 0:
            if self.multiplicity % 2 != 1:
                raise InvalidMultiplicityError()
        elif self.num_electrons % 2 == 1:
            if self.multiplicity % 2 != 0:
                raise InvalidMultiplicityError()

    def format_name_as_tex(self) -> str:
        return f"\\ch{{{self.split_name}}}"

    def format_smiles_as_tex(self) -> str:
        return self.smiles.replace("#", "\\#")


def get_possible_multiplicities(
    name: str, smiles: str, charge: int = 0, max_multiplicity: int = 4
) -> list[Species]:
    if max_multiplicity % 2 != 0:
        print(
            "Warning: max_multiplicity is odd, which means that more states will be checked for systems with an even number of electrons than systems with an odd amount of electrons"
        )
    species = []
    for multiplicity_try in range(1, max_multiplicity + 1):
        try:
            spec = Species(
                name=f"{name}_{multiplicity_try}",
                smiles=smiles,
                charge=charge,
                multiplicity=multiplicity_try,
            )
        except InvalidMultiplicityError:
            continue
        species.append(spec)
    return species


def get_ground_state_species(species_list: list[Species], name: str) -> Species:
    """Get the ground state of a certain species."""
    minimum_energy = sys.float_info.max
    minimum_spec = None
    for spec in species_list:
        if spec.split_name != name:
            continue
        if spec.energy is None:
            continue
        if spec.energy < minimum_energy:
            minimum_energy = spec.energy
            minimum_spec = spec
    if minimum_spec is None:
        raise ValueError()
    minimum_spec.name = minimum_spec.split_name
    return minimum_spec


def get_reference_species(
    reference_atoms: list[str],
    directory: str | Path | None = None,
    method: Literal[available_methods] = "G2-MP2-SVP",
    force: bool = False,
) -> dict[str, Species]:
    ground_species = {}
    for atom in reference_atoms:
        possibilities = get_possible_multiplicities(atom, f"[{atom}]", charge=0)
        for state in possibilities:
            state.write_input_files(directory=directory, method=method)
            state.calculate_energy(force=force)
        ground_species[atom] = get_ground_state_species(possibilities, atom)
    return ground_species
