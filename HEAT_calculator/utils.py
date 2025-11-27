"""Useful tools."""

import importlib
import os
import re
import stat
from enum import Enum, auto
from pathlib import Path
from textwrap import dedent
from typing import Any, Literal

import numpy as np

from .data import (
    AMU_TO_KG,
    ANGSTROM_TO_METERS,
    KGM2_TO_AMU_ANGSTROM2,
    atomic_masses,
    element_list,
)


class InvalidMultiplicityError(Exception):
    """Error to indicate that the given multiplicity is impossible."""


class IncorrectGeneratedXYZ(Exception):
    """Error to indicate that a generated XYZ file does not match
    the desired molecular formula.
    """


class CalculationResult(Enum):
    """A useful way to easily determine whether the calculation of a
    species succeeded or not.
    """

    SUCCESS_CALCULATED = auto()
    """Calculation succeeded"""

    SUCCESS_READ = auto()
    """Energy was read from a previous time the Species was calculated"""

    FAILED_OPTIMIZATION = auto()
    """Calculation failed because the geometry optimization required too many steps"""

    FAILED_OTHER = auto()
    """Calculation failed for a reason that is not explicitly implemented"""


def read_final_energy_from_compound(filepath: str | Path) -> float:
    """Read the final energy after a compound method.

    Args:
        filepath (str | Path): filepath of the "compound_detailed" file created by ORCA

    Returns:
        energy: the FinalEnergy in Hartree

    """
    with open(filepath, "r") as file:
        lines = file.readlines()
    regexp_finalenergy = re.compile("FINALENERGY")
    regexp_value = re.compile("Value")
    for i, line in enumerate(lines):
        if not regexp_finalenergy.search(line):
            continue
        for inner_line in lines[i:]:
            if not regexp_value.search(inner_line):
                continue
            energy = float(inner_line.split()[-1])
            return energy


def determine_reason_calculation_failed(filepath: str | Path) -> CalculationResult:
    """Determine why a calculation failed.

    Args:
        filepath (str | Path): filepath of orca output file

    Returns:
        CalculationResult: reason why the calculation failed

    """
    with open(filepath) as file:
        lines = file.readlines()
    for line in lines[::-1]:
        if (
            "The optimization has not yet converged - more geometry cycles are needed"
            in line
        ):
            return CalculationResult.FAILED_OPTIMIZATION
    return CalculationResult.FAILED_OTHER


def set_file_executable(filepath: str | Path) -> None:
    """Set a file to be executable from subprocess.run.

    Args:
        filepath (str | Path): file to set to be executable

    """
    os.chmod(filepath, mode=stat.S_IRWXG | stat.S_IRWXU | stat.S_IREAD)  # noqa: PTH101


def verify_type(value: Any, desired_type: type, name: str) -> None:  # noqa: ANN401
    """Verify the type of a parameter is as desired.

    Args:
        value (Any): parameter to check
        desired_type (Any): desired type of parameter
        name (str): name of parameter to make error more clear

    """
    if not isinstance(value, desired_type):
        raise TypeError(
            f"{name} should be of type {desired_type} but was {type(value)}"
        )


""" List of available methods in ORCA to calculate high accuracy energies
See https://www.faccts.de/docs/orca/6.0/manual/contents/detailed/compound.html#list-of-known-simple-input-commands"""
available_methods = [
    "G2-MP2",
    "G2-MP2-SV",
    "G2-MP2-SVP",
    "CCCA-DZ-QCISD-T",
    "CCCA-TZ-QCISD-T",
    "CCCA-ATZ-QCISD-T",
    "CCCA-CBS-1",
    "CCCA-CBS-2",
]


def get_method(
    is_atomic: bool, method: Literal[available_methods] = "G2-MP2-SVP"
) -> str:
    """Verify that the desired method is valid, and add "-ATOM"
        to it if the species is atomic.

    Args:
        is_atomic (bool): Whether the species is atomic
            (i.e. consists of only one atom)
        method (Literal[available_methods]): one of the available methods.
            Case-insensitive. Default: "G2-MP2-SVP"

    Returns:
        method (str): the method

    """
    method = method.upper()
    if "ATOM" in method:
        raise ValueError(
            "This code will take care of adding ATOM to the method if necessary."
        )
    if method not in available_methods:
        raise ValueError(
            f"Method should be one of {available_methods}, but was {method}.\nSee https://www.faccts.de/docs/orca/6.0/manual/contents/detailed/compound.html#list-of-known-simple-input-commands for more info"
        )
    if is_atomic:
        method += "-ATOM"
    return method


def write_run_orca_file(
    run_path: str | Path,
    orca_input_path: str | Path,
    orca_output_path: str | Path | None = None,
    orca_path: str | Path | None = None,
) -> None:
    """Write the executable file that will run ORCA.

    Args:
        run_path (str | Path): path of run file
        orca_input_path (str | Path): ORCA input path
        orca_output_path (str | Path | None): path of ORCA output. If None,
            simply take the same name as orca_input_path, but with ".out"
        orca_path (str | Path | None): path to ORCA executable.
            If None, simply execute "orca". Default: None

    """
    command = "#!/usr/bin/bash"

    if slurm_manager_is_installed():
        slurm_options = dedent("""#SBATCH --nodes 1
            #SBATCH --ntasks-per-node 1
            #SBATCH --cpus-per-task 1
            #SBATCH --threads-per-core 1
            #SBATCH --time 10:00:00
            #SBATCH --mem-per-cpu 5G""")
        command = "\n\n".join((command, slurm_options))

    if orca_output_path is None:
        orca_output_path = f"{Path(orca_input_path).stem}.out"
    if orca_path is None:
        orca_path = "orca"

    command = "\n\n".join(
        (
            command,
            f"{orca_path} {orca_input_path} > {orca_output_path}\n\nrm *tmp*\nrm *gbw\nrm *densities*",
        )
    )

    with open(run_path, "w") as file:
        file.write(command)
    set_file_executable(run_path)


def determine_atoms_from_molecular_formula(formula: str) -> list[str]:
    """Determine the constituent atoms in a molecular formula.

    Args:
        formula (str): molecular formula

    Returns:
        atoms (list[str]): atoms in the molecular formula

    For example:
        >> determine_atoms_from_molecular_formula("H2O") -> ["H", "H", "O"]
        >> determine_atoms_from_molecular_formula("(CH)2") -> ["C", "H", "C", "H"]

    """
    # Adapted from https://github.com/uclchem/UCLCHEM/blob/main/src/uclchem/makerates/species.py
    verify_type(formula, str, "formula")

    if formula[0].isdigit():
        raise ValueError(
            f"First character of formula {formula} was a digit. Please put repeated parts in a bracket with number after, e.g. (CH3)2"
        )

    char_idx = 0
    atoms = []
    currently_in_bracket = False
    # loop over characters in species name to work out what it is made of
    while char_idx < len(formula):
        # if character isn't a + or - then check it, otherwise move on
        if formula[char_idx] not in ["+", "-", "(", ")"]:
            if char_idx + 1 < len(formula):
                # if next two characters are (eg) 'MG' then atom is Mg not M and G
                if formula[char_idx : char_idx + 2] in element_list:
                    j = char_idx + 2
                # otherwise work out which element it is
                elif formula[char_idx] in element_list:
                    j = char_idx + 1

            # if there aren't two characters left just try next one
            elif formula[char_idx] in element_list:
                j = char_idx + 1
            # if we've found a new element check for numbers otherwise print error
            if j <= char_idx:
                raise ValueError(
                    f"formula {formula} contains element(s) not in element list"
                )

            num_digits = find_number_of_consecutive_digits(formula, j)
            if num_digits == 0:
                nrepeat = 1
            else:
                nrepeat = int(formula[j : j + num_digits])
            for _ in range(nrepeat):
                if currently_in_bracket:
                    bracket_content.append(formula[char_idx:j])  # noqa: F821
                else:
                    atoms.append(formula[char_idx:j])
            char_idx = j + num_digits
        else:
            # if symbol is start of a bracketed part of molecule, keep track
            if formula[char_idx] == "(":
                currently_in_bracket = True
                bracket_content = []
            # if it's the end then add bracket contents to list
            elif formula[char_idx] == ")":
                if not currently_in_bracket:
                    raise ValueError(
                        "Found closing bracket before opening bracket in formula {formula}"
                    )
                currently_in_bracket = False
                num_digits = find_number_of_consecutive_digits(formula, char_idx + 1)
                if num_digits == 0:
                    nrepeat = 1
                else:
                    nrepeat = int(formula[char_idx + 1 : char_idx + 1 + num_digits])
                for _ in range(nrepeat):
                    atoms.extend(bracket_content)
                char_idx += num_digits
            char_idx += 1
    return atoms


def find_number_of_consecutive_digits(string: str, start: int) -> int:
    """Determine the number of consecutive digits in a string.

    Args:
        string (str): the string
        start (int): the starting index

    Returns:
        num_digits (int): the number of consecutive digits in the string
            starting from "start".

    For example:
        >> find_number_of_consecutive_digits("Hello123", 0) -> 0,
        >> find_number_of_consecutive_digits("Hello123", 5) -> 3,
        >> find_number_of_consecutive_digits("Hello123", 6) -> 2,
        >> find_number_of_consecutive_digits("He1llo23", 2) -> 1,

    """
    num_digits = 0
    while start + num_digits < len(string) and string[start + num_digits].isdigit():
        num_digits += 1
    return num_digits


def find_all_numbers(string: str) -> dict[int, str]:
    """Find all numbers in a string.

    Args:
        string (str): string

    Returns:
        dict[int, str]: dictionary with keys the starting index,
            and values the number at that index.

    For example:
        >> find_all_numbers("Hello123") -> {5: '123'}
        >> find_all_numbers("He1llo23") -> {2: '1', 6: '23'}

    """
    return {m.start(): m.group() for m in re.finditer(r"\d+", string)}


def format_formula_as_tex(formula: str, use_ch: bool = True) -> str:
    """Format a molecular formula as a string in TeX.

    Args:
        formula (str): molecular formula
        use_ch (bool): Whether to use ch to format it. Default: True

    Returns:
        str: formatted formula

    """
    if use_ch:
        return f"\\ch{{{formula}}}"
    numbers = find_all_numbers(formula)
    to_skip = 0
    for j, number in numbers.items():
        formula = (
            formula[: j + to_skip]
            + r"$_{{{0}}}$".format(number)
            + formula[j + to_skip + len(number) :]
        )
        to_skip += len(number) + 4
    return formula


def slurm_manager_is_installed() -> bool:
    """Find whether the slurm_manager package is installed, and slurm is active.

    Returns:
        bool: whether slurm_manager is installed and slurm is running.

    """
    if importlib.util.find_spec("slurm_manager") is None:
        return False
    try:
        # Try to import it. This import will fail if slurm is not active
        import slurm_manager  # noqa: F401

        return True
    except ImportError:
        return False


def read_xyz(filepath: str | Path) -> tuple[list[str], list[list[float]], str]:
    """Read xyz file.

    Args:
        filepath (str | Path): xyz path

    Returns:
        atoms (list[str]): list of atoms
        coordinates (list[list[float]]): 2-dimensional list of coordinates
            of each atom
        comment (str): comment on second line of file

    """
    with open(filepath) as file:
        lines = file.readlines()
    natoms = int(lines[0].strip())
    comment = lines[1].strip()
    atoms = [""] * natoms
    coordinates = [[0.0, 0.0, 0.0]] * natoms
    for atom_idx, line in enumerate(lines[2:]):
        split_line = line.split()
        atoms[atom_idx] = split_line[0]
        coordinates[atom_idx] = [float(field) for field in split_line[1:4]]
    return atoms, coordinates, comment


def calculate_center_of_mass(
    coordinates: list[list[float]] | np.ndarray, masses: list[float] | np.ndarray
) -> list[float]:
    """Calculate the center of mass for a set of point masses.

    Args:
        coordinates (list[list[float]] | np.ndarray): list of coordinates
        masses (list[float] | np.ndarray): masses of the point masses

    Returns:
        np.ndarray: center of mass

    """
    if isinstance(coordinates, list) and isinstance(coordinates[0], list):
        coordinates = np.array(coordinates)
    if isinstance(masses, list):
        masses = np.array(masses)

    return np.sum(coordinates.T * masses, axis=1) / np.sum(masses)


def calculate_inertia_tensor(
    atoms: list[str], coordinates: list[list[float]] | np.ndarray
) -> np.ndarray:
    """Get the moment of inertia tensor.

    Args:
        atoms (list[str]): list of atoms
        coordinates (list[float[float]] | np.ndarray): list of coordinates in Angstrom

    Returns:
        inertia_tensor (np.ndarray): inertia tensor in kg m^2

    """
    # https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
    if isinstance(coordinates, list) and isinstance(coordinates[0], list):
        coordinates = np.array(coordinates)

    masses = np.array([atomic_masses[atom] for atom in atoms])
    center_of_mass = calculate_center_of_mass(coordinates, masses)

    # Calculate moments of inertia to axes wrt COM
    coordinates = coordinates - center_of_mass

    inertia_tensor = np.zeros((3, 3))
    for mass, coords in zip(masses, coordinates):
        inertia_tensor[0, 0] += mass * (coords[1] * coords[1] + coords[2] * coords[2])
        inertia_tensor[1, 1] += mass * (coords[0] * coords[0] + coords[2] * coords[2])
        inertia_tensor[2, 2] += mass * (coords[0] * coords[0] + coords[1] * coords[1])
        inertia_tensor[0, 1] -= mass * coords[0] * coords[1]
        inertia_tensor[0, 2] -= mass * coords[0] * coords[2]
        inertia_tensor[1, 2] -= mass * coords[1] * coords[2]
    inertia_tensor[1, 0] = inertia_tensor[0, 1]
    inertia_tensor[2, 0] = inertia_tensor[0, 2]
    inertia_tensor[2, 1] = inertia_tensor[1, 2]
    return inertia_tensor * AMU_TO_KG * ANGSTROM_TO_METERS * ANGSTROM_TO_METERS


def calculate_principal_moments_of_inertia(
    atoms: list[str],
    coordinates: list[list[float]],
    eps: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the principal moments of inertia and the principal axes.

    Args:
        atoms (list[str]): list of atoms
        coordinates (list[float[float]] | np.ndarray): list of coordinates in Angstrom
        eps (float): if a principal moment of inertia is below this, set it to 0.
            Default: 10^-9 amu Angstrom^2

    Returns:
        principal_moments (ndarray): array of length 3 with the three principal
            moments of inertia in amu Angstrom^2
        principal_axes (ndarray): matrix of shape 3x3 with three principal moment axes

    """
    inertia_tensor = calculate_inertia_tensor(atoms, coordinates)
    principal_moments, principal_axes = np.linalg.eig(inertia_tensor)
    indeces = np.argsort(principal_moments)

    principal_moments = principal_moments[indeces] * KGM2_TO_AMU_ANGSTROM2
    principal_moments[principal_moments < eps] = 0.0
    return principal_moments, principal_axes[indeces]
