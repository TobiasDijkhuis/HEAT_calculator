"""Collection of io functions."""

import re
from pathlib import Path
from textwrap import dedent

from utils import (
    CalculationResult,
    set_file_executable,
    slurm_manager_is_installed,
    verify_type,
)


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
        slurm_options = dedent("""\
            #SBATCH --nodes 1
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


def write_xyz(
    atoms: list[str],
    coordinates: list[list[float]],
    filepath: str | Path,
    comment: str | None = None,
    num_decimal_places: int | None = None,
) -> None:
    """Write an xyz file.

    Args:
        atoms (list[str]): list of atoms
        coordinates (list[list[float]]): list of coordinates in Angstrom
        filepath (str | Path): filepath
        comment (str | None): comment to put on the second line of the xyz file.
            Default: None
        num_decimal_places (str | None): number of decimal places to use in coordinates.
            If None, do not round at all.

    """
    if not len(atoms) == len(coordinates):
        raise ValueError(
            f"Number of atoms ({len(atoms)}) is not the same as the number of coordinates ({len(coordinates)})"
        )

    if comment is None:
        comment = ""

    if num_decimal_places is not None:
        verify_type(num_decimal_places, int, "num_decimal_places")
        if not num_decimal_places > 0:
            raise ValueError(
                f"num_decimal_places should be bigger than 0, but was {num_decimal_places}"
            )
        coordinates = [
            [round(coord, num_decimal_places) for coord in atom_coordinates]
            for atom_coordinates in coordinates
        ]

    coordinate_lines = "\n".join(
        [
            f"{atom}    {'    '.join([str(coord) for coord in coordinate])}"
            for atom, coordinate in zip(atoms, coordinates)
        ]
    )

    lines = "\n".join((f"{len(atoms)}\n{comment}", coordinate_lines, ""))
    with open(filepath, "w") as file:
        file.write(lines)
