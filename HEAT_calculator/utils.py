import os
import re
import stat
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Literal


class InvalidMultiplicityError(Exception):
    """Error to indicate that the given multiplicity is impossible."""


class IncorrectGeneratedXYZ(Exception):
    """Error to indicate that a generated XYZ file does not match the desired molecular formula"""


class CalculationResult(Enum):
    """A useful way to easily determine whether the calculation of a species succeeded or not."""

    SUCCESS_CALCULATED = auto()
    """Calculation succeeded"""

    SUCCESS_READ = auto()
    """Energy was read from a previous time the Species was calculated"""

    FAILED_OPTIMIZATION = auto()
    """Calculation failed because the geometry optimization required too many steps"""

    FAILED_OTHER = auto()
    """Calculation failed for a reason that is not explicitly implemented"""


def read_final_energy_from_compound(filepath: str | Path) -> float:
    """Read the final energy after a compound method

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
    """An easy way to determine why a calculation failed."""
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
    """Set a file to be executable from subprocess.run

    Args:
        filepath (str | Path): file to set to be executable
    """
    os.chmod(filepath, mode=stat.S_IRWXG | stat.S_IRWXU | stat.S_IREAD)


def verify_type(value: Any, desired_type: type, name: str) -> None:
    """Verify the type of a parameter is as desired

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
    """Verify that the desired method is valid, and add "-ATOM" to it if the species is atomic

    Args:
        is_atomic (bool): Whether the species is atomic (i.e. consists of only one atom)
        method (Literal[available_methods]): one of the available methods. Case-insensitive. Default: "G2-MP2-SVP"

    Returns:
        method (str): the method
    """
    method = method.upper()
    if "ATOM" in method:
        raise ValueError(
            f"This code will take care of adding ATOM to the method if necessary."
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
    command = f"#!/usr/bin/bash"

    try:
        slurm_options = """#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --threads-per-core 1
#SBATCH --time 10:00:00 
#SBATCH --mem-per-cpu 5G"""
        command = "\n\n".join((command, slurm_options))
    except ImportError:
        pass

    if orca_output_path is None:
        orca_output_path = f"{os.path.splitext(orca_input_path)[0]}.out"
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
