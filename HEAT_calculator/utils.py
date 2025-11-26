import os
import re
import stat
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal


class InvalidMultiplicityError(Exception):
    """Error to indicate that the given multiplicity is impossible."""


class IncorrectGeneratedXYZ(Exception):
    """Error to indicate that a generated XYZ file does not match the desired molecular formula"""


class CalculationResult(Enum):
    SUCCESS = 1
    FAILED_OPTIMIZATION = 2
    FAILED_OTHER = 3


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


# See https://www.faccts.de/docs/orca/6.0/manual/contents/detailed/compound.html#list-of-known-simple-input-commands
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
