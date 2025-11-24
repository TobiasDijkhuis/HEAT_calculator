import re
from pathlib import Path
import stat
import matplotlib.pyplot as plt
from typing import Any
from datetime import datetime
from typing import Literal


class InvalidMultiplicityError(Exception):
    """Error to indicate that the given multiplicity is impossible."""


class IncorrectGeneratedXYZ(Exception):
    """Error to indicate that a generated XYZ file does not match the desired molecular formula"""


def read_final_energy_from_compound(filepath: str | Path) -> float:
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
    Path(filepath).chmod(mode=stat.S_IRWXG | stat.S_IRWXU | stat.S_IREAD)


def get_colors() -> list[str]:
    """Get all colors in the matplotlib color cycler"""
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


def verify_type(value: Any, desired_type: Any, name: str) -> None:
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


def get_orca_input(charge, multiplicity, xyz_path, method) -> str:
    return f"""# Automatically generated ORCA input at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Calculate high accuracy energies for the use of calculating thermodynamic values
!compound[{method}]
* xyzfile {charge} {multiplicity} {xyz_path}
%compound[{method}]
    with
        molecule = {xyz_path}
RunEnd"""
