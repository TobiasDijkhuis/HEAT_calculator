import re
from pathlib import Path
import stat
import matplotlib.pyplot as plt
from typing import Any


class InvalidMultiplicityError(Exception):
    """Error to indicate that the given multiplicity is impossible."""


class IncorrectGeneratedXYZ(Exception):
    """Error to indicate that a generated XYZ file does not match the desired molecular formula"""


available_methods = ["G2-MP2", "G2-MP2-SV", "G2-MP2-SVP"]  # TODO: Add ccCC to this.


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
