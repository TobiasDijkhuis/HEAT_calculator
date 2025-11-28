"""A collection of helpful things involving chemical species."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path
from subprocess import run
from textwrap import dedent
from time import time
from typing import Literal

from tqdm import tqdm

from .data import (
    HARTREE_TO_KCALPERMOL,
    atom_ground_state_multiplicities,
    atomic_masses,
    atomic_numbers,
    experimental_formation_0K,
)
from .io import (
    determine_reason_calculation_failed,
    read_final_energy_from_compound,
    read_xyz,
    write_run_orca_file,
    write_xyz,
)
from .utils import (
    CalculationResult,
    IncorrectGeneratedXYZ,
    InvalidMultiplicityError,
    available_methods,
    determine_atoms_from_molecular_formula,
    format_formula_as_tex,
    get_method,
    slurm_manager_is_installed,
    verify_type,
)


@dataclass
class Species:
    """Species. TODO elaborate."""

    name: str
    smiles: str
    charge: int = 0
    multiplicity: int = 1
    # TODO: Maybe rename "name" to "formula", and allow "name" to be an optional
    #   string for things like name="methanol", formula="CH3OH".
    # TODO: Move smiles from here to create_input_files as optional argument.
    #   Ensure either atoms_and_coordinates or smiles is passed there,
    #   we really dont need the smiles if we have the atoms and coordinates.

    def __post_init__(self) -> None:
        """Verify types of arguments of __init__, and calculate things like the mass."""
        verify_type(self.name, str, "name")
        verify_type(self.smiles, str, "smiles")
        verify_type(self.charge, int, "charge")

        self.formula = self.name.split("_")[0]

        self.constituents = self._find_constituents()
        self.num_electrons = self._calculate_num_electrons()
        self._verify_multiplicity()

        self.elements = list(set(self.constituents))
        self.mass = self._calculate_mass()

    def write_input_files(
        self,
        directory: str | Path | None = None,
        method: Literal[available_methods] = "G2-MP2-SVP",
        atoms_and_coordinates: tuple[list[str], list[list[float]]] | None = None,
        reduce_coordinate_precision: bool = True,
    ) -> None:
        """Write the input files for the ORCA calculation.

        Args:
            directory (str | Path | None): directory to calculate in. Default: None
            method (Literal[available_methods]): method to use. Default: "G2-MP2-SVP"
            atoms_and_coordinates (tuple[list[str], list[list[float]]] | None):
                list of atoms, and list of coordinates of atoms.
                If None, generate the structure using OpenBabel.
            reduce_coordinate_precision (bool): whether to reduce the precision of
                coordinates in the generated xyz file. This can help with ORCA
                inferring incorrect symmetries. Default: True

        """
        self._setup_paths(directory=directory)

        self._generate_xyz(
            atoms_and_coordinates=atoms_and_coordinates,
            reduce_coordinate_precision=reduce_coordinate_precision,
        )

        input = self._get_orca_input(method=method)
        with open(self.orca_inp_path) as file:
            file.write(input)

    def _setup_paths(self, directory: str | Path | None = None) -> None:
        """Set up the paths for input, xyz and smiles.

        Args:
            directory (str | Path | None): directory to calculate in. Default: None

        """
        self.directory_safe_name = self.name.replace(")", "b").replace("(", "b")

        if directory is not None:
            self.directory = Path(directory) / self.directory_safe_name
        else:
            self.directory = Path(self.directory_safe_name)
        if not self.directory.is_dir():
            self.directory.mkdir(parents=True)

        self.smi_path = self.directory / f"{self.directory_safe_name}.smi"
        self.xyz_path = self.directory / f"{self.directory_safe_name}.xyz"
        self.orca_inp_path = self.directory / f"{self.directory_safe_name}.inp"

    def _generate_xyz(
        self,
        atoms_and_coordinates: tuple[list[str], list[list[float]]] | None = None,
        reduce_coordinate_precision: bool = True,
    ) -> None:
        """Generate the coordinates of the Species.

        Args:
            atoms_and_coordinates (tuple[list[str], list[list[float]]] | None):
                list of atoms, and list of coordinates of atoms in Angstrom.
                If None, generate the structure using OpenBabel.
            reduce_coordinate_precision (bool): whether to reduce the precision of
                coordinates in the generated xyz file. This can help with ORCA
                inferring incorrect symmetries. Default: True

        """
        if atoms_and_coordinates is None:
            with open(self.smi_path, "w") as file:
                file.write(self.smiles)
            command = f'obabel --title "{self.name}, created from SMILES {self.smiles}" -ismi {self.smi_path} -oxyz -O {self.xyz_path} -h --gen3d --best'
            try:
                result = run(command.split(), text=True, capture_output=True)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    "Command 'obabel' was not found. OpenBabel is required for 3D geometry generation.\nSee https://openbabel.org/docs/Installation/install.html"
                ) from e
            if not result.stderr == "1 molecule converted\n":
                raise RuntimeError(
                    f"Error in generating xyz file of {self}\n{result.stderr}"
                )
        else:
            write_xyz(
                *atoms_and_coordinates,
                self.xyz_path,
                comment=f"{self.name}, created from user-supplied coordinates",
            )

        self._process_generated_xyz(
            reduce_coordinate_precision=reduce_coordinate_precision
        )

    def _process_generated_xyz(self, reduce_coordinate_precision: bool = True) -> None:
        """Verify that the generated xyz structure has the correct number of
        atoms and correct number of each element.

        Args:
            reduce_coordinate_precision (bool): whether to reduce the precision of
                coordinates in the generated xyz file. This can help with ORCA
                inferring incorrect symmetries. Default: True

        """
        atoms, coordinates, comment = read_xyz(self.xyz_path)
        if not self.num_atoms == len(atoms):
            raise IncorrectGeneratedXYZ(
                f"Number of atoms in generated xyz file ({len(atoms)}) does not match the number of atoms inferred from the molecular formula ({self.num_atoms}) of {self.formula}. Check smiles ({self.smiles})"
            )
        if not sorted(atoms) == sorted(self.constituents):
            raise IncorrectGeneratedXYZ(
                f"The atoms in the generated xyz file and inferred from the formula did not match.\nGenerated: {sorted(atoms)}\nFormula: {sorted(self.constituents)}"
            )

        if not reduce_coordinate_precision:
            return

        write_xyz(
            atoms,
            coordinates,
            self.xyz_path,
            comment=comment,
            num_decimal_places=3,
        )

    def _get_orca_input(self, method: Literal[available_methods] = "G2-MP2-SVP") -> str:
        method = get_method(self.is_atomic(), method=method)
        input_text = f"""# Automatically generated ORCA input at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        # Calculate high accuracy energies for the use of calculating thermodynamic values
        !compound[{method}]
        * xyzfile {self.charge} {self.multiplicity} {self.xyz_path.name}
        %compound[{method}]
            with
                molecule = {self.xyz_path.name}
        RunEnd"""
        return dedent(input_text)

    def calculate_energy(
        self, orca_path: str | Path | None = None, force: bool = False
    ) -> CalculationResult:
        """Calculate the energy of the Species.

        Args:
            orca_path (str | Path | None): path to ORCA executable.
                If None, simply execute "orca". Default: None
            force (bool): whether to do the calculation, even if it was
                attempted previously. Default: False

        Returns:
            CalculationResult: the result of the calculation.
                Indicates whether the calculation succeeded or not.

        """
        if self.smiles == "[H]" and self.charge == 0 and self.multiplicity == 2:
            # The energy of the hydrogen atom is -0.5 Hartree by definition
            self.energy = -0.5 * HARTREE_TO_KCALPERMOL
            return CalculationResult.SUCCESS_CALCULATED

        self._check_necessary_input_files()

        optimization_failed_path = self.directory / ".optimization_failed"
        if not force and optimization_failed_path.is_file():
            return CalculationResult.FAILED_OPTIMIZATION

        compound_path = (
            self.directory / f"{self.directory_safe_name}_compound_detailed.txt"
        )
        if not force and compound_path.is_file():
            self.energy = (
                read_final_energy_from_compound(compound_path) * HARTREE_TO_KCALPERMOL
            )
            return CalculationResult.SUCCESS_READ

        if orca_path is None:
            orca_path = "orca"
        write_run_orca_file(
            self.directory / "run.sh",
            self.orca_inp_path.name,
            orca_path=orca_path,
        )

        if slurm_manager_is_installed():
            from slurm_manager.job import SlurmJob

            job = SlurmJob.start_from_command("sbatch run.sh", directory=self.directory)
            # TODO: This is very inefficient. For every Species, a new slurm
            # job will be started, but also a process by Pool() will be started,
            # so two processes, and the one spawned by the pool will just be waiting.
            # How to do this better?
            try:
                job.wait()
            except KeyboardInterrupt as e:
                job.cancel()
                raise KeyboardInterrupt() from e
        else:
            init_dir = Path.cwd()
            os.chdir(self.directory)
            result = run("./run.sh", text=True, capture_output=True)
            if f"{orca_path}: command not found" in result.stderr:
                raise FileNotFoundError(
                    f"Command {orca_path} was not found. Please set path to ORCA executable correctly."
                )
            os.chdir(init_dir)

        if compound_path.is_file():
            self.energy = (
                read_final_energy_from_compound(compound_path) * HARTREE_TO_KCALPERMOL
            )
            return CalculationResult.SUCCESS_CALCULATED

        reason = determine_reason_calculation_failed(
            self.directory / f"{self.directory_safe_name}.out"
        )
        if reason == CalculationResult.FAILED_OPTIMIZATION:
            optimization_failed_path.touch()
        return reason

    def _check_necessary_input_files(self) -> None:
        """Check that the input files are written in the correct directories."""
        # This needs to be redone. If the input files have not been written,
        # also self.directory, self.xyz_path, self.orca_inp_path do not exist.
        # Maybe do hasattr(self, 'directory') stuff?
        if not self.directory.is_dir():
            raise NotADirectoryError(
                f"{self.directory} is not a directory. This directory should be created during Species.write_input_files. Write input files before trying to calculate energy."
            )
        if not self.xyz_path.is_file():
            raise FileNotFoundError(f"File {self.xyz_path} does not exist.")
        if not self.orca_inp_path.is_file():
            raise FileNotFoundError(f"File {self.orca_inp_path} does not exist.")

    def calculate_enthalpy_of_formation(
        self,
        calculated_reference_atoms: dict[str, Species],
        experimental_reference_atoms: dict[str, float] = experimental_formation_0K,
    ) -> float:
        """Calculate the enthalpy of formation.

        Args:
            calculated_reference_atoms (dict[str, Species]): dictionary of calculated
                reference atoms
            experimental_reference_atoms (dict[str, float]): dictionary of
                enthalpies of formation of reference atoms

        Returns:
            formation_enthalpy (float): enthalpy of formation in kcal/mol

        """
        if not hasattr(self, "energy"):
            raise AttributeError(
                f"Calculating the enthalpy of formation of Species {self} requires the energy. Calculate the energy first using Species.calculate_energy"
            )
        if self.energy is None:
            raise ValueError(
                f"Energy of Species {self} was None. Calculating the enthalpy of formation requires valid energy"
            )
        formation_enthalpy = self.energy
        for atom in self.constituents:
            formation_enthalpy += (
                experimental_reference_atoms[atom]
                - calculated_reference_atoms[atom].energy
            )
        return formation_enthalpy

    def get_optimized_geometry(self) -> tuple[list[str], list[list[float]], str]:
        """Get the optimized geometry."""
        optimized_geometry_path = (
            self.directory / f"{self.directory_safe_name}_Compound_2.xyz"
        )
        if not optimized_geometry_path.is_file():
            raise FileNotFoundError(
                f"Optimized geometry path {optimized_geometry_path} does not exist. First calculate energy using Species.calculate_energy"
            )
        return read_xyz(self.directory / f"{self.directory_safe_name}_Compound_2.xyz")

    def _find_constituents(self) -> list[str]:
        """Loop through the species' name and work out what its consituent atoms are.

        Returns:
            atoms (list[str]): list of atoms in molecule

        """
        return determine_atoms_from_molecular_formula(self.formula)

    def _check_charge_from_name(self) -> None:
        """Check the charge from the name, and corrects the given charge."""
        # TODO: Remove this?
        if "-" in self.name:
            if self.charge != -1:
                print(f"Warning: Assuming the ion {self.name} is singly charged")
                print(
                    f"Found '-' in name of species {self}, but charge was {self.charge}. Setting charge to -1"
                )
                self.charge = -1
        elif "+" in self.name:
            if self.charge != 1:
                print(f"Warning: Assuming the ion {self.name} is singly charged")
                print(
                    f"Found '+' in name of species {self}, but charge was {self.charge}. Setting charge to 1"
                )
                self.charge = 1

    @property
    def num_atoms(self) -> int:
        """Number of atoms inferred from molecular formula.

        Returns:
            int: number of atoms in the molecule

        """
        return len(self.constituents)

    def is_atomic(self) -> bool:
        """Whether the species is atomic.

        Returns:
            bool: whether the species is atomic

        """
        return self.num_atoms == 1

    def _calculate_num_electrons(self) -> int:
        """Calculate the number of electrons in the molecule.

        Returns:
            num_electrons (int): Number of electrons in the molecule

        """
        num_electrons = 0
        for atom in self.constituents:
            num_electrons += atomic_numbers[atom]
        num_electrons -= self.charge
        return num_electrons

    def _calculate_mass(self) -> float:
        """Calculate the mass of the molecule.

        Returns:
            mass (float): inferred mass of the molecule

        """
        mass = 0.0
        for atom in self.constituents:
            mass += atomic_masses[atom]
        return mass

    def _verify_multiplicity(self) -> None:
        """Verify the given multiplicity. If the number of electrons is even,
        the multiplicity should be odd and vice versa.
        """
        verify_type(self.multiplicity, int, "multiplicity")
        if self.multiplicity < 1:
            raise InvalidMultiplicityError(
                f"multiplicity should be at least 1, but was {self.multiplicity} in {self.name}"
            )
        if (self.multiplicity - 1.0) / 2.0 >= self.num_electrons:
            raise InvalidMultiplicityError(
                f"The multiplicity cannot be higher than 2*num_electrons+1={self.num_electrons * 2 + 1}, but was {self.multiplicity} in {self.name}"
            )
        if self.num_electrons % 2 == 0:
            if self.multiplicity % 2 != 1:
                raise InvalidMultiplicityError(
                    f"Number of electrons in {self.name} is even ({self.num_electrons}), so the multiplicity should be odd, but was {self.multiplicity}"
                )
        elif self.num_electrons % 2 == 1:
            if self.multiplicity % 2 != 0:
                raise InvalidMultiplicityError(
                    f"Number of electrons in {self.name} is odd ({self.num_electrons}), so the multiplicity should be even, but was {self.multiplicity}"
                )

    def format_formula_as_tex(self, use_ch: bool = True) -> str:
        """Format the formula of the Species as a string in LaTeX.

        Returns:
            str: formatted formula

        """
        return format_formula_as_tex(self.formula, use_ch=use_ch)

    def format_smiles_as_tex(self) -> str:
        """Format the smiles of the Species as a string in LaTeX.

        Returns:
            str: formatted smiles

        """
        return self.smiles.replace("#", "\\#")


def get_possible_multiplicities(
    name: str, smiles: str, charge: int = 0, max_multiplicity: int = 4
) -> list[Species]:
    """Get Species instances with the possible multiplicities.

    Args:
        name (str): name of the species
        smiles (str): smiles string of the species
        charge (int): charge of the species
        max_multiplicity (int): maximum multiplicity to try. Default: 4

    Returns:
        species (list[Species]): list of Species instances with possible multiplicities

    """
    verify_type(max_multiplicity, int, "max_multiplicity")
    if max_multiplicity < 1:
        raise ValueError(
            f"max_multiplicity should be at least 1, but was {max_multiplicity}"
        )

    if max_multiplicity % 2 == 1:
        print(
            f"Warning: max_multiplicity is odd ({max_multiplicity}), which means that more states will be checked for systems with an even number of electrons than systems with an odd amount of electrons"
        )
    try:
        spec = Species(
            name=f"{name}_1",
            smiles=smiles,
            charge=charge,
            multiplicity=1,
        )
        multiplicity_should_be_even = False
        if max_multiplicity <= 2:
            return [spec]
    except InvalidMultiplicityError:
        multiplicity_should_be_even = True

    return [
        Species(
            name=f"{name}_{state}", smiles=smiles, charge=charge, multiplicity=state
        )
        for state in range(
            1 + int(multiplicity_should_be_even), max_multiplicity + 1, 2
        )
    ]


def get_ground_state_species(species_list: list[Species], name: str) -> Species:
    """Get the ground state of a certain species.

    Args:
        species_list (list[Species]): list of Species with calculated energies
        name (str): name of species to filter for

    Returns:
        minimum_spec (Species): Species instance with the minimum energy.
            Its name is changed to the Species.formula.

    """
    minimum_energy = sys.float_info.max
    minimum_spec = None
    for spec in species_list:
        if spec.formula != name:
            continue
        if spec.energy is None:
            continue
        if spec.energy < minimum_energy:
            minimum_energy = spec.energy
            minimum_spec = spec
    if minimum_spec is None:
        raise ValueError(
            f"No suitable Species with name {name} with a valid energy was found in species_list"
        )
    minimum_spec.name = minimum_spec.formula
    return minimum_spec


def _calculate_wrapper(
    species: Species, orca_path: str | Path | None = None, force: bool = False
) -> tuple[Species, CalculationResult, float]:
    """Wrap around Species.calculate_energy for the use in multiprocessing.pool.Pool().

    Args:
        species (Species): the Species instance to calculate
        orca_path (str | Path | None): path to ORCA executable.
            If None, simply execute "orca". Default: None
        force (bool): whether to do the calculation, even if it was
            attempted previously. Default: False

    Returns:
        tuple[Species, CalculationResult, float]: a tuple of the calculated Species,
            CalculationResult indicating whether the calculation succeeded or not,
            and the duration of the calculation.

    """
    time_start = time()
    result = species.calculate_energy(orca_path=orca_path, force=force)
    return species, result, time() - time_start


def calculate_species(
    species_lst: list[Species],
    orca_path: str | Path | None = None,
    directory: str | Path | None = None,
    method: Literal[available_methods] = "G2-MP2-SVP",
    force: bool = False,
    reduce_coordinate_precision: bool = True,
    njobs: int = 1,
    disable_progress_bar: bool = False,
) -> dict[str, Species]:
    """Calculate the energies of a list of species.

    Args:
        species_lst (list[Species]): list of all species
        orca_path (str | Path | None): path to ORCA executable.
            If None, simply execute "orca". Default: None
        directory (str | Path | None): directory to calculate in. Default: None
        method (Literal[available_methods]): method to use. Default: "G2-MP2-SVP"
        force (bool): whether to do the calculation, even if it was attempted
            previously. Default: False
        reduce_coordinate_precision (bool): whether to reduce the precision of
            coordinates in the generated xyz file. This can help with ORCA inferring
            incorrect symmetries. Default: True
        njobs (int): Number of calculations to run at the same time. Default: 1
        disable_progress_bar (bool): whether to disable the progress bar.
            Default: False

    Return:
        calculated_species (dict[str, Species]): dictionary of calculated species

    """
    for spec in species_lst:
        spec.write_input_files(
            directory=directory,
            method=method,
            reduce_coordinate_precision=reduce_coordinate_precision,
        )

    calculated_species = {}
    with (
        Pool(njobs) as pool,
        tqdm(total=len(species_lst), disable=disable_progress_bar, position=1) as pbar,
        tqdm(
            disable=disable_progress_bar, position=0, bar_format="{desc}"
        ) as calculatedbar,
    ):
        calculatedbar.set_description_str("Starting calculations")
        for spec, result, duration in pool.imap_unordered(
            partial(_calculate_wrapper, orca_path=orca_path, force=force),
            species_lst,
        ):
            calculatedbar.set_description_str(
                f"Calculation of {spec.name} done, took {duration:.2f} seconds. Result: {result.name}"
            )
            pbar.update()
            pbar.refresh()
            calculated_species[spec.name] = spec
    return calculated_species


def calculate_dct_species(
    species_dct: dict[str, tuple[str, int, int | None]],
    max_multiplicity: int = 4,
    orca_path: str | Path | None = None,
    directory: str | Path | None = None,
    method: Literal[available_methods] = "G2-MP2-SVP",
    force: bool = False,
    reduce_coordinate_precision: bool = True,
    njobs: int = 1,
    disable_progress_bar: bool = False,
) -> dict[str, Species]:
    """Create Species instances from a dictionary, and calculate their ground
    state energies.

    Args:
        species_dct (dict[str, tuple[str, int, int | None]): dictionary of all species.
            Keys are names of species, and values are a tuple of
            (smiles, charge, multiplicity). If multiplicity is None,
            multiplicities up to max_multiplicity are attempted.
        max_multiplicity (int): maximum multiplicity to try. Default: 4
        orca_path (str | Path | None): path to ORCA executable.
            If None, simply execute "orca". Default: None
        directory (str | Path | None): directory to calculate in. Default: None
        method (Literal[available_methods]): method to use. Default: "G2-MP2-SVP"
        force (bool): whether to do the calculation, even if it was attempted
            previously. Default: False
        reduce_coordinate_precision (bool): whether to reduce the precision of
            coordinates in the generated xyz file. This can help with ORCA inferring
            incorrect symmetries. Default: True
        njobs (int): Number of calculations to run at the same time. Default: 1
        disable_progress_bar (bool): whether to disable the progress bar.
            Default: False

    Returns:
        dict[str, Species]: ground state calculated Species

    """
    # TODO: Maybe move this first bit of instance creation
    # into its own function, "create_species_instances_from_dct"?
    all_species_lst = []
    for spec, (smiles, charge, multiplicity) in species_dct.items():
        if multiplicity is None:
            possibilities = get_possible_multiplicities(
                spec, smiles, charge=charge, max_multiplicity=max_multiplicity
            )
        else:
            possibilities = [
                Species(spec, smiles, charge=charge, multiplicity=multiplicity)
            ]
        all_species_lst.extend(possibilities)

    calculated_species = list(
        calculate_species(
            all_species_lst,
            orca_path=orca_path,
            directory=directory,
            method=method,
            force=force,
            reduce_coordinate_precision=reduce_coordinate_precision,
            njobs=njobs,
            disable_progress_bar=disable_progress_bar,
        ).values()
    )

    return {
        spec: get_ground_state_species(calculated_species, spec) for spec in species_dct
    }


def calculate_reference_species(
    reference_atoms: list[str],
    max_multiplicity: int = 4,
    orca_path: str | Path | None = None,
    directory: str | Path | None = None,
    method: Literal[available_methods] = "G2-MP2-SVP",
    force: bool = False,
    njobs: int = 1,
    disable_progress_bar: bool = False,
) -> dict[str, Species]:
    """Get the reference species in the electronic ground states.

    Args:
        reference_atoms (list[str]): list of atoms to calculate reference energies for
        max_multiplicity (int): maximum multiplicity to try. Default: 4
        orca_path (str | Path | None): path to ORCA executable.
            If None, simply execute "orca". Default: None
        directory (str | Path | None): directory to calculate in. Default: None
        method (Literal[available_methods]): method to use. Default: "G2-MP2-SVP"
        force (bool): whether to do the calculation, even if it was attempted
            previously. Default: False
        reduce_coordinate_precision (bool): whether to reduce the precision of
            coordinates in the generated xyz file. This can help with ORCA inferring
            incorrect symmetries. Default: True
        njobs (int): Number of calculations to run at the same time. Default: 1
        disable_progress_bar (bool): whether to disable the progress bar.
            Default: False

    Return:
        ground_species (dict[str, Species]): ground state reference atoms

    """
    species_dct = {}
    for reference_atom in reference_atoms:
        if reference_atom in atom_ground_state_multiplicities:
            gs_multiplicity = atom_ground_state_multiplicities[reference_atom]
        else:
            gs_multiplicity = None
        species_dct[reference_atom] = (f"[{reference_atom}]", 0, gs_multiplicity)
    return calculate_dct_species(
        species_dct,
        max_multiplicity=max_multiplicity,
        orca_path=orca_path,
        directory=directory,
        method=method,
        force=force,
        njobs=njobs,
        disable_progress_bar=disable_progress_bar,
    )


def get_elements_in_species(species: list[Species]) -> list[str]:
    """Determine the unique elements in the entire species list.

    Args:
        species (list[Species]): list of species

    Returns:
        list[str]: list of all constituent elemnents

    """
    total_set = set()
    for spec in species:
        total_set.update(spec.elements)
    return list(total_set)
