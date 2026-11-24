"""Iterative closed-shell coupled cluster through CCSDTQ, EOM-CCSDT excitation
energies and CCSDT and CCSDTQ dipole response, on compiled CPU kernels or CUDA."""

__version__ = "0.8.0"

from .hamiltonian import Hamiltonian
from .restricted import RestrictedCC
from .quadruples import CCSDTQState, RestrictedCCSDTQ
from .solver import Iteration, Result, TensorCC
from .eom import EOMSinglet, EOMResult
from .eom_spin_quadruples import TripletEOMCCSDTQ
from .response import DipoleResponse, Polarizability, ResponseCheckpoint
from .response_ccsdtq import QuadruplesDipoleResponse
from .spin_orbit import SpinOrbitResult, spin_orbit_from_components, spin_orbit_state_interaction
from .active_space import fold_closed_shell, select_active_orbitals
from .finite_field import FiniteFieldPolarizability, finite_field_polarizability

__all__ = ["Hamiltonian", "Iteration", "RestrictedCC", "Result", "TensorCC", "EOMSinglet",
           "EOMResult", "TripletEOMCCSDTQ", "DipoleResponse", "QuadruplesDipoleResponse", "Polarizability",
           "RestrictedCCSDTQ", "CCSDTQState",
           "ResponseCheckpoint", "SpinOrbitResult", "spin_orbit_from_components", "spin_orbit_state_interaction",
           "fold_closed_shell", "select_active_orbitals", "FiniteFieldPolarizability",
           "finite_field_polarizability"]

