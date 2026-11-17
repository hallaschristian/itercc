"""Iterative closed-shell coupled cluster through CCSDTQ, EOM-CCSDT excitation
energies and CCSDT dipole response, on compiled CPU kernels or CUDA."""

__version__ = "0.8.0"

from .hamiltonian import Hamiltonian
from .restricted import RestrictedCC
from .quadruples import CCSDTQState, RestrictedCCSDTQ
from .solver import Iteration, Result, TensorCC
from .eom import EOMSinglet, EOMResult
from .eom_spin_quadruples import TripletEOMCCSDTQ
from .response import DipoleResponse, Polarizability, ResponseCheckpoint
from .spin_orbit import SpinOrbitResult, spin_orbit_from_components, spin_orbit_state_interaction

__all__ = ["Hamiltonian", "Iteration", "RestrictedCC", "Result", "TensorCC", "EOMSinglet",
           "EOMResult", "TripletEOMCCSDTQ", "DipoleResponse", "Polarizability", "RestrictedCCSDTQ", "CCSDTQState",
           "ResponseCheckpoint", "SpinOrbitResult", "spin_orbit_from_components", "spin_orbit_state_interaction"]

