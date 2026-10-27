"""The uncertainty of a computed decision, shared by every precision script.

Closed-shell oxides take the calorimetric scatter, 22.5 meV per atom, which the calibration supports (fitted 22, 95%
interval 15 to 37, on the closed-shell decisions of precision_calibration.py). A decision whose target holds an
open-shell 3d or 4f cation takes 43 meV per atom, the lowest value the open-shell compiled decisions allow (fitted 93,
interval 43 upward), so its probabilities are upper bounds. Open shell is judged from the most likely oxidation states,
so d0 cations such as V(V) and Cr(VI) count as closed.
"""
from functools import lru_cache

from pymatgen.core import Composition, Element

SIGMA_CLOSED = 22.5
SIGMA_OPEN = 43.0
D_ELECTRONS_NEUTRAL = {"Sc": 3, "Ti": 4, "V": 5, "Cr": 6, "Mn": 7, "Fe": 8, "Co": 9, "Ni": 10, "Cu": 11, "Zn": 12}
OPEN_CANDIDATES = {"V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
                   "Er", "Tm", "Yb"}


def _open(element, oxidation):
    if element in D_ELECTRONS_NEUTRAL:
        return 0 < D_ELECTRONS_NEUTRAL[element] - oxidation < 10
    el = Element(element)
    if el.is_lanthanoid:
        return 0 < el.Z - 57 - (oxidation - 3) < 14  # 4f count: La(III) 0, Lu(III) 14, Ce(IV) 0
    return False


@lru_cache(maxsize=None)
def open_shell(formula):
    """True when the most likely oxidation states leave a partly filled 3d or 4f shell."""
    comp = Composition(formula)
    if not {e.symbol for e in comp.elements} & OPEN_CANDIDATES:
        return False
    try:
        guesses = comp.oxi_state_guesses(max_sites=-50)
    except Exception:
        guesses = ()
    if not guesses:
        return True  # an element that can be open-shell and no charge-balanced assignment: take the wider sigma
    return any(_open(el, ox) for el, ox in guesses[0].items())


def sigma_for(formula, closed=SIGMA_CLOSED):
    """sigma in meV per atom for a decision about this target; `closed` sets the closed-shell value for sensitivity."""
    return max(SIGMA_OPEN, closed) if open_shell(formula) else closed
