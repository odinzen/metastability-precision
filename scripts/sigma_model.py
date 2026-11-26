"""The uncertainty of a computed decision, shared by every precision script: three regimes.

Compound decisions (a target against other compounds or its binary oxides, no open shell) take the calorimetric
scatter, 22.5 meV per atom, fixed before any outcome was seen. Against outcomes audited with primary calorimetry the
closed-shell compiled decisions want less (fitted 13.6, 95% interval 8 to 26; calibration_audit.py), so 22.5 is
conservative for them. Decisions about a target with an open-shell cation take 34 meV per atom, the lowest value the
audited open-shell decisions allow (fitted 66, interval 34 to 298), so their probabilities are upper bounds. Polymorph
decisions, between structures of one composition, carry a systematic GGA error that favours the less dense structure
rather than a random scatter; they get no Gaussian probability and are reported apart (regime "polymorph").

Open shell is judged from the most likely oxidation states, so d0 cations such as V(V) and Cr(VI) count as closed, and
as the calculation sees it: the Materials Project (and GNoME, with the same settings) puts the 4f electrons of most
lanthanides in the core (Pr_3, Nd_3, Sm_3, Tb_3 to Lu_3, Yb_2), so only Ce, Eu and Gd can have an open 4f shell.
"""
from functools import lru_cache

from pymatgen.core import Composition, Element

SIGMA_CLOSED = 22.5
SIGMA_OPEN = 34.0
D_ELECTRONS_NEUTRAL = {"Sc": 3, "Ti": 4, "V": 5, "Cr": 6, "Mn": 7, "Fe": 8, "Co": 9, "Ni": 10, "Cu": 11, "Zn": 12}
F_IN_VALENCE = {"Ce", "Eu", "Gd"}  # lanthanides whose GGA PAW keeps the 4f electrons in the valence
OPEN_CANDIDATES = {"V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"} | F_IN_VALENCE


def _open(element, oxidation):
    if element in D_ELECTRONS_NEUTRAL:
        return 0 < D_ELECTRONS_NEUTRAL[element] - oxidation < 10
    el = Element(element)
    if element in F_IN_VALENCE:
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


def regime(formula, polymorph=False):
    """'polymorph' for a decision between structures of one composition, else 'open-shell' or 'compound'."""
    if polymorph:
        return "polymorph"
    return "open-shell" if open_shell(formula) else "compound"


def sigma_for(formula, closed=SIGMA_CLOSED, open_=SIGMA_OPEN):
    """sigma in meV per atom for a compound or open-shell decision about this target; `closed` and `open_` set the two
    values for sensitivity runs."""
    return max(open_, closed) if open_shell(formula) else closed
