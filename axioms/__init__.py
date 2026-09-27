"""OProof user axiom files.

Place one OProof axiom per ordinary Python module here (each exporting an
``AXIOMS`` list).  They are loaded automatically when ``opreof`` is imported
and may be added or removed at run time through the api interface.

Contents:

* ``foundational.py`` -- the two seed rules (demonstrate the load/add/remove
  cycle).
* ``arithmetic.py``   -- modular two's-complement structure rewrites
  (cancellation, negation, sign flips).
* ``pow2.py``         -- division/remainder by a literal power of two as
  shifts / masks.
* ``logic.py``        -- BOOL identities and bitwise absorption laws.
"""
