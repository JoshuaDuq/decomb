"""decomb: audited harmonic-comb notching for continuous EEG.

Built for the step after gradient and pulse correction in simultaneous EEG-fMRI, where
what survives both is whatever periodic source is locked to neither -- typically
mechanical, typically a comb of harmonics on one fundamental.

The stages run in the order ``decomb --help`` lists them. ``diagnose`` characterises the
line structure, ``apply`` writes an evidence-bounded FIR-notched BIDS derivative,
``verify`` re-measures the immutable plan from disk, and ``psd`` visualises the result.

Every parameter comes from one configuration file; see :mod:`decomb.config`.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
