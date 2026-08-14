"""decomb: audited sinusoidal-line notching for continuous EEG.

Built for the step after gradient and pulse correction in simultaneous EEG-fMRI, where
what survives both can include periodic sources not locked to either correction.

The stages run in the order ``decomb --help`` lists them. ``diagnose`` characterises the
line structure, ``apply`` writes an evidence-bounded FIR-notched BIDS derivative,
``verify`` reproduces the immutable plan from disk, and ``psd`` visualises the result.

Every parameter comes from one configuration file; see :mod:`decomb.config`.
"""

__version__ = "0.2.0"

__all__ = ["__version__"]
