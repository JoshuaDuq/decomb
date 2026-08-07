"""decomb: audited removal of narrowband line and comb artifacts from continuous EEG.

Built for the step after gradient and pulse correction in simultaneous EEG-fMRI, where
what survives both is whatever periodic source is locked to neither -- typically
mechanical, typically a comb of harmonics on one fundamental.

The stages run in the order ``decomb --help`` lists them. ``diagnose`` measures which lines
exist, whether they form a comb, and what share of each band they carry; ``benchmark``
checks the removal against criteria stated before the measurement; ``apply`` writes a
cleaned BIDS copy and refuses without a passing benchmark; ``verify`` re-measures what was
written with a detector that had no knowledge of where the targets were.

Every parameter comes from one configuration file; see :mod:`decomb.config`.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
