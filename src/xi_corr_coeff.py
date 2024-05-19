"""Xi correlation coefficient array calculation for both ties and no ties data.


Original Python implementation: https://github.com/jlbloesch/miscellaneous.


From https://www.linkedin.com/pulse/correlation-coefficient-xi-justin-bloesch-zxukc#:~:text=Xi%20ξ%20is%20a
%20relatively,makes%20it%20robust%20to%20outliers.: Xi ξ is a relatively new correlation coefficient that is more
effective than the classical measures like Pearson’s r, Spearman’s ρ, or Kendall’s 𝜏 for detecting associations that
are not monotonic. Xi is a function of rank, which makes it robust to outliers. It is easy to interpret as a measure
of dependence of X and Y. The limits of Xi ranges for 0 independent to 1 dependent. It has a simple asymptotic theory
under the hypothesis of independence which is valid for sample sizes as small as 20. It can be applied to categorical
variables by converting them to integer-valued variables. Xi is more powerful than other tests for detecting
oscillatory signals. Against these and other advantages, Xi has only one disadvantage, it is less powerful than other
tests of independence when the signal is non-oscillatory for a small sample.


Related References:
[1] Sourav Chatterjee (2021) A New Coefficient of Correlation, Journal of the American Statistical Association, 116:536,
2009-2022, DOI: 10.1080/01621459.2020.1758115.
"""


__all__ = ["xi_cor_coeff"]
__author__ = "Yuen Shing Yan Hindy"
__version__ = "1.0.0"


from typing import Any
import numpy as np
from scipy.stats import rankdata, norm
from src._util import _is_array_like


def xi_cor_coeff(x: Any, y: Any, ties: bool = False) -> (float, float):
    """
    Compute and returns Xi correlation coefficient and p-values using given variables `x` and `y`.

    Parameters
    ----------
    x : Any
        Array-like variables.
    y : Any
        Array-like variables.
    ties : bool
        Ties determines which formula to compute Xi correlation coefficient is used.

    Returns
    -------
    (float, float)
        Xi correlation coefficient and p-value.
    """
    if not _is_array_like(x):
        raise ValueError("Argument `x` is not array-like.")

    if not _is_array_like(y):
        raise ValueError("Argument `y` is not array-like.")

    if len(x) < 2:
        raise ValueError("Length argument `x` must greater or equals to 2.")

    if len(y) < 2:
        raise ValueError("Length argument `y` must greater or equals to 2.")

    if len(y) != len(x):
        raise IndexError(f'`x` and `y` variables array size mismatch: {len(x)}, {len(y)}')

    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    y = y[np.argsort(x)]
    r = rankdata(y, method='ordinal')
    nominator = np.sum(np.abs(np.diff(r)))
    n = len(x)

    if ties:
        r = rankdata(y, method='max')
        denominator = 2 * np.sum(r * (n - 1))
        nominator *= n
    else:
        denominator = np.power(n, 2) - 1
        nominator *= 3

    xi = 1 - nominator / denominator
    p_value = norm.sf(xi, scale=2 / 5 / np.sqrt(n))

    return xi, p_value
