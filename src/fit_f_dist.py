import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln, digamma, polygamma

def fit_f_dist(x, df1, covariate=None):
    # Convert x to a NumPy array if it is a list
    x = np.asarray(x)
    
    # Check x
    n = len(x)
    if n == 0:
        return {'scale': np.nan, 'df2': np.nan}
    if n == 1:
        return {'scale': x[0], 'df2': 0}

    # Check df1 because for some stupid reason it's coming through as an array of type object
    df1 = df1.astype(np.float64)
    ok = np.isfinite(df1) & (df1 > 1e-15)
 
    if len(df1) == 1:
        if not ok:
            return {'scale': np.nan, 'df2': np.nan}
        else:
            ok = np.repeat(True, n)
    else:
        if len(df1) != n:
            raise ValueError("x and df1 have different lengths")

    # Check covariate
    if covariate is None:
        splinedf = 1
    else:
        covariate = np.asarray(covariate)  # Convert covariate to NumPy array
        if len(covariate) != n:
            raise ValueError("x and covariate must be of the same length")
        if np.any(np.isnan(covariate)):
            raise ValueError("NA covariate values not allowed")
        isfin = np.isfinite(covariate)
        if not np.all(isfin):
            if np.any(isfin):
                r = np.nanpercentile(covariate[isfin], [0, 100])
                covariate[covariate == -np.inf] = r[0] - 1
                covariate[covariate == np.inf] = r[1] + 1
            else:
                covariate = np.sign(covariate)

    # Remove missing or infinite or negative values and zero degrees of freedom
    ok = ok & np.isfinite(x) & (x > -1e-15)
    nok = np.sum(ok)
    if nok == 1:
        return {'scale': x[ok][0], 'df2': 0}
    notallok = (nok < n)
    if notallok:
        x = x[ok]
        if len(df1) > 1:
            df1 = df1[ok]
        if covariate is not None:
            covariate_notok = covariate[~ok]
            covariate = covariate[ok]

    # Set df for spline trend
    if covariate is not None:
        splinedf = 1 + (nok >= 3) + (nok >= 6) + (nok >= 30)
        splinedf = min(splinedf, len(np.unique(covariate)))
        if splinedf < 2:
            return fit_f_dist(x=x, df1=df1)

    # Avoid exactly zero values
    x = np.maximum(x, 0)
    m = np.median(x)
    if m == 0:
        print("Warning: More than half of residual variances are exactly zero: eBayes unreliable")
        m = 1
    else:
        if np.any(x == 0):
            print("Warning: Zero sample variances detected, have been offset away from zero")
    x = np.maximum(x, 1e-5 * m)

    # Better to work on with log(F)
    z = np.log(x)
    e = z + np.log(stats.gamma(df1 / 2).mean())

    if covariate is None:
        emean = np.mean(e)
        evar = np.sum((e - emean) ** 2) / (nok - 1)
    else:
        # Implement spline fitting if covariate is provided
        # This part requires additional libraries like statsmodels or scipy for spline fitting
        # Placeholder for spline fitting
        raise NotImplementedError("Spline fitting for covariate is not implemented yet.")

    # Estimate scale and df2
    evar = evar - np.mean(polygamma(1, df1 / 2))
    if evar > 0:
        df2 = 2 * trigamma_inverse(evar)
        s20 = np.exp(emean - np.log(stats.gamma(df2 / 2).mean()))
    else:
        df2 = np.inf
        if covariate is None:
            s20 = np.mean(x)
        else:
            s20 = np.exp(emean)

    return {'scale': s20, 'df2': df2}

def trigamma_inverse(x):
    # Solve trigamma(y) = x for y
    if not np.isnumeric(x):
        raise ValueError("Non-numeric argument to mathematical function")
    if len(x) == 0:
        return np.array([])

    # Treat out-of-range values as special cases
    omit = np.isnan(x)
    if np.any(omit):
        y = x.copy()
        if np.any(~omit):
            y[~omit] = trigamma_inverse(x[~omit])
        return y
    omit = (x < 0)
    if np.any(omit):
        y = x.copy()
        y[omit] = np.nan
        print("Warning: NaNs produced")
        if np.any(~omit):
            y[~omit] = trigamma_inverse(x[~omit])
        return y
    omit = (x > 1e7)
    if np.any(omit):
        y = x.copy()
        y[omit] = 1 / np.sqrt(x[omit])
        if np.any(~omit):
            y[~omit] = trigamma_inverse(x[~omit])
        return y
    omit = (x < 1e-6)
    if np.any(omit):
        y = x.copy()
        y[omit] = 1 / x[omit]
        if np.any(~omit):
            y[~omit] = trigamma_inverse(x[~omit])
        return y

    # Newton's method
    y = 0.5 + 1 / x
    iter_count = 0
    while True:
        iter_count += 1
        tri = stats.t.trigamma(y)
        dif = tri * (1 - tri / x) / stats.t.psigamma(y, deriv=2)
        y += dif
        if np.max(-dif / y) < 1e-8:
            break
        if iter_count > 50:
            print("Warning: Iteration limit exceeded")
            break
    return y