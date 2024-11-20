import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t, f

from squeezeVar import squeeze_var

def is_fullrank(x):
    """
    Check whether a numeric matrix has full column rank.
    
    Parameters:
    x (array-like): Input matrix.
    
    Returns:
    bool: True if the matrix has full column rank, False otherwise.
    """
    x = np.asarray(x)  # Convert to a numpy array
    e = np.linalg.eigvalsh(np.dot(x.T, x))  # Compute eigenvalues of x^T * x
    return e[0] > 0 and abs(e[-1] / e[0]) > 1e-13  # Check the conditions


def eBayes(fit, proportion=0.01, stdev_coef_lim=(0.1, 4), trend=False, robust=False, winsor_tail_p=(0.05, 0.1)):
    # Check if fit is a valid object
    if not isinstance(fit, dict):
        raise ValueError("fit is not a valid MArrayLM object")
    
    if trend and 'Amean' not in fit:
        raise ValueError("Need Amean component in fit to estimate trend")
    
    eb = _ebayes(fit, proportion, stdev_coef_lim, trend, robust, winsor_tail_p)
    
    #for key in eb.keys():
    #    print(key)
    # Update fit with results from eb
    fit['df_prior'] = eb['df_prior']
    fit['s2_prior'] = eb['s2_prior']
    fit['var_prior'] = eb['var_prior']
    fit['proportion'] = proportion
    fit['s2_post'] = eb['s2_post']
    fit['t'] = eb['t']
    fit['df_total'] = eb['df_total']
    fit['p_value'] = eb['p_value']
    fit['lods'] = eb['lods']

    print(f"Fit: {type(fit)}")
    #for key in fit.keys():
    #    print(key)

    if 'design' in fit and is_fullrank(fit['design']):
        F_stat = classify_tests_F(fit, fstat_only=True)
        fit['F'] = F_stat
        #print("F stat:")
        #print(fit['F'])
        df1 = F_stat['df1']
        #print("df1: ")
        #print(df1
        df2 = F_stat['df2']
        #print("df2: ")
        #print(df2)
        fit['F_p_value'] = stats.f.sf(fit['F']['fstat'], df1, df2)
    
    return fit

def _ebayes(fit, proportion=0.01, stdev_coef_lim=(0.1, 4), trend=False, robust=False, winsor_tail_p=(0.05, 0.1)):
    coefficients = np.asarray(fit['coefficients'])
    stdev_unscaled = np.asarray(fit['stdev_unscaled'])
    sigma = fit['sigma']
    df_residual = np.asarray(fit['df_residual'])
    
    if any(x is None for x in [coefficients, stdev_unscaled, sigma, df_residual]):
        raise ValueError("No data, or argument is not a valid lmFit object")
    
    #print(df_residual)
    if np.max(df_residual) == 0:
        raise ValueError("No residual degrees of freedom in linear model fits")
    
    if not np.any(np.isfinite(sigma)):
        raise ValueError("No finite residual standard deviations")
    
    # Handle trend
    if isinstance(trend, bool):
        covariate = fit['Amean'] if trend else None
        if trend and covariate is None:
            raise ValueError("Need Amean component in fit to estimate trend")
    elif isinstance(trend, (list, np.ndarray)):
        if len(sigma) != len(trend):
            raise ValueError("If trend is numeric then it should have length equal to the number of genes")
        covariate = trend
    else:
        raise ValueError("trend should be either a logical scale or a numeric vector")
    
    # Moderated t-statistic
    out = squeeze_var([s ** 2 for s in sigma], df_residual, covariate=covariate, robust=robust, winsor_tail_p=winsor_tail_p)
    out['s2_prior'] = out['var_prior']
    out['s2_post'] = out['var_post']
    out['var_prior'] = out['var_post'] = None
    #print(coefficients)
    #print(stdev_unscaled)
    #print(out["s2_post"])
    # coeffieicnts and stdev_unscaled are nx2, out['s2_post] is [n], which doesn't work. 
    # option 1 which is this, is to select only column one from coeffiecients and stdev_unscaled, which might not be the 
    # correct course of action. If it's not, option 2 might be to return to this and make out['t'] into [50,2] 
    # by doing coefficients[0]/stdev_unscaled[0] / out['s2_post'] and then repeating for column 1. 

    out['t'] = coefficients / stdev_unscaled / np.sqrt(out['s2_prior'])  # This will work for any number of columns
    out['t'] = pd.DataFrame(out['t'], columns=['intercept', 'predictor'])  # Adjust column names as needed
    #print("out['t']:")
    #print(out['t'])
    #out['t'] = coefficients[: ,0] / stdev_unscaled[: ,0] / np.sqrt(out['s2_post'])
    
    #print(type(df_residual))
    #print(type(out['df_prior']))
    #print(df_residual)
    #print(out['s2_prior'])


    df_total = df_residual + out['df_prior']
    df_pooled = np.nansum(df_residual)
    out['df_total'] = np.minimum(df_total, df_pooled)

    #print(f"t test result I think: {out['t']}")
    #print(f"total: {out['df_total']}")

    #print(np.abs(out['t']))
    #out['p_value'] = 2 * stats.t.sf(np.abs(out['t']), df=out['df_total'])
    #out['p_value'] = 2 * stats.t.sf(np.abs(out['t']), df=df_total[:, None])  # Broadcasting df_total to match shape
    #p_values = out['t'].apply(lambda row: 2 * t.cdf(-np.abs(row), df=out['df_total']), axis=1)
    #print(zip(*p_values))
    # Compute p-values for each column in out["t"]

    p_values = out["t"].apply(
        lambda row: [2 * t.cdf(-np.abs(row[col]), df) for col, df in zip(row.index, [out["df_total"][row.name]] * len(row))], axis=1
    )

    # Create a new DataFrame for p-values and assign it to out["pvalue"]
    out["p_value"] = pd.DataFrame(p_values.tolist(), columns=out["t"].columns, index=out["t"].index)

    
    
    #out['p_value'] = 2 * stats.t.sf(np.abs(out['t']), df=df_total)  # Calculate p-values

    # B-statistic
    var_prior_lim = np.square(stdev_coef_lim) / np.median(out['s2_prior'])
    out['var_prior'] = tmixture_matrix(out['t'], stdev_unscaled, out['df_total'], proportion, var_prior_lim)
    
    if np.any(np.isnan(out['var_prior'])):
        out['var_prior'][np.isnan(out['var_prior'])] = 1 / out['s2_prior']
        print("Warning: Estimation of var.prior failed - set to default value")
    
    r = np.outer(np.ones(len(out['t'])), out['var_prior'])
    r = (stdev_unscaled**2 + r) / stdev_unscaled**2
    t2 = out['t']**2
    Infdf = out['df_prior'] > 10**6
    #print(f"Infdf: {type(Infdf)}")
    #print(Infdf)
    
    #print(out['t'].shape)
    #print(out['t'].columns)
    #print(f"t2: {type(t2)} ")
    #print(t2)
    if np.any(Infdf):
        kernel = t2 * (1 - 1 / r) / 2
        if not Infdf:
            t2_f = t2[~Infdf, :] 
            #t2_f = t2[~Infdf]
            #print("t2_f: ")
            #print(t2_f)
            r_f = r[~Infdf]
            df_total_f = out['df_total'][~Infdf]
            kernel[~Infdf] = (1 + df_total_f) / 2 * np.log((t2_f + df_total_f) / (t2_f / r_f + df_total_f))
    else:
        kernel = (1 + out['df_total']) / 2 * np.log((t2 + out['df_total']) / (t2 / r + out['df_total']))
    
    out['lods'] = np.log(proportion / (1 - proportion)) - np.log(r) / 2 + kernel
    return out

def tmixture_matrix(tstat, stdev_unscaled, df, proportion, v0_lim=None):
    tstat = np.asarray(tstat)
    stdev_unscaled = np.asarray(stdev_unscaled)
    #print(tstat)
    #print(stdev_unscaled)
    
    if tstat.shape != stdev_unscaled.shape:
        raise ValueError("Dims of tstat and stdev.unscaled don't match")
    
    if v0_lim is not None and len(v0_lim) != 2:
        raise ValueError("v0.lim must have length 2")
    
    ncoef = tstat.shape[1]
    v0 = np.zeros(ncoef)
    
    for j in range(ncoef):
        v0[j] = tmixture_vector(tstat[:, j], stdev_unscaled[:, j], df, proportion, v0_lim)
    
    return v0

def tmixture_vector(tstat, stdev_unscaled, df, proportion, v0_lim=None):
    # Remove missing values
    mask = ~np.isnan(tstat)
    tstat = tstat[mask]
    stdev_unscaled = stdev_unscaled[mask]
    df = df[mask]

    ngenes = len(tstat)
    ntarget = int(np.ceil(proportion / 2 * ngenes))
    if ntarget < 1:
        return np.nan
    




def classify_tests_F(object, cor_matrix=None, df=np.inf, p_value=0.01, fstat_only=False):
    # Use F-tests to classify vectors of t-test statistics into outcomes
    #for key in object.keys():
    #    print(key)
    print(f"Object type: {type(object)}")
    print(object["cov_coefficients"])
    if isinstance(object, dict):
        if 't' not in object:
            raise ValueError("tstat cannot be extracted from object")
        
        if cor_matrix is None and object.get('cov_coefficients') is not None:
            print("coeffieinct covariants found")
            # Check for and adjust any coefficient variances exactly zero
            cov_coefficients = object['cov_coefficients']
            print("cov ceoefficients:")
            print(cov_coefficients)
            n = cov_coefficients.shape[0]
            if np.min(np.diag(cov_coefficients)) == 0:
                zero_indices = np.where(np.diag(cov_coefficients) == 0)[0]
                for j in zero_indices:
                    cov_coefficients[j, j] = 1
            cor_matrix = np.cov(cov_coefficients)

        if np.isinf(df) and 'df.prior' in object and 'df.residual' in object:
            df = object['df.prior'] + object['df.residual']
        
        tstat = np.array(object['t'])
    else:
        tstat = np.array(object)

    ngenes, ntests = tstat.shape
    result_dict = {}

    if ntests == 1:
        if fstat_only:
            fstat = tstat ** 2
            result_dict['fstat'] = fstat
            result_dict['df1'] = 1
            result_dict['df2'] = df
            #print("n tests =1. fstat only")
            #for key in result_dict.keys():
            #    print(key)
            return result_dict
        else:
            p = 2 * t.cdf(-np.abs(tstat), df)
            result_dict['test_results'] = np.sign(tstat) * (p < p_value)
            #print("n tests = 1")
            #for key in result_dict.keys():
            #    print(key)
            return result_dict

    print("Correlation Matrix:")
    print(cor_matrix)
    # cor_matrix is estimated correlation matrix of the coefficients
    if cor_matrix is None:
        #r = ntests
        r = ntests
        Q = np.eye(r) / np.sqrt(r)
    else:
        eigenvalues, eigenvectors = linalg.eigh(cor_matrix)
        # Sort in descending order (R's eigen returns in this order)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        # Calculate effective rank r
        r = np.sum(eigenvalues / eigenvalues[0] > 1e-8)
        # Select first r eigenvalues and eigenvectors
        eigenvalues = eigenvalues[:r]
        eigenvectors = eigenvectors[:, :r]
        # Calculate Q matrix
        Q = (eigenvectors * (1/np.sqrt(eigenvalues))) / np.sqrt(r)
        #eigvals, eigvecs = np.linalg.eigh(cor_matrix)
        #r = np.sum(eigvals / eigvals[0] > 1e-8)
        # = eigvecs[:, :r] @ np.diag(1 / np.sqrt(eigvals[:r])) / np.sqrt(r)

    # Return overall moderated F-statistic only
    if fstat_only:
        #print("tstat:")
        #print(tstat)
        #print("Q:")
        #print(Q)
        #fstat = np.sum((tstat @ Q) ** 2, axis=1)
        #intermediate = np.dot(tstat, Q)
        #fstat = np.sum(intermediate ** 2, axis=1)
        intermediate = tstat @ Q 
        #print(intermediate)
        fstat = np.sum(tstat ** 2, axis=1)
    
        #fstat = np.sum((tstat @ Q) ** 2, axis=1, keepdims=True)
        
        #print("fstat internal:")
        #print(fstat)
        result_dict['fstat'] = fstat
        result_dict['df1'] = r
        result_dict['df2'] = df
        #print("ntests >1, fstat only")
        #for key in result_dict.keys():
        #    print(key)
        return result_dict

    # Return TestResults matrix
    qF = f.ppf(1 - p_value, r, df)
    if np.isscalar(qF):
        qF = np.full(ngenes, qF)
    
    result = np.zeros((ngenes, ntests))
    for i in range(ngenes):
        x = tstat[i, :]
        if np.any(np.isnan(x)):
            result[i, :] = np.nan
        else:
            if np.dot(np.dot(x, Q), x) > qF[i]:
                ord_indices = np.argsort(-np.abs(x))
                result[i, ord_indices[0]] = np.sign(x[ord_indices[0]])
                for j in range(1, ntests):
                    bigger = ord_indices[:j]
                    x[bigger] = np.sign(x[bigger]) * np.abs(x[ord_indices[j]])
                    if np.dot(np.dot(x, Q), x) > qF[i]:
                        result[i, ord_indices[j]] = np.sign(x[ord_indices[j]])
                    else:
                        break

    result_dict['test_results'] = result

    return result_dict