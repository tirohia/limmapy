import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve
from statsmodels.regression.rolling import RollingOLS
from statsmodels.regression.linear_model import OLS
from scipy.linalg import qr, inv


def lm_fit(object, design=None, ndups=1, spacing=None, block=None, correlation=None, weights=None, method="ls", **kwargs):
    # Extract components from object
    y = get_eawp(object)

    if y['exprs'].shape[0] == 0:
        raise ValueError("Expression matrix has zero rows")

    # Check design matrix
    if design is None:
        design = pd.DataFrame(np.ones((y['exprs'].shape[0], 1)))
    else:
        design = pd.DataFrame(design)
        if design.shape[0] != y['exprs'].shape[0]:
            raise ValueError("Design must have matching dimensions with expression data")

    # Check weights
    if weights is None:
        weights = y.get('weights', None)

    # Ensure ndups has a default value
    if ndups is None:
        ndups = 1

    # Dispatch fitting algorithms
    if method == "robust":
        fit_results = mrlm(y['exprs'], design, ndups, spacing, weights, **kwargs)
    else:
        if ndups < 2 and block is None:
            fit_results = lm_series(y['exprs'], design, ndups, spacing, weights)
        else:
            if correlation is None:
                raise ValueError("The correlation must be set")
            fit_results = gls_series(y['exprs'], design, ndups, spacing, block, correlation, weights, **kwargs)

    # Create a results dictionary
    print(fit_results["cov_coefficients"])
    results = {
        'coefficients': fit_results["coeffiecients"], 
        'stdev_unscaled': fit_results["stdev_unscaled"], 
        'genes': y['probes'],
        'Amean': y['Amean'],
        'method': method,
        'design': design,
        'sigma': fit_results["sigma"],
        'df_residual': fit_results["df_residual"],
        'cov_coefficients': fit_results["cov_coefficients"]
    }
    
    
    return results

def lm_series(M, design=None, ndups=1, spacing=1, weights=None):
    M = pd.DataFrame(M)  # Ensure M is a DataFrame
    narrays = M.shape[0]  # Number of samples
    n_genes = M.shape[1]   # Number of genes
    n_coefficients = design.shape[1]

    effects = np.empty(M.shape)  # Effects matrix, same shape as M
    ranks = []  # To store rank for each model
    rank = np.linalg.matrix_rank(design)  # Rank of the design matrix
    sigma = np.full(M.shape[1], np.nan)  # Sigma for each gene

    Q, R = qr(design.values, mode='economic')  # Orthogonal (Q) and upper triangular (R)


    print(f"Number of samples {narrays}, number of genes: {n_genes}, number of coefficients {n_coefficients}")
    
    #print(M.iloc[0:5, 50])

    # Ensure design is aligned with M
    if design is not None:
        design = design.reindex(M.index)  # Align design to M's index

    #stdev_unscaled = pd.DataFrame(np.nan, index=[f"Gene_{i}" for i in range(n_genes)], columns=[f"Coef_{i}" for i in range(n_coeffiecients)])
    results = []  # Store results for each gene
    df_residuals = []  # To store residual degrees of freedom

    residual_stds = {}
    sigma = {}  # New dict for sigma values
    coefficients = pd.DataFrame(columns=design.columns)
    stdev_unscaled = pd.DataFrame(columns=design.columns)  # New DataFrame for stdev
    df_residuals = pd.DataFrame(columns=M.columns)  # New DataFrame for stdev
    cov_coefficients = pd.DataFrame(columns=design.columns)  # New DataFrame for covariance coefficients

    for i, gene in enumerate(M.columns):
        y = M[gene].values  # Expression values for the gene
        model = OLS(y, design).fit()

        df = model.params
        df.name = gene
        df = df.to_frame().T

        coefficients = pd.concat([coefficients, df])
        residual_stds[gene] = np.std(model.resid)

        est = np.where(np.abs(model.model.pinv_wexog).sum(axis=0) > 0)[0]
        stdev = np.sqrt(np.diag(model.normalized_cov_params))
        stdev = pd.Series(stdev, index=design.columns)
        stdev.name = gene
        stdev = stdev.to_frame().T
        
        stdev_unscaled = pd.concat([stdev_unscaled, stdev])

        sigma[gene] = np.sqrt(np.sum(model.resid**2) / model.df_resid)

        df_residuals.loc[0, gene] = model.df_resid  # Store the degrees of freedom

        # Calculate covariance coefficients
        # Get the QR matrix from the model
        qr_matrix = model.model.pinv_wexog.T @ model.model.pinv_wexog  # This is equivalent to fit$qr$qr in R
        #print(type(qr_matrix))
        #print(qr_matrix)
        #print(R)
        #print(Q)
        #print(qr_matrix)
        
        #cov_matrix = inv(L) @ inv(L).T  # Inverse of the Cholesky factor gives the covariance matrix

        # Store the covariance coefficients
        
    qr_result = np.linalg.qr(design, mode="reduced")
    qr_dict = {name: val for name, val in zip(['Q', 'R'], qr_result)}
    qr_matrix = qr_dict["R"]

    #print(type(qr_matrix))
    #qr_matrix = dict(qr_matrix)

    #qr_matrix = qr_matrix["R"]
    #cov_coefficients = pd.concat([cov_coefficients, pd.DataFrame(cov_matrix, index=design.columns, columns=design.columns).add_prefix(f"{gene}_")])
    
    residual_stds= pd.Series(residual_stds)
    sigma = pd.Series(sigma)
    df_residuals = df_residuals.iloc[0,]
    
    L = cholesky(residual_stds, lower=True)  # Cholesky decomposition
    # print(df_residuals)
    #print(sigma)
    # print(coefficients)
    #print(pd.Series(residual_stds))
    print(stdev_unscaled)
    #coeffiecients = stdev_unscaled
    results = {
        "stdev_unscaled" : stdev_unscaled,
        "sigma" : sigma,
        "df_residual" : df_residuals,
        "coeffiecients": coefficients,
        "cov_coefficients": cov_coefficients
    }

    return results  # Return the list of fit results

def mrlm(M, design=None, ndups=1, spacing=1, weights=None, **kwargs):
    # Robustly fit linear model for each gene to a series of arrays
    M = pd.DataFrame(M)
    narrays = M.shape[1]
    if design is None:
        design = pd.DataFrame(np.ones((narrays, 1)))
    design = pd.DataFrame(design)

    # Check weights
    if weights is not None:
        weights = pd.Series(weights)
        weights[weights <= 0] = np.nan
        M[~weights.notna()] = np.nan

    # Iterate through genes
    ngenes = M.shape[0]
    stdev_unscaled = pd.DataFrame(np.nan, index=np.arange(ngenes), columns=design.shape[1])
    beta = pd.DataFrame(np.nan, index=np.arange(ngenes), columns=design.shape[1])

    for i in range(ngenes):
        y = M.iloc[i, :]
        obs = np.isfinite(y)
        X = design.loc[obs, :]
        y = y[obs]
        if len(y) > design.shape[1]:
            out = RollingOLS(y, X).fit()
            beta.iloc[i, :] = out.params
            stdev_unscaled.iloc[i, :] = np.sqrt(np.diag(np.linalg.inv(out.normalized_cov_params)))

    return {
        'coefficients': beta,
        'stdev_unscaled': stdev_unscaled
    }

def gls_series(M, design=None, ndups=2, spacing=1, block=None, correlation=None, weights=None, **kwargs):
    # Fit linear model for each gene to a series of microarrays
    M = pd.DataFrame(M)
    ngenes = M.shape[0]
    narrays = M.shape[1]

    if design is None:
        design = pd.DataFrame(np.ones((narrays, 1)))
    design = pd.DataFrame(design)

    # Check correlation
    if correlation is None:
        correlation = duplicate_correlation(M, design, ndups, spacing, block, weights, **kwargs)

    # Check weights
    if weights is not None:
        weights = pd.Series(weights)
        weights[np.isnan(weights)] = 0
        M[weights < 1e-15] = np.nan
        weights[weights < 1e-15] = np.nan

    # Unwrap duplicates and make correlation matrix
    if block is None:
        if ndups < 2:
            print("Warning: No duplicates (ndups<2)")
            ndups = 1
            correlation = 0
        cormatrix = np.eye(narrays) * correlation
        M = unwrap_dups(M, ndups, spacing)
        if weights is not None:
            weights = unwrap_dups(weights, ndups, spacing)
        design = pd.DataFrame(np.tile(design.values, (1, ndups)))
    else:
        ndups = spacing = 1
        block = pd.Series(block)
        if len(block) != narrays:
            raise ValueError("Length of block does not match number of arrays")
        unique_blocks = np.unique(block)
        nblocks = len(unique_blocks)
        Z = (block[:, None] == unique_blocks).astype(float)
        cormatrix = Z @ (correlation * Z.T)

    # Fit the model
    stdev_unscaled = pd.DataFrame(np.nan, index=np.arange(ngenes), columns=design.shape[1])

    # Fast computation if weights and missing values are absent
    no_probe_wts = np.all(np.isfinite(M)) and (weights is None or weights is not None)
    if no_probe_wts:
        V = cormatrix
        if weights is not None:
            wrs = 1 / np.sqrt(weights[0, :])
            V = wrs[:, None] * wrs[None, :] * V
        cholV = cholesky(V, lower=True)
        y = solve(cholV, M.T, lower=True)
        X = solve(cholV, design, lower=True)
        fit = OLS(y, X).fit()
        return fit

    # Weights or missing values are present, iterate over probes
    beta = pd.DataFrame(np.nan, index=np.arange(ngenes), columns=design.shape[1])
    for i in range(ngenes):
        y = M.iloc[i, :]
        obs = np.isfinite(y)
        y = y[obs]
        if len(y) > 0:
            X = design.loc[obs, :]
            V = cormatrix.loc[obs, obs]
            if weights is not None:
                wrs = 1 / np.sqrt(weights[i, obs])
                V = wrs[:, None] * wrs[None, :] * V
            cholV = cholesky(V, lower=True)
            y = solve(cholV, y, lower=True)
            X = solve(cholV, X, lower=True)
            out = OLS(y, X).fit()
            beta.iloc[i, :] = out.params

    return {
        'coefficients': beta,
        'stdev_unscaled': stdev_unscaled
    }

def get_eawp(object):
    # Extract information needed for linear modeling
    if object is None:
        raise ValueError("Data object is NULL")

    if not isinstance(object, pd.DataFrame):
        raise ValueError("Expected input to be a DataFrame")

    y = {}
    
    # Keep the DataFrame structure to retain the index
    y['exprs'] = object  # Retain the DataFrame instead of converting to values
    y['Amean'] = object.mean(axis=1)
    y['probes'] = object.index.values  # Use DataFrame index as probes

    return y

# Additional helper functions like unwrap_dups, duplicate_correlation, etc. would need to be defined.