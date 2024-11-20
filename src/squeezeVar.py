import numpy as np

from fit_f_dist import fit_f_dist

def squeeze_var(var, df, covariate=None, robust=False, winsor_tail_p=(0.05, 0.1), legacy=None):
    df = np.asarray(df)
    n = len(var)
    #print(df)
    #print(type(df))


    # No observations
    if n == 0:
        raise ValueError("var is empty")

    # The code will run for n=2 but there is no theoretical advantage from empirical Bayes with less than 3 observations
    if n < 3:
        return {'var_post': var, 'var_prior': var, 'df_prior': 0}

    # Debugging: Print shapes and types
    #print("var shape:", len(var), "type:", type(var))
    #print("df shape:", len(df), "type:", type(df))

    # When df==0, guard against missing or infinite values in var
    if len(df) > 1:
        # Ensure df is used correctly for indexing
        if np.any(df == 0):
            var[df == 0] = 0  # This assumes df is being used to index var

    # Choose legacy or new method depending on whether df are unequal
    if legacy is None:
        dfp = df[df > 0]
        legacy = np.all(np.min(dfp) == np.max(dfp))

    # Estimate hyperparameters
    if legacy:
        if robust:
            fit = fit_f_dist_robustly(var, df1=df, covariate=covariate, winsor_tail_p=winsor_tail_p)
            df_prior = fit['df2_shrunk']
        else:
            fit = fit_f_dist(var, df1=df, covariate=covariate)
            df_prior = fit['df2']
    else:
        fit = fit_f_dist_unequal_df1(var, df1=df, covariate=covariate, robust=robust)
        df_prior = fit['df2_shrunk']
        if df_prior is None:
            df_prior = fit['df2']

    if np.any(np.isnan(df_prior)):
        raise ValueError("Could not estimate prior df")

    # Posterior variances
    var_post = _squeeze_var(var=var, df=df, var_prior=fit['scale'], df_prior=df_prior)

    return {'df_prior': df_prior, 'var_prior': fit['scale'], 'var_post': var_post}

def _squeeze_var(var, df, var_prior, df_prior):
    # Squeeze posterior variances given hyperparameters
    var_prior = np.asarray(var_prior) if np.ndim(var_prior) > 0 else np.array([var_prior])


    m = np.max(df_prior)
    if np.isfinite(m):
        return (df * var + df_prior * var_prior) / (df + df_prior)

    # Set var_post to var_prior of length n
    n = len(var)
    if len(var_prior) == n:
        var_post = var_prior
    else:
        var_post = np.resize(var_prior, n)

    # Check if df_prior all Inf
    m = np.min(df_prior)
    if m > 1e100:
        return var_post

    # Only some df_prior are finite
    i = np.isfinite(df_prior)
    if len(df) > 1:
        df = df[i]
    df_prior = df_prior[i]
    var_post[i] = (df * var[i] + df_prior * var_post[i]) / (df + df_prior)

    return var_post