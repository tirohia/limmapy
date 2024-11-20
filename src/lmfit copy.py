import pandas as pd
import anndata as ad
import yaml

from icecream import ic

with open('refinementConfig.yaml', 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)


# data locations
dataDirectory = config["dataDirectory"]
resultsDirectory = config["resultsDirectory"]
anndataFile = config["datafile"]

def lmFit(data, desgin=None, ndups=1, spacing=1, block=None, weights=None, method="ls"):

    #extract components from Y
    data = getEAWP(data)



def getEAWP(data):
    """""""""""""""""""""""""""
    Extraction of basic information needed for linear modelling from the data object passed in. 
    In the original R code, this appears to test for a number of different types of objects that I 
    don't really care about. 

    What types of objects do I care about? I'ma gunna go with anndatas, and pandas dataframes. 

    It returns, by the look of it, one of those stupid S4 "anything I want it to be" R objects.
    I want ... either an annotated anndata object, or a dataframe and a dictionary. 
    Or a dictionary where one of the elements is a dataframe. 

    Things to return: 
			y$exprs <- as.matrix(object$E)
			y$Amean <- rowMeans(y$exprs,na.rm=TRUE)
			y$printer <- object$printer
			y$exprs <- as.matrix(object$M)
		y$weights <- object$weights
		y$probes <- object$genes
		y$design <- object$design
    Not all of these look required. exprs, Amean, exprs, probes and weights maybe. 

    It doesn't transform the data object, so maybe just pass back the metadata? 

    """""""""""""""""""""""""""
    y = []
    metadata = {}
    metadata["probes"] = data.columns
    metadata["weight"] = 1 # leaving this until I figure out what it's for. 
    metadata["design"] = pd.DataFrame() # again, leaving empty until I figure out how to fill it/where it's coming from. 

    # in the original, it's rows as variables, columns as samples, updating this to fit with everything else
    # where it's rows as samples, columns as variables. 
    metadata["Amean"] = data.mean()

    # There's a check that the dataframe holds numeric values. 
    # And a thing that gets rownames from probes. How do probes differ from rownames? 
    return metadata


if __name__ == "__main__":
    adata = ad.read_h5ad(anndataFile)
    #ic(getEAWP(data.to_df()))
    
    # check the design matrix
    # I think I don't want to check the design matrix. I think I'ma just gunna make it here. 
    df = pd.DataFrame([adata.obs.sample_id, adata.obs.cancer_type])
    df = df.T 
    ic(df.head())
    df.columns = ["sample_id", "cancerType"]
    
    cancerType = "Sarcoma"
    df.loc[df['cancerType'] != cancerType, 'cancerType'] = 'otherCancerType'

    ic(df.cancerType.value_counts())

    
    
    