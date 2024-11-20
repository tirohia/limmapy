# Define the path to the lmFit.R file
lmfit_file <- "/data/projects/limmapy/limmaRcode/R/lmfit.R"
classFile <- "/data/projects/limmapy/limmaRcode/R/classes.R"
ebayesFile <- "/data/projects/limmapy/limmaRcode/R/ebayes.R"
squeezeVarFile <- "/data/projects/limmapy/limmaRcode/R/squeezeVar.R"
fitDistFile <- "/data/projects/limmapy/limmaRcode/R/fitFDist.R"
testsFile <- "/data/projects/limmapy/limmaRcode/R/decidetests.R"
utilityFile <- "/data/projects/limmapy/limmaRcode/R/utility.R"

# Source the lmFit.R file to load the lmFit function
source(lmfit_file)
source(classFile)
source(ebayesFile)
source(squeezeVarFile)
source(fitDistFile)
source(testsFile)
source(utilityFile)

library("statmod")

# Define file paths for the synthetic data and design matrix
data_file <- "synthetic_data.csv"
design_file <- "synthetic_design_matrix.csv"


# Read the synthetic data and design matrix
data <- read.csv(data_file, row.names = 1)
design_matrix <- read.csv(design_file, row.names = 1)

# Check the structure of the data
#cat("Data Structure:\n")
#str(data)
#cat("Design Matrix Structure:\n")
#str(design_matrix)

# Ensure the data is in the correct format for lmFit
# Convert data to a matrix
data_matrix <- as.matrix(data)
data_matrix <- t(data_matrix)

# Convert design matrix to a matrix
design_matrix <- as.matrix(design_matrix)

# Print dimensions to ensure they match
cat("Dimensions of data matrix:", dim(data_matrix)[1], "samples and", dim(data_matrix)[2], "genes\n")
cat("Dimensions of design matrix:", dim(design_matrix)[1], "samples and", dim(design_matrix)[2], "predictors\n")

# Run the lmFit function from the sourced file
fit <- lmFit(data_matrix, design = design_matrix)

# Print the fitting results
cat("Fitting Results:\n")
#print(fit$stdev.unscaled)
eb_results <- eBayes(fit)
#print(eb_results$t)

# Optionally, you can save the results to a file
saveRDS(fit, file = "lmFit_results.rds")