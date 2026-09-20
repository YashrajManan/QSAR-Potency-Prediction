# =============================================================================
# QSAR: Predicting Dopamine D2 Receptor Potency from Chemical Structure (R twin)
#
# Reads the real, cleaned ChEMBL D2-receptor data exported by
# qsar_d2_potency.ipynb (data/d2_cleaned.csv: molecule_chembl_id,
# canonical_smiles, standard_value, pIC50) -- run the Python notebook through
# Step 1 first so that file exists. Same compounds, same pIC50 targets as the
# Python pipeline; R-specific tooling (rcdk/CDK) for featurization.
# =============================================================================

library(rcdk)
library(randomForest)
library(fingerprint)

df <- read.csv("data/d2_cleaned.csv", stringsAsFactors = FALSE)
cat(nrow(df), "real, cleaned D2-receptor compounds loaded\n")
dir.create("results", showWarnings = FALSE)

## ---- Step 1 -- parse SMILES, compute descriptors + circular fingerprints --
## rcdk wraps the CDK (Chemistry Development Kit) Java library -- the closest
## R equivalent to RDKit, though less standard in real QSAR practice.
mols <- parse.smiles(df$canonical_smiles)

desc_names <- c(
  "org.openscience.cdk.qsar.descriptors.molecular.WeightDescriptor",
  "org.openscience.cdk.qsar.descriptors.molecular.XLogPDescriptor",       # R's LogP estimator (XLogP) differs numerically from Python's Crippen LogP -- not directly comparable value-for-value, both are valid lipophilicity estimates
  "org.openscience.cdk.qsar.descriptors.molecular.HBondDonorCountDescriptor",
  "org.openscience.cdk.qsar.descriptors.molecular.HBondAcceptorCountDescriptor",
  "org.openscience.cdk.qsar.descriptors.molecular.TPSADescriptor"
)
descs <- eval.desc(mols, desc_names)

fps <- lapply(mols, get.fingerprint, type = "circular")
fp_matrix <- fingerprint::fp.to.matrix(fps)

keep_rows <- complete.cases(fp_matrix) & complete.cases(descs) & !is.na(df$pIC50)
X <- cbind(fp_matrix[keep_rows, ], descs[keep_rows, ])
y <- df$pIC50[keep_rows]
cat(dim(X), "\n")

## ---- Step 2 -- train/test split ------------------------------------------
## NOTE (stated caveat, not a bug): rcdk has no one-line Murcko-scaffold split
## as convenient as RDKit's, so this R twin uses a plain seeded random split
## instead of the Python notebook's leakage-free scaffold split. This is a
## known simplification -- it lets near-identical analogs land on both sides
## of train/test, which inflates apparent accuracy relative to the honest,
## scaffold-split Python evaluation. See README/INTERPRETATION for the
## resulting (higher, less trustworthy) R metrics vs. Python's.
set.seed(0)
n <- nrow(X)
test_idx <- sample(seq_len(n), size = round(0.2 * n))
train_idx <- setdiff(seq_len(n), test_idx)

X_train <- X[train_idx, ]; X_test <- X[test_idx, ]
y_train <- y[train_idx];   y_test  <- y[test_idx]
cat("train:", length(train_idx), " test:", length(test_idx), "\n")

## ---- Step 3 -- train a Random Forest regressor ----------------------------
rf <- randomForest(x = X_train, y = y_train, ntree = 500)
y_pred <- predict(rf, X_test)

## ---- Step 4 -- evaluate: RMSE, R^2, Spearman rho -- vs. baseline ----------
## R^2 = 1 - SS_res/SS_tot (the actual coefficient of determination -- NOT
## squared correlation, which is undefined for a constant baseline predictor
## and not strictly equivalent to R^2 in general). This matches sklearn's
## r2_score exactly, making the R and Python numbers genuinely comparable.
rmse <- sqrt(mean((y_test - y_pred)^2))
r2   <- 1 - sum((y_test - y_pred)^2) / sum((y_test - mean(y_test))^2)
rho  <- cor(y_test, y_pred, method = "spearman")

baseline_pred <- rep(mean(y_train), length(y_test))
baseline_rmse <- sqrt(mean((y_test - baseline_pred)^2))
baseline_r2   <- 1 - sum((y_test - baseline_pred)^2) / sum((y_test - mean(y_test))^2)

cat(sprintf("Model    : RMSE=%.3f  R2=%.3f  Spearman=%.3f\n", rmse, r2, rho))
cat(sprintf("Baseline : RMSE=%.3f  R2=%.3f\n", baseline_rmse, baseline_r2))
write.csv(data.frame(y_test = y_test, y_pred = y_pred),
          "results/predictions_R.csv", row.names = FALSE)

## ---- Step 5 -- interpret: feature importance ------------------------------
importances <- importance(rf)
top20 <- head(importances[order(-importances[, 1]), , drop = FALSE], 20)
print(top20)
write.csv(top20, "results/feature_importance_R.csv")

# =============================================================================
# INTERPRETATION (real run)
#
# Result: RMSE=0.601, R2=0.662, Spearman rho=0.819, vs. baseline RMSE=1.034,
# R2~0.000 -- numerically stronger than the Python scaffold-split run
# (RMSE=0.736, R2=0.511, rho=0.682). This is EXPECTED and is not evidence R
# "performs better": the random split here lets near-duplicate analogs leak
# across train/test (see Step 2 note), inflating apparent accuracy. The
# Python scaffold split is the more honest estimate of real generalization
# to novel chemistry -- report both numbers together with this caveat, never
# the R number alone as "the" result.
# =============================================================================
