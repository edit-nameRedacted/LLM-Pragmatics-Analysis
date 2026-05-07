# Install necessary packages if you don't have them
# install.packages(c("tidyverse", "lme4", "lmerTest", "broom.mixed"))

library(tidyverse)
library(lme4)
library(lmerTest)

# 1. Load the compiled data
args     <- commandArgs(trailingOnly = FALSE)
file_arg <- args[startsWith(args, "--file=")]
ROOT     <- if (length(file_arg) > 0)
              normalizePath(file.path(dirname(sub("--file=", "", file_arg[1])), "../.."))
            else
              normalizePath(file.path(getwd()))

df <- read_csv(file.path(ROOT, "data", "human", "data_compiled_sim.csv"))

# 2. Function to prep data and run multilevel models for a specific LLM
run_multilevel_analysis <- function(data, model_prefix, model_name) {
  
  cat(sprintf("\n\n=======================================================\n"))
  cat(sprintf(" MULTILEVEL ANALYSIS: %s\n", toupper(model_name)))
  cat(sprintf("=======================================================\n"))
  
  # Define the columns we want for this specific model
  token_col <- paste0(model_prefix, "_mean_token_entropy")
  beam_col  <- paste0(model_prefix, "_beam_cluster_entropy")
  sem_col   <- paste0(model_prefix, "_semantic_entropy")
  
  # Check if columns exist
  if(!all(c(token_col, beam_col, sem_col) %in% names(data))) {
    stop("Could not find all required columns for this prefix.")
  }
  
  # Extract relevant subset
  df_sub <- data %>%
    select(question, condition, rater_scores, context_question_similarity, ctx_tokens,
           all_of(token_col), all_of(beam_col), all_of(sem_col))
  
  # Standardize predictors
  df_sub <- df_sub %>%
    mutate(
      acc_rater_z = scale(rater_scores)[, 1],
      acc_sim_z   = scale(context_question_similarity)[, 1],
      ctx_token_z = scale(ctx_tokens)[, 1]
    )
  
  # Extract Baseline (No Context)
  df_nc <- df_sub %>%
    filter(condition == "no_context") %>%
    select(question, 
           nc_token = all_of(token_col), 
           nc_beam = all_of(beam_col), 
           nc_sem = all_of(sem_col))
  
  # Calculate Deltas (Condition - Baseline)
  df_analysis <- df_sub %>%
    filter(condition != "no_context") %>%
    left_join(df_nc, by = "question") %>%
    mutate(
      delta_token = !!sym(token_col) - nc_token,
      delta_beam  = !!sym(beam_col) - nc_beam,
      delta_sem   = !!sym(sem_col) - nc_sem
    ) %>%
    drop_na(delta_token, delta_beam, delta_sem, acc_rater_z, acc_sim_z, ctx_token_z)
  
  # 1. Token Entropy with Condition Interactions
  cat("\n--- 1. DV: Change in TOKEN Entropy by Condition ---\n")
  lme_token_cond <- lmer(delta_token ~ acc_rater_z * condition + acc_sim_z * condition + ctx_token_z + (1 | question), data = df_analysis)
  print(summary(lme_token_cond)$coefficients)
  
  # 2. Beam Cluster Entropy with Condition Interactions
  cat("\n--- 2. DV: Change in BEAM CLUSTER Entropy by Condition ---\n")
  lme_beam_cond <- lmer(delta_beam ~ acc_rater_z * condition + acc_sim_z * condition + ctx_token_z + (1 | question), data = df_analysis)
  print(summary(lme_beam_cond)$coefficients)
  
  # 3. Semantic Entropy with Condition Interactions
  cat("\n--- 3. DV: Change in SEMANTIC Entropy by Condition ---\n")
  lme_sem_cond <- lmer(delta_sem ~ acc_rater_z * condition + acc_sim_z * condition + ctx_token_z + (1 | question), data = df_analysis)
  print(summary(lme_sem_cond)$coefficients)
}

# ==============================================================================
# --- EXECUTE THE ANALYSIS ---
# ==============================================================================

# MLA Models
run_multilevel_analysis(df, "DS", "DeepSeek-V3 (MLA)")
run_multilevel_analysis(df, "D1", "DS-V2-Lite (MLA)")

# GQA Models
run_multilevel_analysis(df, "QW", "Qwen (GQA)")
run_multilevel_analysis(df, "LL", "LLaMA (GQA)")
run_multilevel_analysis(df, "MS", "Mistral (GQA)")
