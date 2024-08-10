# Set options and load necessary packages
options(stringsAsFactors = FALSE)

# Install and load packages if not already installed
required_packages <- c("tidyverse", "FactoMineR", "factoextra", "psych", "data.table", "cluster", "ggplot2", "clValid", "clusterCrit", "caret","MASS")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
  library(pkg, character.only = TRUE)
}


# Load the dataset
df <- read.csv('/Users/miadong/Desktop/Updated_Cleaned_Dataset.csv')

# Select relevant columns for PCA
columns_needed <- c(
  'Overall_Impact_of_Immigrants', 
  'Immigrants_Fill_Job_Vacancies', 
  'Immigrants_Strengthen_Diversity',
  'Immigrants_Increase_Crime', 
  'Asylum_for_Refugees', 
  'Immigrants_Increase_Terrorism_Risk', 
  'Immigration_Offers_Better_Living', 
  'Immigration_Increases_Unemployment', 
  'Immigration_Leads_to_Social_Conflict',
  'Preferred_Government_Action'
)

# Filter the dataset for PCA
df_pca <- df %>% select(all_of(columns_needed))

# Remove rows with missing values
df_pca <- na.omit(df_pca)

# Standardize the data
df_pca_scaled <- scale(df_pca)

# Perform PCA using FactoMineR
pca_result <- PCA(df_pca_scaled, graph = FALSE)

# Plot the variance explained by each principal component
fviz_eig(pca_result, addlabels = TRUE, ylim = c(0, 50)) +
  labs(title = "Scree Plot",
       x = "Principal Components",
       y = "Percentage of Variance Explained")

# Eigen-decomposition of the covariance matrix
cov_matrix <- cov(df_pca_scaled)
eigen_cov <- eigen(cov_matrix)

# Calculate loadings
loadings <- eigen_cov$vectors %*% diag(sqrt(eigen_cov$values))

# Interpret loadings
print(loadings)

# Calculate contributions of variables to components
contributions <- loadings^2
print(contributions)

# Calculate variance explained
variance_explained <- eigen_cov$values / sum(eigen_cov$values)
cumulative_variance_explained <- cumsum(variance_explained)

# Interpret variance explained
print(variance_explained)
print(cumulative_variance_explained)

# Calculate row scores
row_scores <- df_pca_scaled %*% loadings

# Check correlations
print(cor(row_scores, df_pca_scaled))

# Reconstruct data using all components
reconstructed_data_all <- row_scores %*% t(loadings)

# Reconstruct data using the first two components
reconstructed_data_2 <- row_scores[, 1:2] %*% t(loadings[, 1:2])

# Compare original and reconstructed data
print(head(df_pca_scaled))
print(head(reconstructed_data_all))
print(head(reconstructed_data_2))

# Perform PCA using the psych package
pca_psych <- pca(df_pca_scaled, nfactors = 2)

# Compare results
print(pca_psych)

# Create biplot for PCA
fviz_pca_biplot(pca_result, label = "var", habillage = df$Global_South, addEllipses = TRUE, ellipse.level = 0.95) +
  labs(title = "PCA Biplot of Immigration Attitudes")
# Clear unnecessary objects from memory to optimize usage
rm(df, df_pca, df_pca_scaled, loadings, contributions, variance_explained, cumulative_variance_explained, row_scores, reconstructed_data_all, reconstructed_data_2, pca_psych, loadings_df, loadings_long, dummies, df_demo, df_demo_encoded)
gc()  # Perform garbage collection to free memory




# Corrected code to plot the loadings for the first five principal components
loadings_df <- as.data.frame(loadings)
colnames(loadings_df) <- paste0("PC", 1:ncol(loadings_df))
loadings_df$Variables <- rownames(loadings_df)

# Pivot the data to a long format
loadings_long <- loadings_df %>%
  pivot_longer(-Variables, names_to = "Principal_Component", values_to = "Loadings")

# Filter to only include the first five principal components
loadings_long_filtered <- loadings_long %>%
  filter(Principal_Component %in% paste0("PC", 1:5))

# Set the correct order for the Variables factor
loadings_long_filtered$Variables <- factor(loadings_long_filtered$Variables, levels = unique(loadings_df$Variables))

# Verify the filtering step
print(unique(loadings_long_filtered$Principal_Component))

# Plot the loadings for the first five principal components
ggplot(loadings_long_filtered, aes(x = Principal_Component, y = Loadings, fill = Variables)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Loadings of Variables on Principal Components (PC1 to PC5)",
       x = "Principal Components",
       y = "Loadings") +
  theme_minimal()

# Extract PCA loadings and contributions
loadings <- pca_result$var$coord

# Check the structure of loadings
print(str(loadings))

if (!is.null(loadings) && ncol(loadings) > 0) {
  loadings_df <- as.data.frame(loadings)
  colnames(loadings_df) <- paste0("PC", 1:ncol(loadings_df))
  loadings_df$Variables <- rownames(loadings_df)
  
  loadings_long <- loadings_df %>%
    pivot_longer(-Variables, names_to = "Principal_Component", values_to = "Loadings")
  
  ggplot(loadings_long, aes(x = Principal_Component, y = Loadings, fill = Variables)) +
    geom_bar(stat = "identity", position = "dodge") +
    labs(title = "Loadings of Variables on Principal Components",
         x = "Principal Components",
         y = "Loadings") +
    theme_minimal()
} else {
  print("Error: Loadings object is empty or not correctly formed.")
}
# New one Now continue with K-means clustering
# K-means Clustering Part

# Select relevant columns for demographic and other variables
columns_needed_demo <- c(
  'Global_South', 'Age', 'Sex', 'Highest_Education_Level', 'Employment_Status',
  'Religious', 'Income_Scale', 'Social_Class_Subjective', 'Sector_of_Employment',
  'Father_Highest_Education_Level', 'Mother_Highest_Education_Level'
)

# Filter the dataset for these variables
df_demo <- df %>% select(all_of(columns_needed_demo))
df_demo <- na.omit(df_demo)

# Convert categorical variables to factors
categorical_vars <- c(
  'Sex', 'Highest_Education_Level', 'Employment_Status', 'Religious',
  'Sector_of_Employment', 'Father_Highest_Education_Level', 'Mother_Highest_Education_Level'
)

df_demo[categorical_vars] <- lapply(df_demo[categorical_vars], factor)

# Apply one-hot encoding to categorical variables
dummies <- dummyVars(" ~ .", data = df_demo)
df_demo_encoded <- data.frame(predict(dummies, newdata = df_demo))

# Combine PCA scores with encoded demographic data
df_combined <- cbind(row_scores, df_demo_encoded)

# Standardize the combined data
df_combined_scaled <- scale(df_combined)


# Perform K-means clustering
set.seed(123)
kmeans_result <- kmeans(df_combined_scaled, centers = 4, nstart = 25)

# Add cluster labels to the data
df_combined$Cluster <- factor(kmeans_result$cluster)

# Plot the clustering results
fviz_cluster(kmeans_result, data = df_combined_scaled, geom = "point",
             ellipse.type = "convex", ggtheme = theme_minimal(),
             main = "K-means Clustering of Immigration Attitudes with Demographics")

# Inspect the clustering results
print(table(df_combined$Cluster))
print(summary(df_combined))
print(kmeans_result$centers)

# Analyze cluster centers to understand the characteristics of each cluster
cluster_centers <- as.data.frame(kmeans_result$centers)
cluster_centers <- cbind(cluster_centers, Cluster = 1:4)
print(cluster_centers)
# Clear unnecessary objects from memory to optimize usage
rm(df, df_pca, df_pca_scaled, loadings, contributions, variance_explained, cumulative_variance_explained, row_scores, reconstructed_data_all, reconstructed_data_2, pca_psych, loadings_df, loadings_long, dummies, df_demo, df_demo_encoded)
gc()  # Perform garbage collection to free memory
#the rest of methods
# Identify columns that are constant within each cluster
constant_columns <- sapply(df_combined, function(x) {
  all(sapply(split(x, df_combined$Cluster), function(y) length(unique(y)) == 1))
})

# Print constant columns
print(which(constant_columns))

# Remove constant columns
df_combined_no_constants <- df_combined[, !constant_columns]

# Ensure 'Cluster' is still in the dataframe
if(!'Cluster' %in% colnames(df_combined_no_constants)) {
  df_combined_no_constants$Cluster <- df_combined$Cluster
}

# Inspect the cleaned dataframe
print(str(df_combined_no_constants))
# Load necessary package
library(MASS)

# Perform LDA
lda_model <- lda(Cluster ~ ., data = df_combined_no_constants)

# Display the results
print(lda_model)
# Coefficients of linear discriminants
print(lda_model$scaling)

# Proportion of trace (variance explained by each linear discriminant)
print(lda_model$svd^2 / sum(lda_model$svd^2))
# Predict the cluster memberships
lda_predictions <- predict(lda_model)
df_combined_no_constants$LDA_Predicted <- lda_predictions$class

# Evaluate the LDA model
confusion_matrix <- table(df_combined_no_constants$Cluster, df_combined_no_constants$LDA_Predicted)
print(confusion_matrix)

# Calculate the accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy of LDA model:", accuracy))
# Plot LDA results
lda_df <- data.frame(LD1 = lda_predictions$x[, 1], LD2 = lda_predictions$x[, 2], Cluster = df_combined_no_constants$Cluster)

ggplot(lda_df, aes(x = LD1, y = LD2, color = Cluster)) +
  geom_point(alpha = 0.5) +
  labs(title = "LDA: Linear Discriminants Plot",
       x = "Linear Discriminant 1",
       y = "Linear Discriminant 2") +
  theme_minimal()

# Plot the first two linear discriminants
lda_df$Cluster <- as.factor(lda_df$Cluster)
ggplot(lda_df, aes(x = LD1, y = LD2, color = Cluster)) +
  geom_point(alpha = 0.7) +
  stat_ellipse(level = 0.95, aes(fill = Cluster), alpha = 0.2, show.legend = FALSE) +
  labs(title = "LDA: Linear Discriminants Plot",
       x = "Linear Discriminant 1",
       y = "Linear Discriminant 2") +
  theme_minimal() +
  scale_color_manual(values = c("red", "blue", "green", "purple"))

# Load necessary libraries
library(tidyverse)
library(caret)

# Split the dataset into Global South and Global North
df_global_south <- df_combined %>% filter(Global_South == 1)
df_global_north <- df_combined %>% filter(Global_South == 0)

# Ensure only numeric columns are used by selecting numeric columns manually
df_global_south_numeric <- df_global_south %>% select_if(is.numeric)
df_global_north_numeric <- df_global_north %>% select_if(is.numeric)

# Function to remove rows with NA/NaN/Inf values
remove_na_inf <- function(df) {
  df <- df %>% filter_all(all_vars(!is.na(.))) %>% filter_all(all_vars(!is.nan(.))) %>% filter_all(all_vars(is.finite(.)))
  return(df)
}

# Remove rows with NA/NaN/Inf values
df_global_south_numeric <- remove_na_inf(df_global_south_numeric)
df_global_north_numeric <- remove_na_inf(df_global_north_numeric)

# Additional check for columns with any NA/NaN/Inf values
na_south <- sapply(df_global_south_numeric, function(x) sum(!is.finite(x)))
na_north <- sapply(df_global_north_numeric, function(x) sum(!is.finite(x)))

print("Columns with NA/NaN/Inf values in Global South dataset after removal:")
print(na_south[na_south > 0])

print("Columns with NA/NaN/Inf values in Global North dataset after removal:")
print(na_north[na_north > 0])

# Ensure there are no NA/NaN/Inf values left
if (sum(na_south) > 0 | sum(na_north) > 0) {
  stop("There are still NA/NaN/Inf values in the dataset.")
}

# Perform K-means clustering
set.seed(123)
kmeans_south <- kmeans(scale(df_global_south_numeric), centers = 4, nstart = 25)
kmeans_north <- kmeans(scale(df_global_north_numeric), centers = 4, nstart = 25)

# Print cluster centers
print("Cluster centers for Global South:")
print(kmeans_south$centers)

print("Cluster centers for Global North:")
print(kmeans_north$centers)

# Load necessary libraries
library(caret)
library(tidyverse)

# Function to perform K-means clustering and return cluster labels
perform_kmeans <- function(data, centers) {
  kmeans_result <- kmeans(data, centers = centers, nstart = 25)
  return(kmeans_result$cluster)
}

# Install and load caret package
install.packages("caret")
library(caret)

# Cross-Validation
train_control <- trainControl(method = "cv", number = 10)
lda_cv_model <- train(Cluster ~ ., data = df_combined_no_constants, method = "lda", trControl = train_control)
print(lda_cv_model)

# External Validation
external_variable <- 'Income_Scale'
mean_values <- aggregate(df_combined[, external_variable], by = list(Cluster = df_combined$Cluster), FUN = mean)
print(mean_values)
# External Validation for Social_Class_Subjective
external_variable <- 'Social_Class_Subjective'
mean_values <- aggregate(df_combined[, external_variable], by = list(Cluster = df_combined$Cluster), FUN = mean)
print(mean_values)
# External Validation for Religious
external_variable <- 'Religious'
mean_values <- aggregate(df_combined[, external_variable], by = list(Cluster = df_combined$Cluster), FUN = mean)
print(mean_values)
# External Validation for Age
external_variable <- 'Age'
mean_values <- aggregate(df_combined[, external_variable], by = list(Cluster = df_combined$Cluster), FUN = mean)
print(mean_values)
# External Validation for Highest_Education_Level
external_variable <- 'Highest_Education_Level'
mean_values <- aggregate(df_combined[, external_variable], by = list(Cluster = df_combined$Cluster), FUN = mean)
print(mean_values)
# External Validation for Employment_Status
external_variable <- 'Employment_Status'
mean_values <- aggregate(df_combined[, external_variable], by = list(Cluster = df_combined$Cluster), FUN = mean)
print(mean_values)
# External Validation for Global_South
external_variable <- 'Global_South'
mean_values <- aggregate(df_combined[, external_variable], by = list(Cluster = df_combined$Cluster), FUN = mean)
print(mean_values)




# Optionally, combine all results into a single data frame for easier comparison
combined_mean_values <- Reduce(function(x, y) merge(x, y, by = "Cluster", all = TRUE), mean_values_list)
print(combined_mean_values)
# Define all demographic variables
demographic_variables <- c('Global_South', 'Age', 'Sex', 'Highest_Education_Level', 
                           'Employment_Status', 'Religious', 'Income_Scale', 
                           'Social_Class_Subjective', 'Sector_of_Employment', 
                           'Father_Highest_Education_Level', 'Mother_Highest_Education_Level')

# Function to calculate and print mean values for each variable
validate_demographics <- function(variable) {
  mean_values <- aggregate(df_combined[, variable], by = list(Cluster = df_combined$Cluster), FUN = mean)
  print(paste("Mean values for", variable))
  print(mean_values)
}

# Iterate over all demographic variables
for (variable in demographic_variables) {
  validate_demographics(variable)
}

# Define the list of demographic variables
demographic_variables <- c(
  'Global_South', 'Age', 'Sex', 'Highest_Education_Level', 
  'Employment_Status', 'Religious', 'Income_Scale', 
  'Social_Class_Subjective', 'Sector_of_Employment', 
  'Father_Highest_Education_Level', 'Mother_Highest_Education_Level'
)

# Initialize an empty list to store the results
validation_results <- list()

# Loop through each demographic variable and calculate the mean values for each cluster
for (variable in demographic_variables) {
  if (variable %in% colnames(df_combined)) {
    mean_values <- aggregate(df_combined[, variable], by = list(Cluster = df_combined$Cluster), FUN = mean, na.rm = TRUE)
    validation_results[[variable]] <- mean_values
    print(paste("Mean values for", variable))
    print(mean_values)
  } else {
    print(paste("Variable", variable, "not found in the dataset."))
  }
}

# Display the validation results
validation_results
# Print column names to verify the available variables
print(colnames(df_combined))

# Define the list of demographic variables
demographic_variables <- c(
  'Global_South', 'Age', 'Sex.1', 'Highest_Education_Level.1', 
  'Employment_Status.1', 'Religious.1', 'Income_Scale', 
  'Social_Class_Subjective', 'Sector_of_Employment.1', 
  'Father_Highest_Education_Level.1', 'Mother_Highest_Education_Level.1'
)

# Initialize an empty list to store the results
validation_results <- list()

# Loop through each demographic variable and calculate the mean values for each cluster
for (variable in demographic_variables) {
  if (variable %in% colnames(df_combined)) {
    mean_values <- aggregate(df_combined[, variable], by = list(Cluster = df_combined$Cluster), FUN = mean, na.rm = TRUE)
    validation_results[[variable]] <- mean_values
    print(paste("Mean values for", variable))
    print(mean_values)
  } else {
    print(paste("Variable", variable, "not found in the dataset."))
  }
}

# Display the validation results
validation_results
# Define the list of demographic variables including all levels
demographic_variables <- c(
  'Global_South', 'Age', 'Sex', 'Highest_Education_Level', 
  'Employment_Status', 'Religious', 'Income_Scale', 
  'Social_Class_Subjective', 'Sector_of_Employment', 
  'Father_Highest_Education_Level', 'Mother_Highest_Education_Level'
)
# External Validation for all demographic variables
validation_results <- list()

# Loop through each demographic variable
for (variable in demographic_variables) {
  # Check if the variable exists in the dataset
  if (variable %in% colnames(df_combined)) {
    mean_values <- aggregate(df_combined[, variable], by = list(Cluster = df_combined$Cluster), FUN = mean)
    validation_results[[variable]] <- mean_values
    print(paste("Mean values for", variable))
    print(mean_values)
  } else {
    print(paste("Variable", variable, "not found in the dataset."))
  }
}

# Display the validation results
validation_results

# Cluster Validation

# Sample a subset of the data for silhouette analysis
set.seed(123)
sample_indices <- sample(1:nrow(df_combined_scaled), size = 10000, replace = FALSE)
df_combined_sample <- df_combined_scaled[sample_indices, ]
kmeans_sample_result <- kmeans(df_combined_sample, centers = 3, nstart = 25)

# Silhouette Analysis on the sample
silhouette_values <- silhouette(kmeans_sample_result$cluster, dist(df_combined_sample))
# Explore different values for k
sil_width <- vector()
for (k in 2:10) {
  kmeans_result <- kmeans(df_combined_scaled, centers = k, nstart = 25)
  sil <- silhouette(kmeans_result$cluster, dist(df_combined_scaled))
  sil_width[k] <- mean(sil[, 3])
}

# Plot silhouette width for different k values
plot(1:10, sil_width, type = "b", xlab = "Number of clusters (k)", ylab = "Average Silhouette Width")

# Plot silhouette values
fviz_silhouette(silhouette_values) + labs(title = "Silhouette Plot for K-means Clustering (Sampled Data)")

# Calculate Dunn Index on full data
dunn_index <- intCriteria(as.matrix(df_combined_scaled), kmeans_result$cluster, "Dunn")
print(paste("Dunn Index:", dunn_index$dunn))

# Calculate Davies-Bouldin Index on full data
davies_bouldin_index <- intCriteria(as.matrix(df_combined_scaled), kmeans_result$cluster, "Davies_Bouldin")
print(paste("Davies-Bouldin Index:", davies_bouldin_index$davies_bouldin))




#plot of distribution of attitude scores by global south and north
# Filter the data
df_filtered <- df_long %>% filter(Score >= 0 & Score <= 5)

# Calculate percentage for each group
df_filtered <- df_filtered %>%
  group_by(Attitude, Global_South, Score) %>%
  summarise(Count = n()) %>%
  mutate(Percentage = Count / sum(Count) * 100)

# Create the percentage bar plot 
ggplot(df_filtered, aes(x = factor(Score), y = Percentage, fill = Global_South)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~Attitude, scales = "free_y", ncol = 2) +
  scale_x_discrete(breaks = 0:5) +
  scale_y_continuous(labels = scales::percent_format(scale = 1)) +
  labs(title = "Distribution of Attitude Scores by Global Region",
       x = "Attitude Score",
       y = "Percentage",
       fill = "Region") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 0, hjust = 0.5, size = 8),   # Adjusts x-axis text size and alignment
    axis.text.y = element_text(size = 8),                          # Adjusts y-axis text size
    strip.text = element_text(size = 6),                          # Adjusts subplot title text size
    plot.title = element_text(size = 10, hjust = 0.5),             # Adjusts main plot title text size and alignment
    legend.title = element_text(size = 7),                        # Adjusts legend title text size
    legend.text = element_text(size = 7),                         # Adjusts legend text size
    strip.background = element_blank(),                            # Removes background from subplot titles
    panel.grid.major = element_blank(),                            # Removes major grid lines
    panel.grid.minor = element_blank(),                            # Removes minor grid lines
    axis.line = element_line(color = "black")                      # Adds axis lines for better clarity
  )
