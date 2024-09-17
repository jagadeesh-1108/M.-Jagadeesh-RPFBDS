# Load necessary libraries
library(ggplot2)
library(caret)
library(dplyr)
library(glmnet)

# Check the column names in your data
print(colnames(df))

# Convert the 'Date' column to Date type if it's not already in that format
df$Date <- as.Date(df$Date, format="%Y-%m-%d")  # Adjust the format if necessary

# Extract the month from the 'Date' column
df$Month <- format(df$Date, "%b")  # This will give you abbreviated month names (Jan, Feb, etc.)

# Perform one-hot encoding on the new 'Month' column
df <- df %>%
  mutate_at(vars(Month), as.factor) %>%  # Convert to factor
  bind_cols(as.data.frame(model.matrix(~ Month - 1, data = df))) %>%  # One-hot encode
  select(-Month)  # Remove original 'Month' column

# Defining X (features) and y (target variable)
X <- df %>% select(Customers, MonthJan, MonthFeb, MonthMar, MonthApr, MonthMay, MonthJun, MonthJul, MonthAug, MonthSep, MonthOct, MonthNov, MonthDec)
y <- df$Sales

# Splitting data into training and testing sets (80/20 split)
set.seed(42)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[trainIndex,]
X_test <- X[-trainIndex,]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]

# Standardizing the features (scale)
scaler <- preProcess(X_train, method = c("center", "scale"))
X_train_scaled <- predict(scaler, X_train)
X_test_scaled <- predict(scaler, X_test)

# Fitting the linear regression model
model <- lm(y_train ~ ., data = as.data.frame(X_train_scaled))

# Make predictions on the test set
y_pred <- predict(model, newdata = as.data.frame(X_test_scaled))

# Evaluate the model performance (calculate RMSE and R-squared)
rmse <- sqrt(mean((y_test - y_pred)^2))
train_r2 <- summary(model)$r.squared
test_r2 <- cor(y_test, y_pred)^2

print(paste("Train R-squared: ", train_r2))
print(paste("Test R-squared: ", test_r2))
print(paste("RMSE: ", rmse))

# Plot the actual vs predicted values using ggplot2
actual_vs_pred <- data.frame(y_test = y_test, y_pred = y_pred)

ggplot(actual_vs_pred, aes(x = y_test, y = y_pred)) +
  geom_point(alpha = 0.2) +
  geom_smooth(method = "lm", color = "blue") +
  labs(x = "Actual Values", y = "Predicted Values", title = "Linear Regression Model: Actual vs Predicted") +
  theme_minimal()
