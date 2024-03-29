# Load necessary libraries
library(data.table)
library(fixest)
library(ggplot2)
library(haven)

mode <- "Heating" # Change this to "Heating" as needed

columns_to_read <- c("time", "Outdoor_Temperature", "id", if (mode == "Heating") "HeatingRuntime" else "CoolingRuntime", "s0_T", "s1_T", "s2_T", "s3_T", "s4_T", "s5_T")

# Load the data using haven::read_dta
data <- read_dta("/Users/omulayim/Downloads/final_df_unbinned.dta")

# Convert the data to data.table
setDT(data)

# Select only the necessary columns
data <- data[, ..columns_to_read]


# Convert 'time' to POSIXct to work with date and time functions
data[, time := as.POSIXct(time)]

# Extract hour and month from the 'time' column
data[, hour := format(time, "%H")]
data[, month := format(time, "%m")]

# Specify the runtime column based on the mode
runtime_column <- if (mode == "Heating") "HeatingRuntime" else "CoolingRuntime"

# Calculate duty cycle and name the column duty_cycle consistently
data[, duty_cycle := get(runtime_column) / 3600]

# Convert your dataset back to a data.frame if needed for compatibility with certain modeling functions
data_df <- as.data.frame(data)

# Remove the original data.table to save memory
rm(data)
gc()  # Call garbage collection to free up memory

# Fit the model with an additional fixed effect for month
model <- feols(duty_cycle ~ s0_T + s1_T + s2_T + s3_T + s4_T + s5_T + 
                 as.factor(id) + as.factor(hour)+ as.factor(month),
               data = data_df,  
               fixef = c("id", "hour","month"),  # Include 'month' as a fixed effect
               cluster = ~id)

# Summary of the model
summary(model)

# Save model
save(model, file = paste0("model_", tolower(mode), ".RData"))

# Assuming 'model' is your feols model object
summary_stats <- summary(model, se = "cluster")

# Extracting Estimates and Standard Errors correctly
estimates <- coef(summary_stats)  # This gets the coefficients
standard_errors <- se(summary_stats)  # This gets the standard errors


# Ensuring the initial creation of plot_data anticipates all necessary columns
plot_data <- data.frame(
  SensorCategory = factor(c("0","1", "2", "3", "4", "5")),
  Estimate = estimates[names(estimates) %in% c("s0_T","s1_T", "s2_T", "s3_T", "s4_T", "s5_T")],
  StdError = standard_errors[names(standard_errors) %in% c("s0_T","s1_T", "s2_T", "s3_T", "s4_T", "s5_T")],
  LowerCI = NA,  # Initialize with NAs, to be filled in
  UpperCI = NA   # Initialize with NAs, to be filled in
)

# Recalculate CIs for the entire plot_data, including the newly added reference entry
plot_data$LowerCI <- plot_data$Estimate - 1.96 * plot_data$StdError
plot_data$UpperCI <- plot_data$Estimate + 1.96 * plot_data$StdError

# Proceed with the ggplot2 plotting code as is


ggplot(plot_data, aes(x = SensorCategory, y = Estimate, group = 1)) +
  geom_point() +
  geom_errorbar(aes(ymin = LowerCI, ymax = UpperCI), width = 0.1) +
  geom_line() +  # This will now connect all points since they are considered part of the same group
  labs(title = paste("Impact of Sensor Count on", mode, "Duty Cycle"),
       x = "Number of Sensors",
       y = paste(mode, "Duty Cycle")) +
  theme_minimal() +  # Changed to theme_classic for white background and no gridlines
  theme(legend.position = "bottom",
        text = element_text(size = 14),  # Increase general text size
        plot.title = element_text(size = 16),  # Slightly larger title font size for emphasis
        axis.title = element_text(size = 16),  # Increase axis titles font size
        axis.text.x = element_text(size = 16),  # Increase x-axis tick label font size
        axis.text.y = element_text(size = 16),  # Increase y-axis tick label font size
        panel.border = element_rect(colour = "black", fill = NA, size = 1))  # Add border with specified color and size







