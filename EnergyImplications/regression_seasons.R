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

# Extract month from 'time' column
data[, month := format(time, "%m")]

# Specify the runtime column based on the mode
runtime_column <- if (mode == "Heating") "HeatingRuntime" else "CoolingRuntime"

# Calculate duty cycle and name the column duty_cycle consistently
data[, duty_cycle := get(runtime_column) / 3600]

# Create a 'season' column based on 'month'
data[, season := fifelse(month %in% c('03', '04', '05'), 'Spring',
                         fifelse(month %in% c('06', '07', '08'), 'Summer',
                                 fifelse(month %in% c('09', '10', '11'), 'Autumn', 'Winter')))]

# Create dummy variables directly for each season
data[, `Spring` := as.integer(season == 'Spring')]
data[, `Summer` := as.integer(season == 'Summer')]
data[, `Autumn` := as.integer(season == 'Autumn')]
data[, `Winter` := as.integer(season == 'Winter')]

# Create interaction terms for each temperature variable with the season dummies
data[, `s0_T:Spring` := s0_T * Spring]
data[, `s0_T:Summer` := s0_T * Summer]
data[, `s0_T:Autumn` := s0_T * Autumn]
data[, `s0_T:Winter` := s0_T * Winter]

data[, `s1_T:Spring` := s1_T * Spring]
data[, `s1_T:Summer` := s1_T * Summer]
data[, `s1_T:Autumn` := s1_T * Autumn]
data[, `s1_T:Winter` := s1_T * Winter]

data[, `s2_T:Spring` := s2_T * Spring]
data[, `s2_T:Summer` := s2_T * Summer]
data[, `s2_T:Autumn` := s2_T * Autumn]
data[, `s2_T:Winter` := s2_T * Winter]

data[, `s3_T:Spring` := s3_T * Spring]
data[, `s3_T:Summer` := s3_T * Summer]
data[, `s3_T:Autumn` := s3_T * Autumn]
data[, `s3_T:Winter` := s3_T * Winter]

data[, `s4_T:Spring` := s4_T * Spring]
data[, `s4_T:Summer` := s4_T * Summer]
data[, `s4_T:Autumn` := s4_T * Autumn]
data[, `s4_T:Winter` := s4_T * Winter]

data[, `s5_T:Spring` := s5_T * Spring]
data[, `s5_T:Summer` := s5_T * Summer]
data[, `s5_T:Autumn` := s5_T * Autumn]
data[, `s5_T:Winter` := s5_T * Winter]

# Convert your dataset back to a data.frame if needed for compatibility with certain modeling functions
data_df <- as.data.frame(data)

# Remove the original data.table to save memory
rm(data)
gc()  # Call garbage collection to free up memory

# Fit the model including interaction terms between each s_T variable and season
model <- feols(duty_cycle ~  
                 `s0_T:Spring` + `s0_T:Summer` + `s0_T:Autumn` + `s0_T:Winter` +
                 `s1_T:Spring` + `s1_T:Summer` + `s1_T:Autumn` + `s1_T:Winter` +
                 `s2_T:Spring` + `s2_T:Summer` + `s2_T:Autumn` + `s2_T:Winter` +
                 `s3_T:Spring` + `s3_T:Summer` + `s3_T:Autumn` + `s3_T:Winter` +
                 `s4_T:Spring` + `s4_T:Summer` + `s4_T:Autumn` + `s4_T:Winter` +
                 `s5_T:Spring` + `s5_T:Summer` + `s5_T:Autumn` + `s5_T:Winter` +
                  as.factor(id) + as.factor(hour)  + as.factor(month),
               data = data_df,  
               fixef = c("id", "hour","month"),
               cluster = ~id)

# Summary of the model
summary(model)

# Save model
save(model, file = paste0("model_seasoned_", tolower(mode), ".RData"))

# Assuming 'model' is your feols model object
summary_stats <- summary(model, se = "cluster")

# Extracting coefficients and standard errors
estimates <- coef(summary(model))  # This gets the coefficients
standard_errors <- se(summary(model))  # This gets the standard errors

# Filter for interaction terms and prepare for manual sensor categorization
sensor_season_terms <- names(estimates)[grep("s[0-5]_T:", names(estimates))]

plot_data <- data.frame(Term = sensor_season_terms,
                        Estimate = estimates[sensor_season_terms],
                        StdError = standard_errors[sensor_season_terms],
                        stringsAsFactors = FALSE)

# Calculate confidence intervals
plot_data$LowerCI <- plot_data$Estimate - 1.96 * plot_data$StdError
plot_data$UpperCI <- plot_data$Estimate + 1.96 * plot_data$StdError

# Manually assign SensorCategory and Season
plot_data$SensorCategory <- c(0,0,0,0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5)
plot_data$Season <- rep(c("Spring", "Summer", "Autumn", "Winter"), 6)

# Convert SensorCategory to a factor for plotting
plot_data$SensorCategory <- factor(plot_data$SensorCategory)
plot_data$Season <- factor(plot_data$Season, levels = c("Spring", "Summer", "Autumn", "Winter"))

ggplot(plot_data, aes(x = SensorCategory, y = Estimate, color = Season, group = Season)) +
  geom_point() +
  geom_line(aes(group = Season), size = 1.5) +  # Increase line thickness
  geom_errorbar(aes(ymin = LowerCI, ymax = UpperCI), width = 0.1) +
  scale_color_manual(values = c("Spring" = "green", "Summer" = "red", "Autumn" = "orange", "Winter" = "blue")) +
  labs(title = paste("Impact of Sensor Count on", mode, "Duty Cycle by Season"),
       x = "Number of Sensors",
       y = paste(mode, "Duty Cycle"),
       color = "Season") +
  theme_minimal() +
  theme(legend.position = "bottom",
        text = element_text(size = 14),  # Increase general text size
        plot.title = element_text(size = 16),  # Slightly larger title font size for emphasis
        axis.title = element_text(size = 14),  # Increase axis titles font size
        axis.text.x = element_text(size = 12),  # Increase x-axis tick label font size
        axis.text.y = element_text(size = 12),  # Increase y-axis tick label font size
        legend.title = element_text(size = 14),  # Increase legend title font size
        legend.text = element_text(size = 14),  # Increase legend text font size
        panel.border = element_rect(colour = "black", fill = NA, size = 1))  # Add border with specified color and size

