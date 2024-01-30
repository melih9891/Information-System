# Script dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Function to visualize distribution of target variable
def visualize_price_distribution(data, title):
    plt.figure(figsize=(10, 6))
    sns.histplot(data['realSum'], bins=60, kde=True)
    plt.title(title)
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.show()

# Function to visualize boxplot
def visualize_boxplot(data, x, y, title):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=x, y=y, data=data)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

# Function to evaluate and print model summary
def fit_ols_model(X, y, title):
    # Check and display data types of features
    print(f"Data types of features in {title}:")
    print(X.dtypes)

    # Convert boolean features to numeric
    X_numeric = X.copy()
    bool_cols = X.select_dtypes(include='bool').columns
    X_numeric[bool_cols] = X_numeric[bool_cols].astype(int)

    # Check and display data types after conversion
    print("\nData types after conversion:")
    print(X_numeric.dtypes)

    # Handle missing values if any
    X_numeric = X_numeric.apply(pd.to_numeric, errors='coerce')
    X_numeric = X_numeric.dropna()
    y = pd.to_numeric(y, errors='coerce')
    y = y[X_numeric.index]

    # Add a constant term to the features
    X_numeric = sm.add_constant(X_numeric)

    # Fit the OLS model
    model = sm.OLS(y, X_numeric).fit()

    # Print the model summary
    print(f"\n{title} OLS Model Summary:")
    print(model.summary())


# Function to fit and evaluate random forest model
def fit_random_forest_model(X_train, X_test, y_train, y_test, title):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluate the model on training and testing sets
    train_predictions = rf_model.predict(X_train)
    test_predictions = rf_model.predict(X_test)

    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)

    print(f"\n{title} Random Forest Model Evaluation:")
    print(f"Training MSE: {train_mse:.2f}")
    print(f"Testing MSE: {test_mse:.2f}")

    # Feature importance
    feature_importances = rf_model.feature_importances_
    top_features = X_train.columns[np.argsort(feature_importances)[::-1]][:3]
    print(f"Top 3 features for {title}: {top_features}")

# Import datasets
london_weekdays = pd.read_csv("me_goeckce/data/london_weekdays.csv")
london_weekends = pd.read_csv("me_goeckce/data/london_weekends.csv")

london_weekdays.columns

# Data Exploration
visualize_price_distribution(london_weekdays, 'Distribution of Price on Weekdays')
visualize_price_distribution(london_weekends, 'Distribution of Price on Weekends')

visualize_boxplot(london_weekdays, 'room_type', 'realSum', 'Room Type vs. Price on Weekdays')
visualize_boxplot(london_weekends, 'room_type', 'realSum', 'Room Type vs. Price on Weekends')

visualize_boxplot(london_weekdays, 'host_is_superhost', 'realSum', 'Host Superhost Influence on Price on Weekdays')
visualize_boxplot(london_weekends, 'host_is_superhost', 'realSum', 'Host Superhost Influence on Price on Weekends')

# Feature Engineering
# Consider creating new features or transforming existing ones based on your domain knowledge

# Model Preparation and Evaluation
# Weekdays
X_weekdays = london_weekdays[['cleanliness_rating', 'room_private', 'host_is_superhost', 
                              'person_capacity', 'dist', 'metro_dist', 'attr_index_norm', 'rest_index_norm']]
y_weekdays = london_weekdays['realSum']

fit_ols_model(X_weekdays, y_weekdays, 'Weekdays')
X_train_weekdays, X_test_weekdays, y_train_weekdays, y_test_weekdays = train_test_split(X_weekdays, y_weekdays, test_size=0.2, random_state=42)
fit_random_forest_model(X_train_weekdays, X_test_weekdays, y_train_weekdays, y_test_weekdays, 'Weekdays')

# Weekends
X_weekends = london_weekends[['cleanliness_rating', 'room_private', 'host_is_superhost', 
                              'person_capacity', 'dist', 'metro_dist', 'attr_index_norm', 'rest_index_norm']]
y_weekends = london_weekends['realSum']

fit_ols_model(X_weekends, y_weekends, 'Weekends')
X_train_weekends, X_test_weekends, y_train_weekends, y_test_weekends = train_test_split(X_weekends, y_weekends, test_size=0.2, random_state=42)
fit_random_forest_model(X_train_weekends, X_test_weekends, y_train_weekends, y_test_weekends, 'Weekends')


# Descriptive Statistics
numerical_columns = ['realSum', 'person_capacity', 'cleanliness_rating', 'guest_satisfaction_overall', 'bedrooms', 'dist', 'metro_dist', 'attr_index', 'rest_index', 'lng', 'lat']
categorical_columns = ['room_type', 'host_is_superhost', 'multi', 'biz']

# Mean, Median, Mode for numerical columns
descriptive_stats_week_days = london_weekdays[numerical_columns].agg(['mean', 'median', lambda x: x.mode().iloc[0]])
descriptive_stats_weekends = london_weekends[numerical_columns].agg(['mean', 'median', lambda x: x.mode().iloc[0]])

# Count, Unique values, Frequency for categorical columns
categorical_stats_week_days = london_weekdays[categorical_columns].agg(['count', 'nunique', lambda x: x.value_counts().index[0]])
categorical_stats_weekends = london_weekends[categorical_columns].agg(['count', 'nunique', lambda x: x.value_counts().index[0]])

# Display descriptive statistics
print("Descriptive Statistics for Weekdays:")
print(descriptive_stats_week_days)

print("Descriptive Statistics for Weekends:")
print(descriptive_stats_weekends)

print("\nCategorical Statistics for Weekdays:")
print(categorical_stats_week_days)

print("\nCategorical Statistics for Weekends:")
print(categorical_stats_weekends)

# Summary statistics
# room type
room_type_weekdays = london_weekdays['room_type'].value_counts().sort_index()
room_type_weekends = london_weekends['room_type'].value_counts().sort_index()

room_type_weekdays
room_type_weekends

# Person capacity
person_capacity_weekdays = london_weekdays['person_capacity'].value_counts().sort_index()
person_capacity_weekends = london_weekends['person_capacity'].value_counts().sort_index()

person_capacity_weekdays
person_capacity_weekends

# Private rooms
room_private_weekdays = london_weekdays['room_private'].value_counts().sort_index()
room_private_weekends = london_weekends['room_private'].value_counts().sort_index()

room_private_weekdays
room_private_weekends

# Shared rooms
room_shared_weekdays = london_weekdays['room_shared'].value_counts().sort_index()
room_shared_weekends = london_weekends['room_shared'].value_counts().sort_index()

room_shared_weekdays
room_shared_weekends

# superhosts
host_is_superhost_weekdays = london_weekdays['host_is_superhost'].value_counts().sort_index()
host_is_superhost_weekends = london_weekends['host_is_superhost'].value_counts().sort_index()

host_is_superhost_weekdays
host_is_superhost_weekends

# business
biz_weekdays = london_weekdays['biz'].value_counts().sort_index()
biz_weekends = london_weekends['biz'].value_counts().sort_index()

biz_weekdays
biz_weekends

# attraction index
attraction_index_weekdays = london_weekdays['attr_index_norm'].mean()
attraction_index_weekends = london_weekends['attr_index_norm'].mean()

attraction_index_weekdays
attraction_index_weekends

# distance
dist_weekdays = london_weekdays['dist'].mean()
dist_weekends = london_weekends['dist'].mean()

dist_weekdays
dist_weekends

# guest satisfaction rating
guest_satisfaction_overall_weekdays = london_weekdays['guest_satisfaction_overall'].mean()
guest_satisfaction_overall_weekends = london_weekends['guest_satisfaction_overall'].mean()

guest_satisfaction_overall_weekdays
guest_satisfaction_overall_weekends

# avg price per room type
avg_price_by_room_type_weekdays = london_weekdays.groupby('room_type')['realSum'].mean()
avg_price_by_room_type_weekends = london_weekends.groupby('room_type')['realSum'].mean()

avg_price_by_room_type_weekdays
avg_price_by_room_type_weekends

# average guest satisfaction by room type
average_satisfaction_by_room_type_weekdays = london_weekdays.groupby('room_type')['guest_satisfaction_overall'].mean()
average_satisfaction_by_room_type_weekends = london_weekends.groupby('room_type')['guest_satisfaction_overall'].mean()

average_satisfaction_by_room_type_weekdays
average_satisfaction_by_room_type_weekends

# Average cleanliness rating by room type
avg_cleanliness_rating_by_room_type_weekdays = london_weekdays.groupby('room_type')['cleanliness_rating'].mean()
avg_cleanliness_rating_by_room_type_weekends = london_weekends.groupby('room_type')['cleanliness_rating'].mean()

avg_cleanliness_rating_by_room_type_weekdays
avg_cleanliness_rating_by_room_type_weekends

# average distance to metro by room type
avg_metro_dist_by_room_type_weekdays = london_weekdays.groupby('room_type')['dist'].mean()
avg_metro_dist_by_room_type_weekends = london_weekends.groupby('room_type')['dist'].mean()

avg_metro_dist_by_room_type_weekdays
avg_metro_dist_by_room_type_weekends

# Weekday listings' prices distribution
plt.figure(figsize=(10, 6))
plt.hist(london_weekdays['realSum'], bins=90, color='darkgreen', edgecolor='black')
plt.title('Histogram of Real Sum')
plt.xlabel('Real Sum')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.08)
plt.show()

# Weekend listings' prices distribution
plt.figure(figsize=(10, 6))
plt.hist(london_weekdays['realSum'], bins=90, color='teal', edgecolor='black')
plt.title('Histogram of Real Sum')
plt.xlabel('Real Sum')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.08)
plt.show()
