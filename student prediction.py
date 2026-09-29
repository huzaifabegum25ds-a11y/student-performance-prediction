from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np

# Sample student dataset
# [Study Hours, Attendance %, Previous Score]
X = np.array([
    [2, 60, 55],
    [3, 65, 60],
    [4, 70, 65],
    [5, 75, 70],
    [6, 80, 75],
    [7, 85, 80],
    [8, 90, 85],
    [9, 92, 88],
    [10, 95, 92],
    [1, 55, 50]
])

# Final scores
y = np.array([55, 60, 65, 70, 75, 80, 85, 88, 92, 50])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

# Calculate error
error = mean_absolute_error(y_test, predictions)

print("Student Performance Prediction")
print("--------------------------------")

print("Actual Scores:", y_test)
print("Predicted Scores:", predictions)

print("\nMean Absolute Error:", error)

# Predict a new student's score
new_student = [[7, 85, 80]]

predicted_score = model.predict(new_student)

print("\nNew Student Details:")
print("Study Hours: 7")
print("Attendance: 85%")
print("Previous Score: 80")

print("\nPredicted Final Score:",
      round(predicted_score[0], 2))
