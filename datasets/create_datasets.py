# Sample datasets for testing machine learning algorithms

import numpy as np
import csv

# Iris dataset (simplified version)
iris_data = [
    [5.1, 3.5, 1.4, 0.2, 0],
    [4.9, 3.0, 1.4, 0.2, 0],
    [4.7, 3.2, 1.3, 0.2, 0],
    [4.6, 3.1, 1.5, 0.2, 0],
    [5.0, 3.6, 1.4, 0.2, 0],
    [5.4, 3.9, 1.7, 0.4, 0],
    [4.6, 3.4, 1.4, 0.3, 0],
    [5.0, 3.4, 1.5, 0.2, 0],
    [4.4, 2.9, 1.4, 0.2, 0],
    [4.9, 3.1, 1.5, 0.1, 0],
    [7.0, 3.2, 4.7, 1.4, 1],
    [6.4, 3.2, 4.5, 1.5, 1],
    [6.9, 3.1, 4.9, 1.5, 1],
    [5.5, 2.3, 4.0, 1.3, 1],
    [6.5, 2.8, 4.6, 1.5, 1],
    [5.7, 2.8, 4.5, 1.3, 1],
    [6.3, 3.3, 4.7, 1.6, 1],
    [4.9, 2.4, 3.3, 1.0, 1],
    [6.6, 2.9, 4.6, 1.3, 1],
    [5.2, 2.7, 3.9, 1.4, 1],
    [6.3, 3.3, 6.0, 2.5, 2],
    [5.8, 2.7, 5.1, 1.9, 2],
    [7.1, 3.0, 5.9, 2.1, 2],
    [6.3, 2.9, 5.6, 1.8, 2],
    [6.5, 3.0, 5.8, 2.2, 2],
    [7.6, 3.0, 6.6, 2.1, 2],
    [4.9, 2.5, 4.5, 1.7, 2],
    [7.3, 2.9, 6.3, 1.8, 2],
    [6.7, 2.5, 5.8, 1.8, 2],
    [7.2, 3.6, 6.1, 2.5, 2]
]

# Boston Housing dataset (simplified)
housing_data = [
    [0.00632, 18.0, 2.31, 24.0],
    [0.02731, 0.0, 7.07, 21.6],
    [0.02729, 0.0, 7.07, 34.7],
    [0.03237, 0.0, 2.18, 33.4],
    [0.06905, 0.0, 2.18, 36.2],
    [0.02985, 0.0, 2.18, 28.7],
    [0.08829, 12.5, 7.87, 22.9],
    [0.14455, 12.5, 7.87, 27.1],
    [0.21124, 12.5, 7.87, 16.5],
    [0.17004, 12.5, 7.87, 18.9],
    [0.22489, 12.5, 7.87, 15.0],
    [0.11747, 12.5, 7.87, 18.9],
    [0.09378, 12.5, 7.87, 21.7],
    [0.62976, 0.0, 8.14, 20.4],
    [0.63796, 0.0, 8.14, 18.2],
    [0.62739, 0.0, 8.14, 19.9],
    [1.05393, 0.0, 8.14, 23.1],
    [0.78420, 0.0, 8.14, 17.5],
    [0.80271, 0.0, 8.14, 20.2],
    [0.72580, 0.0, 8.14, 18.2]
]

# Wine quality dataset (simplified)
wine_data = [
    [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 5],
    [7.8, 0.88, 0.00, 2.6, 0.098, 25.0, 67.0, 0.9968, 3.20, 0.68, 9.8, 5],
    [7.8, 0.76, 0.04, 2.3, 0.092, 15.0, 54.0, 0.9970, 3.26, 0.65, 9.8, 5],
    [11.2, 0.28, 0.56, 1.9, 0.075, 17.0, 60.0, 0.9980, 3.16, 0.58, 9.8, 6],
    [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 5],
    [7.4, 0.66, 0.00, 1.8, 0.075, 13.0, 40.0, 0.9978, 3.51, 0.56, 9.4, 5],
    [7.9, 0.60, 0.06, 1.6, 0.069, 15.0, 59.0, 0.9964, 3.30, 0.46, 9.4, 5],
    [7.3, 0.65, 0.00, 1.2, 0.065, 15.0, 21.0, 0.9946, 3.39, 0.47, 10.0, 7],
    [7.8, 0.58, 0.02, 2.0, 0.073, 9.0, 18.0, 0.9968, 3.36, 0.57, 9.5, 7],
    [7.5, 0.50, 0.36, 6.1, 0.071, 17.0, 102.0, 0.9978, 3.35, 0.80, 10.5, 5]
]

# Save datasets to CSV files
with open('iris.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
    writer.writerows(iris_data)

with open('housing.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['crime_rate', 'zn', 'indus', 'price'])
    writer.writerows(housing_data)

with open('wine.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 
                    'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 
                    'pH', 'sulphates', 'alcohol', 'quality'])
    writer.writerows(wine_data)

print("Sample datasets created:")
print("- iris.csv: Classification dataset with 3 classes")
print("- housing.csv: Regression dataset for house price prediction")
print("- wine.csv: Classification dataset for wine quality prediction")
