# Weather Prediction Project

This project uses Seattle weather data to classify weather types based on features such as precipitation, temperature, and wind. Three machine learning models—Naive Bayes, Decision Trees, and Random Forests—are implemented and evaluated to determine the most accurate model.

## Models Used
- **Naive Bayes**: A probabilistic classifier based on Bayes' theorem.
- **Decision Trees**: A hierarchical model that splits the data based on feature values.
- **Random Forest**: An ensemble method that uses multiple decision trees to improve accuracy.

---

## Dataset
The dataset used in this project contains daily weather information for Seattle, with the following columns:

- **date**: The date of the weather record (format: `YYYY-MM-DD`).
- **precipitation**: The amount of precipitation in millimeters (mm) for the day.
- **temp_max**: The maximum temperature in degrees Celsius (°C) for the day.
- **temp_min**: The minimum temperature in degrees Celsius (°C) for the day.
- **wind**: The wind speed in meters per second (m/s) for the day.
- **weather**: The weather type observed on that day. The possible values are:
  - `drizzle`
  - `rain`
  - `sun`
  - `snow`
  - `fog`

The dataset is preprocessed by encoding the `weather` column into numerical values and dropping the irrelevant `date` column for model training.

---

## Installation
To run the project, you need to install the required libraries. You can do so by running:
Clone the repository:
   ```bash
   git clone https://github.com/csm34/Seattle-Weather-ML-Algorithms.git
   ```

---

## Evaluation
The models are evaluated using accuracy, classification reports, and confusion matrices. Additionally, the feature importances for Decision Trees and Random Forests are visualized.

## Results
The models are compared based on their accuracy, and the Random Forest model generally performs the best.

---

## License
This project is licensed under the MIT License - see the LICENSE file for details.
