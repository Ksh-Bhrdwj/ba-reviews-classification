# British Airways Reviews Classification

## Overview

This project aims to predict whether a review will recommend British Airways based on various features including review text, ratings, and other categorical attributes. The dataset comprises customer reviews with attributes such as seat comfort, cabin staff service, food and beverages, ground service, value for money, and more. By leveraging machine learning techniques, we seek to build models that accurately predict recommendations and provide insights into the factors that influence customer satisfaction.

## Key Objectives

1. **Data Preprocessing**: Clean and preprocess the data, handling missing values and transforming features appropriately.
2. **Feature Engineering**: Extract meaningful features from the review text using TF-IDF vectorization.
3. **Model Training**: Train multiple machine learning models including Logistic Regression, Random Forest, and Gradient Boosting.
4. **Hyperparameter Tuning**: Optimize model performance through hyperparameter tuning.
5. **Model Evaluation**: Evaluate model performance using metrics such as accuracy, precision, recall, F1 score, and AUC-ROC.
6. **Model Interpretation**: Interpret model predictions using SHAP values to understand feature importance.
7. **Deployment**: Save the best-performing model for future predictions and demonstrate its usage on new data.

## Repository Contents

- `ba_reviews_classification.ipynb`: Jupyter notebook containing the entire analysis and model development process.
- `best_random_forest_model.joblib`: Saved model file.
- `requirements.txt`: List of required dependencies.
- `LICENSE`: License file for the project.

## Data Dictionary

| Column Name            | Description                                      |
|------------------------|--------------------------------------------------|
| aircraft               | Type of aircraft                                 |
| type_of_traveller      | Type of traveler (e.g., Solo Leisure)            |
| seat_type              | Class of the seat (e.g., Economy)                |
| route                  | Route of the flight                              |
| rating                 | Overall rating given by the reviewer             |
| seat_comfort           | Rating for seat comfort                          |
| cabin_staff_service    | Rating for cabin staff service                   |
| food_and_beverages     | Rating for food and beverages                    |
| ground_service         | Rating for ground service                        |
| value_for_money        | Rating for value for money                       |
| wifi_and_connectivity  | Rating for wifi and connectivity                 |
| recommend              | Whether the reviewer recommends the airline (bool) |
| review                 | Text of the review                               |
| review_date            | Date when the review was written                 |
| date_flown             | Date when the reviewer flew                      |

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ba_reviews_classification.git
    cd ba_reviews_classification
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Notebook

1. Open the Jupyter notebook:
    ```bash
    jupyter notebook ba_reviews_classification.ipynb
    ```

2. Run all cells in the notebook to reproduce the analysis and model development process.

### Using the Model

1. Load the saved model:
    ```python
    import joblib
    model_filename = 'best_random_forest_model.joblib'
    loaded_model = joblib.load(model_filename)
    ```

2. Preprocess new data and make predictions:
    ```python
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    import scipy.sparse

    # Sample new data (replace this with your actual new data)
    new_data = pd.DataFrame({
        'aircraft': ['Boeing 747'],
        'type_of_traveller': ['Solo Leisure'],
        'seat_type': ['Economy'],
        'route': ['London to New York'],
        'rating': [8],
        'seat_comfort': [7],
        'cabin_staff_service': [9],
        'food_and_beverages': [8],
        'ground_service': [6],
        'value_for_money': [7],
        'wifi_and_connectivity': [5],
        'review': ['Great flight with excellent service!']  # Ensure 'review' column is present
    })

    # Preprocess the new data
    tfidf_vectorizer = TfidfVectorizer(max_features=500)
    new_data_text = tfidf_vectorizer.fit_transform(new_data['review'])
    new_data.drop(columns=['review'], inplace=True)

    # Define preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Transform structured features using the same preprocessing pipeline
    new_data_transformed = preprocessor.fit_transform(new_data)

    # Combine text features with structured features
    new_data_combined = scipy.sparse.hstack((new_data_transformed, new_data_text))

    # Make predictions
    predictions = loaded_model.predict(new_data_combined)
    prediction_probabilities = loaded_model.predict_proba(new_data_combined)[:, 1]

    # Output predictions
    print("Predictions:", predictions)
    print("Prediction Probabilities:", prediction_probabilities)
    ```

## Key Findings

- The Random Forest model with tuned hyperparameters achieved the highest AUC-ROC score.
- Feature importance analysis highlighted the most influential factors in predicting recommendations.

## Future Work

1. **Further Optimization**:
    - Continue optimizing the Gradient Boosting model.
    - Explore additional ensemble methods to combine predictions from multiple models.

2. **Enhanced Interpretability**:
    - Utilize tools like LIME for more granular explanations of individual predictions.

3. **Expanded Evaluation**:
    - Visualize additional performance metrics such as Precision-Recall curves and Confusion Matrices for a more comprehensive evaluation.

## Common Issues and Troubleshooting

1. **Issue**: Jupyter notebook fails to start.
   - **Solution**: Ensure Jupyter is installed correctly and you have activated your Python environment.

2. **Issue**: Missing data or incorrect data types.
   - **Solution**: Verify that the data preprocessing steps are correctly applied, especially handling missing values and type conversions.

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- shap
- joblib
- lime

## License

This project is licensed under the MIT License - see the [LICENSE](License.txt) file for details.

## Acknowledgements

- [SkyTrax](https://www.airlinequality.com/) for providing the review data.
- [scikit-learn](https://scikit-learn.org/stable/) for the machine learning tools and libraries.
- [SHAP](https://github.com/slundberg/shap) for the SHAP values interpretation.

---

Feel free to contribute to this project by opening issues or submitting pull requests. For any questions, contact [kushh2706@gmail.com](mailto:kushh2706@gmail.com).
