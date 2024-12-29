import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler


def get_dataset_overview(df):
    """
    Generate a comprehensive overview of the dataset
    """
    return {
        "Total Records": len(df),
        "Features": len(df.columns) - 1,  # Excluding target column
        "Target Classes": len(df['weather'].unique()),
        "Missing Values": df.isnull().sum().sum()
    }


def load_data():
    """Load and preprocess the Seattle weather dataset"""
    df = pd.read_csv('seattle-weather.csv')
    df_cleaned = df.drop(columns=['date'])
    weather_mapping = {'drizzle': 0, 'rain': 1, 'sun': 2, 'snow': 3, 'fog': 4}
    df_cleaned['weather_encoded'] = df_cleaned['weather'].map(weather_mapping)

    # Split features and target
    X = df_cleaned.drop(columns=['weather', 'weather_encoded'])
    y = df_cleaned['weather_encoded']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return df, df_cleaned, X, y, X_train, X_test, y_train, y_test, weather_mapping


def plot_weather_distribution(df):
    """Plot distribution of weather types"""
    fig, ax = plt.subplots()
    sns.countplot(x='weather', data=df, palette='viridis', ax=ax)
    ax.set_title("Distribution of Weather Types")
    st.pyplot(fig)


def plot_temp_relationship(df):
    """Plot relationship between max and min temperatures"""
    fig, ax = plt.subplots()
    sns.scatterplot(x='temp_max', y='temp_min', hue='weather', data=df, ax=ax)
    ax.set_title("Relationship Between Temp_max and Temp_min")
    st.pyplot(fig)


def train_models(X_train, X_test, y_train, y_test):
    """Train Naive Bayes, Decision Tree, and Random Forest models"""
    models = {
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)

        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'pred': y_pred
        }

    return results


def plot_confusion_matrix(y_test, y_pred, model_name, weather_mapping):
    """Plot confusion matrix for a given model"""
    fig, ax = plt.subplots()
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(weather_mapping.keys()),
                yticklabels=list(weather_mapping.keys()), ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)


def plot_feature_importance(model, X, model_name):
    """Plot feature importance for a given model"""
    fig, ax = plt.subplots()
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis', ax=ax)
    ax.set_title(f"{model_name} Feature Importance")
    st.pyplot(fig)


def main():
    st.title("Seattle Weather Analysis Dashboard")

    # Load data
    df, df_cleaned, X, y, X_train, X_test, y_train, y_test, weather_mapping = load_data()

    # Sidebar menu
    menu = st.sidebar.selectbox("Choose Analysis", [
        "Data Overview",
        "Data Visualization",
        "Model Training",
        "Model Comparison"
    ])

    if menu == "Data Overview":
        st.header("Dataset Overview")

        # Get dataset overview
        overview = get_dataset_overview(df)

        # Create columns for side-by-side display
        col1, col2, col3, col4 = st.columns(4)

        # Display overview metrics
        with col1:
            st.metric(label="Total Records", value=overview["Total Records"])

        with col2:
            st.metric(label="Features", value=overview["Features"])

        with col3:
            st.metric(label="Target Classes", value=overview["Target Classes"])

        with col4:
            st.metric(label="Missing Values", value=overview["Missing Values"])

        # Display first few rows
        st.subheader("First Few Rows")
        st.dataframe(df.head())

        # Weather Type Distribution
        st.subheader("Weather Type Distribution")
        weather_dist = df['weather'].value_counts()
        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(weather_dist)

        with col2:
            fig, ax = plt.subplots()
            weather_dist.plot(kind='pie', autopct='%1.1f%%', ax=ax)
            ax.set_title("Weather Type Percentage")
            st.pyplot(fig)

        # Descriptive Statistics
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe())

    elif menu == "Data Visualization":
        st.header("Weather Data Visualizations")

        viz_option = st.selectbox("Choose Visualization", [
            "Weather Type Distribution",
            "Temperature Relationship",
            "Correlation Heatmap"
        ])

        if viz_option == "Weather Type Distribution":
            plot_weather_distribution(df)

        elif viz_option == "Temperature Relationship":
            plot_temp_relationship(df)

        elif viz_option == "Correlation Heatmap":
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = pd.concat([X, y], axis=1).corr()
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)

    elif menu == "Model Training":
        st.header("Machine Learning Models")

        # Train models
        results = train_models(X_train, X_test, y_train, y_test)

        model_select = st.selectbox("Choose Model", list(results.keys()))

        model_result = results[model_select]

        st.write(f"{model_select} Results:")
        st.write(f"Test Accuracy: {model_result['accuracy']:.4f}")
        st.write(f"Cross-Validation Mean Accuracy: {model_result['cv_mean']:.4f}")
        st.write(f"Cross-Validation Std: {model_result['cv_std']:.4f}")

        # Confusion Matrix
        plot_confusion_matrix(y_test, model_result['pred'], model_select, weather_mapping)

        # Feature Importance (for Decision Tree and Random Forest)
        if model_select != 'Naive Bayes':
            plot_feature_importance(model_result['model'], X, model_select)

    elif menu == "Model Comparison":
        st.header("Model Performance Comparison")

        # Train models if not already trained
        results = train_models(X_train, X_test, y_train, y_test)

        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Test Accuracy': [results[model]['accuracy'] for model in results],
            'CV Mean Accuracy': [results[model]['cv_mean'] for model in results],
            'CV Std': [results[model]['cv_std'] for model in results]
        })

        st.write("Model Performance Comparison:")
        st.dataframe(comparison_df)

        # Bar plots for comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Test Accuracy Comparison
        sns.barplot(x='Model', y='Test Accuracy', data=comparison_df, ax=ax1)
        ax1.set_title('Test Accuracy Comparison')
        ax1.tick_params(axis='x', rotation=45)

        # Cross-validation Comparison
        sns.barplot(x='Model', y='CV Mean Accuracy', data=comparison_df, ax=ax2)
        ax2.errorbar(x=range(len(comparison_df)),
                     y=comparison_df['CV Mean Accuracy'],
                     yerr=comparison_df['CV Std'] * 2,
                     fmt='none', color='black', capsize=5)
        ax2.set_title('Cross-validation Accuracy')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        st.pyplot(fig)


if __name__ == "__main__":
    main()
