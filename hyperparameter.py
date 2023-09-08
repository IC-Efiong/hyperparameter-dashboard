import streamlit as st
import pickle
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

#Create a streamlit app
st.title("ML Hyperparameter Tuning Dashboard")

#Upload data
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("Data uploaded successfully")
    st.sidebar.subheader("Data Preview")
    st.sidebar.write(data.head())

    #Data Preporcessing
    st.sidebar.header("Data Preprocessing")
    target_column = st.sidebar.selectbox("Select the target column", data.columns)
    feature_column = st.sidebar.multiselect("Select the feature column", data.columns)
    X = data[feature_column]
    y = data[target_column]

    #One-hot encoding
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    if st.sidebar.checkbox("One-Hot Encoding"):
        X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

    #Standard Scaling
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if st.sidebar. checkbox("Standard Scaling"):
        scaler = StandardScaler()
        X[numeric_features] = scaler.fit_transform(X[numeric_features])

    #Model selection
    st.sidebar.header("Select ML Algorithm")
    algorithm = st.sidebar.selectbox("Select an algorithm", ["Random Forest", "Gradient Boosting", "SVM", "K-Nearest Neighbors"])

    #Splitting data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Hyperparameter tuning
    st.sidebar.header("Hyperparameter Tuning")

    if st.sidebar.checkbox("Perform Hyperparameter Tuning"):
        grid_search = None  # Initialize grid_search variable outside the conditional block

        if algorithm == "Random Forest":
            n_estimators = st.sidebar.slider('Number of Estimators', min_value=10, max_value=300, value=100)
            max_depth = st.sidebar.selectbox('Max Depth', [None, 10, 20, 30])
            max_depth = [max_depth] if max_depth is not None else [None]  # Ensure it's always a list
            min_samples_split = st.sidebar.selectbox('Min Samples Split', [2, 5, 10])
            min_samples_split = [min_samples_split]  # Wrap the value in a list
    
            min_samples_leaf = st.sidebar.selectbox('Min Samples Leaf', [1, 2, 4])
            min_samples_leaf = [min_samples_leaf]  # Wrap the value in a list
            param_grid = {
                'n_estimators': [n_estimators],
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf
                }
            model = RandomForestClassifier()
            grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            st.subheader("Random Forest Hyperparameters")
            st.write(grid_search.best_params_)

        elif algorithm == "Gradient Boosting":
            n_estimators = st.sidebar.slider('Number of Estimators', min_value=10, max_value=300, value=100)
            learning_rate = st.sidebar.slider('Learning Rate', 0.01, 1.0, 0.1)
            max_depth = st.sidebar.selectbox('Max Depth', [3, 4, 5])

            param_grid = {
                'n_estimators': [n_estimators],
                'learning_rate': [learning_rate],
                'max_depth': [max_depth]
            }
            model = GradientBoostingClassifier()
            grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            st.subheader("Gradient Boosting Hyperparameters")
            st.write(grid_search.best_params_)

        elif algorithm == "SVM":
            C = st.sidebar.slider('C (Regularization Parameter)', min_value=0.01, max_value=10.0, step=0.01, value=1.0)
            kernel = st.sidebar.selectbox('Kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
            param_grid = {
                'C': [C],
                'kernel': [kernel],
                'gamma': ['scale', 'auto']  # You can adjust this list for different gamma options
            }
            model = SVC()
            grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            st.subheader("SVM Hyperparameters")
            st.write(grid_search.best_params_)

        elif algorithm == "K-Nearest Neighbors":
            n_neighbors = st.sidebar.slider('Number of Neigbors (K)', min_value=1, max_value=20, value=5)
            weights = st.sidebar.selectbox('Weights', ['uniform', 'distance'])
            algorithm = st.sidebar.selectbox('Algorithm', ['ball_tree', 'kd_tree', 'brute'])

            param_grid = {
                'n_neighbors': [n_neighbors],
                'weights': [weights],
                'algorithm':[algorithm]
            }

            model = KNeighborsClassifier()
            grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            st.subheader("K-Neighbors Hyperparameters")
            st.write(grid_search.best_params_)
        
        
        # Train the model with the best hyperparameters
        if grid_search is not None:
            model = grid_search.best_estimator_
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.subheader("Tuned Model Accuracy")
            st.write("Accuracy:", accuracy)

            # Save the model
            st.subheader("Save Model")
            save_option = st.radio("Select the format to save the model", ("joblib", "pickle"))

            if save_option == "joblib":
                model_filename = "tuned_model.joblib"
                joblib.dump(model, model_filename)
                st.write(f"The tuned model has been saved as {model_filename}")

            elif save_option == "pickle":
                model_filename = "tuned_model.pkl"
                with open(model_filename, 'wb') as file:
                    pickle.dump(model, file)
                    st.write(f"The tuned model has been saved as {model_filename}")
            
            # Generate confusion matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)

            # Display confusion matrix as text
            st.write("Confusion Matrix (as text):")
            st.write(cm)

            # Display confusion matrix as heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
                xticklabels=model.classes_, yticklabels=model.classes_)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(plt)


            # Feature Importance
            if algorithm == "Random Forest" or algorithm == "Gradient Boosting":
                feature_importance = model.feature_importances_
                feature_names = X.columns
                importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
                importance_df = importance_df.sort_values(by='Importance', ascending=False)

                # Display Feature Importance Table
                st.subheader("Feature Importance")
                st.write(importance_df)

                # Create a Bar Plot for Feature Importance
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=importance_df)
                plt.title('Feature Importance')
                #plt.tight_layout()

                # Display the plot
                st.pyplot(plt)
    
else:
    st.warning("Please upload a CSV file.")

