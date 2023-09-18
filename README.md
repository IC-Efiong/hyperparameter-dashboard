
# ML Hyperparameter Tuning Dashboard




## Introduction
This repository contains a Streamlit web application for hyperparameter tuning of machine learning models. The application offers a user-friendly interface to upload a CSV file, preprocess the data, select a machine learning algorithm, and fine-tune its hyperparameters.
## Getting Started
To use this application, ensure you have the required libraries installed. You can install them using the following command:


## Installation

```bash
  pip install -r requirements.txt

```
    
## Usage

**1.    Uploading Data**

* Click on the "Upload Data" section on the sidebar.
* Upload your CSV file using the file uploader.
  
**2.    Data Preprocessing**

* Select the target column and one or more feature columns from the dataset.
* Optionally apply one-hot encoding and standard scaling.

**3.    Select ML Algorithm**

* Choose one of the available algorithms: Random Forest, Gradient Boosting, SVM, or K-Nearest Neighbors.

**4.    Hyperparameter Tuning**

* If selected, the application will perform hyperparameter tuning for the chosen algorithm. This step is optional.

**5.    Model Training and Evaluation**

* The model is trained using the best hyperparameters obtained from the tuning process.
* The tuned model's accuracy is displayed.

**6. Save Model**

* The tuned model can be saved in either joblib or pickle format.

**7. Confusion Matrix**

* A confusion matrix is displayed to evaluate the model's performance.

**8. Feature Importance (Random Forest and Gradient Boosting only)**

* For these algorithms, the feature importance is displayed in a table and as a bar plot.

## Notes
* Ensure that the uploaded CSV file contains both feature and target columns.
* The application currently supports classification tasks.
* Before uploading your data makesure the target column is in numerical form (it has already been labeled).
## How to Run
Use the following command to run the Streamlit app:
```bash
    streamlit run <filename>.py
```
Replace <filename> with the name of your Python file containing the code.
## Dependencies
* Streamlit
* Pandas
* Numpy
* Matplotlib
* Seaborn
* Scikit-learn
## License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/)


## Authors

- [@IC-Efiong](https://www.github.com/IC-Efiong)


## Tech Stack

**Client:** Streamlit, Python

**Server:** Streamlit

