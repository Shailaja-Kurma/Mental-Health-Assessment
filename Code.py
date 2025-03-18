# Import necessary libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from google.colab import files 
from sklearn.linear_model import LinearRegression, LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score 
from sklearn.preprocessing import LabelEncoder 
from sklearn.exceptions import ConvergenceWarning 
from sklearn.metrics import accuracy_score 
import warnings 
import ipywidgets as widgets 
from ipywidgets import interact, HBox, VBox 
# Suppress convergence warnings 
warnings.filterwarnings('ignore', category=ConvergenceWarning) 
# Load the data 
data = pd.read_csv('/content/Supression.csv') 
data.isnull().sum() 
df = data.dropna() 
# Data visualization 
data2 = data[[ 'Sleep', 'Appetite', 'Interest', 'Fatigue', 'Worthlessness', 
'Concentration', 'Agitation', 'Suicidal Ideation', 'Sleep Disturbance', 
18 
'Aggression', 'Panic Attacks', 'Hopelessness', 'Restlessness', 'Low Energy']] 
plt.imshow(data2.corr()) 
plt.show() 
# Converting categorical data to numerical 
categorical_data = ["Mild", "Moderate", "Severe", "No depression"] 
label_encoder = LabelEncoder() 
numerical_data = label_encoder.fit_transform(categorical_data) 
print(dict(zip(categorical_data, numerical_data))) 
# One-hot encoding 
one_hot_encoded_data = pd.get_dummies(df) 
# Splitting the data 
X = df.drop('Depression State', axis=1) 
Y = df['Depression State'] 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) 
print("Training set shape:", X_train.shape, Y_train.shape) 
print("Test set shape:", X_test.shape, Y_test.shape) 
# Define and train the models 
logistic_reg = LogisticRegression(max_iter=1000) 
logistic_reg.fit(X_train, Y_train) 
# Accuracy for Logistic Regression 
y_pred_logistic = logistic_reg.predict(X_test) 
accuracy_logistic = accuracy_score(Y_test, y_pred_logistic) 
print(f"Accuracy of Logistic Regression: {accuracy_logistic:.4f}") 
# Grid search for RandomForestClassifier 
param_grid = { 
19 
'bootstrap': [True, False], 
'max_depth': [10, None], 
'min_samples_leaf': [1, 2], 
'min_samples_split': [2, 5], 
'n_estimators': [100, 200] 
} 
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, 
cv=3, n_jobs=-1, verbose=2) 
grid_search.fit(X_train, Y_train) 
best_params = grid_search.best_params_ 
best_rfc = grid_search.best_estimator_ 
# Accuracy for Random Forest 
y_pred_rf = best_rfc.predict(X_test) 
accuracy_rf = accuracy_score(Y_test, y_pred_rf) 
print(f"Accuracy of Random Forest: {accuracy_rf:.4f}") 
# Cross-validation accuracy for Random Forest 
cv_scores = cross_val_score(best_rfc, X_train, Y_train, cv=3) 
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}") 
# Define and train the KNN classifier 
def train_knn(X_train, y_train, k): 
knn_model = KNeighborsClassifier(n_neighbors=k) 
knn_model.fit(X_train, Y_train) 
return knn_model 
knn_model = train_knn(X_train, Y_train, k=5) 
# Accuracy for KNN 
y_pred_knn = knn_model.predict(X_test) 
20 
21 
 
accuracy_knn = accuracy_score(Y_test, y_pred_knn) 
print(f"Accuracy of KNN Classifier: {accuracy_knn:.4f}") 
 
# Prediction function 
def predict_depression(model, input_data): 
    if len(input_data) != X_train.shape[1]: 
        raise ValueError(f"Input data should have {X_train.shape[1]} features.") 
 
    if model == 'Logistic regression': 
        clf = logistic_reg 
    elif model == 'Random Forest': 
        clf = best_rfc 
    elif model == 'KNN Classifier': 
        clf = knn_model 
    else: 
        raise ValueError("Invalid model specified. Choose from the dropdown.") 
 
    input_df = pd.DataFrame([input_data], columns=X_train.columns) 
    prediction = clf.predict(input_df)[0] 
    depression_states_mapping = dict(zip(clf.classes_, df['Depression State'].unique())) 
    return depression_states_mapping[prediction] 
 
# List of meaningful feature names corresponding to their features 
meaningful_feature_names = [ 'Sleep', 'Appetite', 'Interest', 'Fatigue', 'Worthlessness', 
                            'Concentration', 'Agitation', 'Suicidal Ideation', 'Sleep Disturbance', 
                            'Aggression', 'Panic Attacks', 'Hopelessness', 'Restlessness', 'Low Energy'] 
 
# Create input widgets 
input_widgets = [widgets.FloatText(description=meaningful_feature_names[i]) for i in 
range(len(meaningful_feature_names))] 
 
# Create model selection dropdown 
model_dropdown = widgets.Dropdown(options=['Logistic regression', 'Random Forest', 'KNN 
Classifier'], description='Model:') 
# Create a button for prediction 
predict_button = widgets.Button(description="Predict") 
# Create an output widget to display the prediction result 
output = widgets.Output() 
def on_predict_button_clicked(b): 
input_values = [widget.value for widget in input_widgets] 
model = model_dropdown.value 
prediction = predict_depression(model, input_values) 
with output: 
output.clear_output() 
print(f"Predicted depression state ({model}): {prediction}") 
# Attach the event handler to the button 
predict_button.on_click(on_predict_button_clicked) 
# Display the widgets 
input_widgets_box = VBox(input_widgets) 
ui = VBox([model_dropdown, input_widgets_box, predict_button, output]) 
display(ui)
