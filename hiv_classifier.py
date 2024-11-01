# hiv_classifier.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

class HIVClassifier:
    def __init__(self):
        # Train model upon initialization
        self.model, self.scaler, self.columns, self.label_encoder = self.train_model()

    def train_model(self):
        # Load dataset
        file_path = 'hiv_ds.csv'
        data = pd.read_csv(file_path)

        # Encode target and categorical features
        label_encoder = LabelEncoder()
        data['Result'] = label_encoder.fit_transform(data['Result'])

        # One-hot encode categorical features
        data_encoded = pd.get_dummies(data, columns=[
            'Marital Staus', 'STD', 'Educational Background', 'HIV TEST IN PAST YEAR',
            'AIDS education', 'Places of seeking sex partners', 'SEXUAL ORIENTATION', 'Drug- taking'
        ])

        # Split features and target
        X = data_encoded.drop(columns=['Result'])
        y = data_encoded['Result']

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Initialize and train SVM classifier
        model = SVC(kernel='linear', random_state=42)
        model.fit(X_train, y_train)

        # Make predictions and evaluate the model
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

        return model, scaler, X.columns, label_encoder

    def predict(self, input_data):
        # Convert input data into a DataFrame
        input_df = pd.DataFrame([input_data])

        # One-hot encode input and align with training columns
        input_encoded = pd.get_dummies(input_df).reindex(columns=self.columns, fill_value=0)

        # Scale the input features
        input_scaled = self.scaler.transform(input_encoded)

        # Predict the result and decode the label
        prediction = self.model.predict(input_scaled)
        return self.label_encoder.inverse_transform(prediction)[0]
