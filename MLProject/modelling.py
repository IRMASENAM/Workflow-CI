import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Aktifkan autolog untuk sklearn
mlflow.sklearn.autolog()

# Load dataset preprocessing
df = pd.read_csv("heart_preprocessing.csv")

# Pisahkan fitur dan target
X = df.drop("target", axis=1)
y = df["target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Start MLflow run
with mlflow.start_run():

    # Buat model
    model = LogisticRegression(max_iter=1000)

    # Training
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)

    # Logging manual tambahan
    mlflow.log_metric("accuracy_manual", acc)