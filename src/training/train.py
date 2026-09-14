import os
import mlflow
import mlflow.sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.data.make_dataset import load_raw_data, split_data
from src.features.build_features import get_preprocessor
from src.evaluation.evaluate import calculate_metrics

# Configuración del Tracking Server de MLflow en DagsHub

DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "VeroGonzalez25")
MLFLOW_TRACKING_URI = f"https://dagshub.com/{DAGSHUB_USERNAME}/customer-churn-ml.mlflow"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("customer-churn-experiment")

def run_training():
    # 1. Cargar datos y realizar partición
    df = load_raw_data("data/raw/customer_churn_historical.csv")
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.20, random_state=42)

    # 2. Identificar tipos de columnas
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

    # 3. Crear preprocesador base
    preprocessor = get_preprocessor(numeric_features, categorical_features)

    # 4. Definir las 6 configuraciones de experimentos
    experiments_config = [
        {
            "name": "baseline_dummy",
            "model": DummyClassifier(strategy="most_frequent"),
            "params": {"model_type": "DummyClassifier", "strategy": "most_frequent"}
        },
        {
            "name": "logreg_c1",
            "model": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            "params": {"model_type": "LogisticRegression", "C": 1.0, "max_iter": 1000}
        },
        {
            "name": "logreg_c01",
            "model": LogisticRegression(C=0.1, max_iter=1000, random_state=42),
            "params": {"model_type": "LogisticRegression", "C": 0.1, "max_iter": 1000}
        },
        {
            "name": "rf_default",
            "model": RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42),
            "params": {"model_type": "RandomForestClassifier", "n_estimators": 100, "max_depth": "None"}
        },
        {
            "name": "rf_depth_10",
            "model": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            "params": {"model_type": "RandomForestClassifier", "n_estimators": 100, "max_depth": 10}
        },
        {
            "name": "rf_depth_5_n50",
            "model": RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
            "params": {"model_type": "RandomForestClassifier", "n_estimators": 50, "max_depth": 5}
        }
    ]

    best_recall = -1.0
    best_model_uri = None

    # 5. Iterar y registrar cada Run en MLflow
    for cfg in experiments_config:
        with mlflow.start_run(run_name=cfg["name"]) as run:
            # Construir Pipeline completo (Preprocessor + Model)
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', cfg["model"])
            ])

            # Entrenar Pipeline
            pipeline.fit(X_train, y_train)

            # Predecir sobre conjunto de prueba
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

            # Calcular y registrar métricas
            metrics = calculate_metrics(y_test, y_pred, y_proba)
            
            mlflow.log_params(cfg["params"])
            mlflow.log_metrics(metrics)
            
            # Guardar el artefacto del pipeline entrenado
            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                skops_trusted_types=["numpy.dtype"]
            )

            print(f"Run '{cfg['name']}' completado -> Recall: {metrics['recall']:.4f} | ROC-AUC: {metrics.get('roc_auc', 0):.4f}")

            # Criterio de selección del mejor modelo (priorizar Recall para reducir Falsos Negativos)
            if metrics['recall'] > best_recall:
                best_recall = metrics['recall']
                best_model_uri = model_info.model_uri

    # 6. Registrar el mejor modelo en el MLflow Model Registry
    if best_model_uri:
        registered_model_name = "CustomerChurnModel"
        mlflow.register_model(best_model_uri, registered_model_name)
        print(f"\n✅ Mejor modelo registrado en el Registry: '{registered_model_name}' con Recall = {best_recall:.4f}")

if __name__ == "__main__":
    run_training()