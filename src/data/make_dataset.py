import pandas as pd
from sklearn.model_selection import train_test_split

# Contratos de datos
TARGET = 'Churn'
ID_COL = 'customerID'

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Carga el dataset histórico y realiza limpieza inicial de tipos."""
    df = pd.read_csv(filepath)
    
    # TotalCharges puede contener espacios vacíos; los convertimos a NaN numéricos
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
    return df

def split_data(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    """
    Separa features (X) y target (y), excluyendo el identificador único.
    Realiza partición stratify para preservar la proporción de Churn.
    """
    if TARGET not in df.columns:
        raise ValueError(f"La columna objetivo '{TARGET}' no está en el DataFrame.")
        
    # Mapear target a binario 1/0
    y = df[TARGET].map({'Yes': 1, 'No': 0})
    
    # Excluir Target e ID de las características predictivas
    drop_cols = [c for c in [TARGET, ID_COL] if c in df.columns]
    X = df.drop(columns=drop_cols)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test