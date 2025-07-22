import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

def load_and_preprocess(
    csv_path: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Loads raw anxiety data, bins the target into classes, encodes features, applies SMOTE to handle class imbalance,
    and splits into train/test sets.

    Parameters
    ----------
    csv_path : str
        Path to the raw CSV file.
    test_size : float, optional
        Proportion of the dataset to include in the test split. Defaults to 0.2.
    random_state : int, optional
        Random state for reproducibility. Defaults to 42.

    Returns
    -------
    X_train, X_test, y_train, y_test : array-like
        Feature matrices and target vectors for training and testing.
    preprocessor : ColumnTransformer
        Fitted preprocessing pipeline for transforming new data.
    smote : SMOTE
        Fitted SMOTE instance for resampling.
    """
    # Load data
    df = pd.read_csv(csv_path)

    # Bin continuous Anxiety Level into categorical classes
    bins = [0, 3, 6, 10]
    labels = ['low', 'medium', 'high']
    df['Anxiety_Class'] = pd.cut(
        df['Anxiety Level (1-10)'],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    # Split features and target
    X = df.drop(columns=['Anxiety Level (1-10)', 'Anxiety_Class'])
    y = df['Anxiety_Class']

    # Identify categorical and numerical columns
    categorical_cols = [
        'Gender', 'Occupation', 'Smoking',
        'Family History of Anxiety', 'Dizziness',
        'Medication', 'Recent Major Life Event'
    ]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    # Create preprocessing pipelines
     numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # Apply preprocessing
    X_processed = preprocessor.fit_transform(X)

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_processed, y)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled,
        y_resampled,
        test_size=test_size,
        random_state=random_state,
        stratify=y_resampled
    )

    return X_train, X_test, y_train, y_test, preprocessor, smote
