import pandas as pd

def clean_data(df):
    df = df.copy()

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Strip whitespace from string values
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()

    # Drop ID column if it exists
    df.drop(columns=['loan_id'], inplace=True, errors='ignore')

    # Fill missing text columns with most common value
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Fill missing number columns with median
    for col in df.select_dtypes(include='number').columns:
        df[col].fillna(df[col].median(), inplace=True)

    # Encode loan_status explicitly so Approved=1 always
    if 'loan_status' in df.columns:
        df['loan_status'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})

    # Explicit encoding so predictor knows exact mapping
    if 'education' in df.columns:
        df['education'] = df['education'].map({'Graduate': 0, 'Not Graduate': 1})
    if 'self_employed' in df.columns:
        df['self_employed'] = df['self_employed'].map({'No': 0, 'Yes': 1})

    # Encode any remaining binary text columns
    for col in df.select_dtypes(include='object').columns:
        if df[col].nunique() == 2:
            vals = df[col].unique()
            df[col] = df[col].map({vals[0]: 0, vals[1]: 1})

    # One-hot encode remaining text columns
    df = pd.get_dummies(df, drop_first=True)

    return df