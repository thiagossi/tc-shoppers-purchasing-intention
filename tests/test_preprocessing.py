import pandas as pd

from purchase_intent.features.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
)


def _sample_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Administrative": [0, 1, 2],
            "Administrative_Duration": [0.0, 10.5, 20.0],
            "Informational": [0, 0, 1],
            "Informational_Duration": [0.0, 0.0, 5.0],
            "ProductRelated": [1, 2, 3],
            "ProductRelated_Duration": [0.0, 64.0, 120.0],
            "BounceRates": [0.2, 0.0, 0.1],
            "ExitRates": [0.2, 0.1, 0.05],
            "PageValues": [0.0, 0.0, 12.3],
            "SpecialDay": [0.0, 0.0, 0.6],
            "Month": ["Feb", "Mar", "Feb"],
            "VisitorType": ["Returning_Visitor", "New_Visitor", "Returning_Visitor"],
            "Weekend": [False, True, False],
            "OperatingSystems": [1, 2, 1],
            "Browser": [1, 2, 1],
            "Region": [1, 1, 3],
            "TrafficType": [1, 2, 3],
        }
    )


def test_build_preprocessor_transforms_all_declared_features() -> None:
    df = _sample_features()
    assert list(df.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(df)

    expected_categorical_columns = 2 + 2 + 2 + 2 + 2 + 2 + 3
    expected_columns = len(NUMERIC_FEATURES) + expected_categorical_columns
    assert transformed.shape == (3, expected_columns)
