"""Model code for subii – Employee Turnover & Early Risk Prediction app."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


DATA_FILE = "HR-Employee-Attrition-New.csv"
DROP_COLS = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]


class IQRCapper(BaseEstimator, TransformerMixin):
    """Caps outliers using IQR bounds learned on training data only."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        num = X.select_dtypes(include="number")

        self.cols_ = [
            c for c in num.columns
            if num[c].nunique() > 10
        ]

        self.lo_, self.hi_ = {}, {}

        for c in self.cols_:
            q1 = X[c].quantile(0.25)
            q3 = X[c].quantile(0.75)
            iqr = q3 - q1

            self.lo_[c] = q1 - 1.5 * iqr
            self.hi_[c] = q3 + 1.5 * iqr

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        for c in self.cols_:
            X[c] = X[c].clip(
                self.lo_[c],
                self.hi_[c]
            )

        return X


def _pre(cat, num):
    return ColumnTransformer([
        (
            "categorical",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ),
            cat
        ),
        (
            "numeric",
            "passthrough",
            num
        )
    ])


def _pipe(model, cat, num, scale=False):
    steps = [
        ("iqr", IQRCapper()),
        ("preprocessor", _pre(cat, num))
    ]

    if scale:
        steps.append(
            ("scaler", StandardScaler())
        )

    return Pipeline(
        steps + [("model", model)]
    )


def load_data(path=DATA_FILE):
    df = pd.read_csv(path)

    return df.drop(
        columns=[
            c for c in DROP_COLS
            if c in df.columns
        ]
    )


def train(path=DATA_FILE):

    df = load_data(path)

    X = df.drop(columns="Attrition")

    y = df["Attrition"].map({
        "No": 0,
        "Yes": 1
    })

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    cat = Xtr.select_dtypes(
        exclude="number"
    ).columns.tolist()

    num = Xtr.select_dtypes(
        include="number"
    ).columns.tolist()


    # -------------------------------------------------
    # Base Models
    # -------------------------------------------------

    base = {

        "RF": _pipe(
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),
            cat,
            num
        ),

        "DT": _pipe(
            DecisionTreeClassifier(
                max_depth=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42
            ),
            cat,
            num
        ),

        "KNN": _pipe(
            KNeighborsClassifier(
                n_neighbors=11,
                weights="distance",
                metric="manhattan"
            ),
            cat,
            num,
            scale=True
        ),

        # Updated Logistic Regression
        # Uses l1_ratio instead of deprecated penalty parameter
        "LR": _pipe(
            LogisticRegression(
                C=0.1,
                l1_ratio=1.0,
                solver="saga",
                class_weight="balanced",
                max_iter=2000,
                random_state=42
            ),
            cat,
            num,
            scale=True
        )
    }


    # -------------------------------------------------
    # Stacking Ensemble
    # -------------------------------------------------

    stack = StackingClassifier(

        estimators=[
            (k.lower(), v)
            for k, v in base.items()
        ],

        final_estimator=LogisticRegression(
            C=1,
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ),

        cv=5,
        passthrough=False,
        n_jobs=1

    ).fit(Xtr, ytr)


    # -------------------------------------------------
    # Predictions
    # -------------------------------------------------

    pred = stack.predict(Xte)

    proba = stack.predict_proba(Xte)[:, 1]


    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------

    metrics = dict(

        Accuracy=accuracy_score(
            yte,
            pred
        ),

        Precision=precision_score(
            yte,
            pred,
            zero_division=0
        ),

        Recall=recall_score(
            yte,
            pred,
            zero_division=0
        ),

        F1=f1_score(
            yte,
            pred
        ),

        ROC_AUC=roc_auc_score(
            yte,
            proba
        )
    )


    # -------------------------------------------------
    # Model Comparison
    # -------------------------------------------------

    names = {
        "rf": "Random Forest",
        "dt": "Decision Tree",
        "knn": "KNN",
        "lr": "Logistic Regression"
    }

    comparison = {

        names[k]: f1_score(
            yte,
            m.predict(Xte)
        )

        for k, m
        in stack.named_estimators_.items()
    }

    comparison["Stacking Ensemble"] = metrics["F1"]


    # -------------------------------------------------
    # Feature Importance
    # -------------------------------------------------

    rf = stack.named_estimators_["rf"]

    imp = pd.DataFrame({

        "Feature":
            rf.named_steps[
                "preprocessor"
            ].get_feature_names_out(),

        "Importance":
            rf.named_steps[
                "model"
            ].feature_importances_

    })

    imp["Feature"] = imp[
        "Feature"
    ].str.replace(
        r"^(categorical|numeric)__",
        "",
        regex=True
    )

    imp = imp.sort_values(
        "Importance",
        ascending=False
    ).head(15)


    # -------------------------------------------------
    # Default Values
    # -------------------------------------------------

    defaults = {

        c: (
            X[c].median()
            if c in num
            else X[c].mode()[0]
        )

        for c in X.columns
    }


    # -------------------------------------------------
    # Return Model Bundle
    # -------------------------------------------------

    return {

        "model": stack,

        "metrics": metrics,

        "cm": confusion_matrix(
            yte,
            pred
        ),

        "comparison": comparison,

        "importance": imp,

        "defaults": defaults,

        "columns": X.columns.tolist(),

        "df": df,

        "income_p25":
            float(
                X["MonthlyIncome"].quantile(0.25)
            )
    }