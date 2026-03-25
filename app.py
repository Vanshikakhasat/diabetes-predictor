import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

st.title("Diabetes Prediction using Decision Tree & KNN")

# ----------------------------------------------------
# Upload Dataset
# ----------------------------------------------------
uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])

if uploaded_file is not None:

    # Load file
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(data.head())

    # ------------------------------------------------
    # Data Cleaning
    # ------------------------------------------------
    data = data.fillna(data.mean(numeric_only=True))
    data = data.select_dtypes(include=['number'])

    st.write("Columns:", data.columns)

    # ------------------------------------------------
    # Auto-detect valid target columns (classification)
    # ------------------------------------------------
    possible_targets = [col for col in data.columns if data[col].nunique() <= 5]

    if len(possible_targets) == 0:
        st.error("No valid classification target found (need column with few unique values like 0/1)")
        st.stop()

    target = st.selectbox("Select Target Column (Auto-filtered)", possible_targets)

    st.write("Selected Target Unique Values:", data[target].unique())

    X = data.drop(target, axis=1)
    y = data[target]

    # ------------------------------------------------
    # Train Test Split
    # ------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ------------------------------------------------
    # Models
    # ------------------------------------------------
    models = {
        "Decision Tree": (
            DecisionTreeClassifier(),
            {"max_depth": [3, 5, 7, None]}
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 7, 9]}
        )
    }

    results = {}
    roc_data = {}

    # ------------------------------------------------
    # Train Models
    # ------------------------------------------------
    if st.button("Train Models"):

        for name, (model, params) in models.items():

            grid = GridSearchCV(model, params, cv=5)
            grid.fit(X_train, y_train)

            best = grid.best_estimator_

            pred = best.predict(X_test)
            prob = best.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, pred)
            results[name] = acc

            fpr, tpr, _ = roc_curve(y_test, prob)
            roc_data[name] = (fpr, tpr)

        # --------------------------------------------
        # Accuracy Chart
        # --------------------------------------------
        st.subheader("Accuracy Comparison")

        fig1, ax1 = plt.subplots()
        ax1.bar(results.keys(), results.values())
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Model Comparison")

        st.pyplot(fig1)

        # --------------------------------------------
        # ROC Curve
        # --------------------------------------------
        st.subheader("ROC Curve")

        fig2, ax2 = plt.subplots()

        for name, (fpr, tpr) in roc_data.items():
            ax2.plot(fpr, tpr, label=name)

        ax2.plot([0, 1], [0, 1], '--')
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.legend()

        st.pyplot(fig2)

        # --------------------------------------------
        # Confusion Matrices
        # --------------------------------------------
        st.subheader("Confusion Matrices")

        for name, (model, params) in models.items():

            best = GridSearchCV(model, params, cv=5).fit(X_train, y_train).best_estimator_
            pred = best.predict(X_test)

            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='d', ax=ax_cm)

            ax_cm.set_title(name)
            
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")

            st.pyplot(fig_cm)