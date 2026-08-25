import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib


DATA_PATH = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_PATH = "models/attrition_model.pkl"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def show_dashboard(df):
    total = len(df)
    left = (df["Attrition"] == "Yes").sum()
    stayed = (df["Attrition"] == "No").sum()
    attrition_rate = left / total * 100
    avg_salary = df["MonthlyIncome"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Employees", total)
    c2.metric("Left", left)
    c3.metric("Stayed", stayed)
    c4.metric("Attrition", f"{attrition_rate:.2f}%")
    c5.metric("Avg Salary", f"₹{avg_salary:,.0f}")

    st.subheader("Dataset Summary")
    st.dataframe(df.describe())

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    dept = df["Department"].value_counts()
    fig, ax = plt.subplots()
    ax.bar(dept.index, dept.values)
    ax.set_title("Employees by Department")
    st.pyplot(fig)
    plt.close(fig)

    gender = df["Gender"].value_counts()
    fig, ax = plt.subplots()
    ax.bar(gender.index, gender.values)
    ax.set_title("Employees by Gender")
    st.pyplot(fig)
    plt.close(fig)

    attrition = df["Attrition"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(attrition.values, labels=attrition.index, autopct="%1.1f%%")
    ax.set_title("Attrition Distribution")
    st.pyplot(fig)
    plt.close(fig)


def show_analytics(df):
    department = st.selectbox(
        "Department",
        ["All"] + sorted(df["Department"].unique())
    )

    filtered = (
        df if department == "All"
        else df[df["Department"] == department]
    )

    st.dataframe(filtered)

    st.subheader("Top 10 Highest Paid Employees")
    st.dataframe(
        df.sort_values("MonthlyIncome", ascending=False).head(10)
    )

    st.subheader("Attrition by Department")
    st.bar_chart(pd.crosstab(df["Department"], df["Attrition"]))

    st.info(
        "Reduce overtime, improve job satisfaction, review salaries "
        "and improve work-life balance."
    )


def show_prediction(model):
    age = st.number_input("Age", 18, 60, 30)
    salary = st.number_input("Monthly Income", 1000, 50000, 5000)
    distance = st.number_input("Distance From Home", 1, 30, 5)
    job = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
    work = st.selectbox("Work Life Balance", [1, 2, 3, 4])
    years = st.number_input("Years At Company", 0, 40, 5)
    overtime = 1 if st.selectbox("OverTime", ["No", "Yes"]) == "Yes" else 0
    gender = 1 if st.selectbox("Gender", ["Female", "Male"]) == "Male" else 0

    if st.button("Predict"):
        try:
            employee = pd.DataFrame([{
                "Age": age,
                "MonthlyIncome": salary,
                "DistanceFromHome": distance,
                "JobSatisfaction": job,
                "WorkLifeBalance": work,
                "YearsAtCompany": years,
                "OverTime": overtime,
                "Gender": gender
            }])

            prediction = model.predict(employee)[0]
            confidence = max(model.predict_proba(employee)[0]) * 100

            if prediction == 1:
                st.error("⚠ High Risk of Attrition")
            else:
                st.success("✅ Low Risk of Attrition")

            st.info(f"Confidence: {confidence:.2f}%")

        except Exception as error:
            st.error(f"Prediction failed: {error}")


def main():
    st.set_page_config(
        page_title="WorkInsight",
        page_icon="📊",
        layout="wide"
    )

    st.sidebar.title("🏢 WorkInsight")
    st.sidebar.write("AI-Powered HR Decision Intelligence Platform")

    try:
        df = load_data()
        model = load_model()
        st.sidebar.success("✅ Model Loaded")
    except FileNotFoundError as error:
        st.error(f"Required file not found: {error}")
        st.stop()
    except Exception as error:
        st.error(f"Error loading application files: {error}")
        st.stop()

    st.title("📊 WorkInsight")
    st.caption("AI-Powered HR Decision Intelligence Platform")

    tab1, tab2, tab3 = st.tabs(
        ["📊 Dashboard", "📈 Analytics", "🤖 Prediction"]
    )

    with tab1:
        show_dashboard(df)

    with tab2:
        show_analytics(df)

    with tab3:
        show_prediction(model)

    st.caption("© 2026 WorkInsight | Developed by Priya")


if __name__ == "__main__":
    main()
