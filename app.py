
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="WorkInsight", page_icon="📊", layout="wide")
st.sidebar.title("🏢 WorkInsight")
st.sidebar.write("AI-Powered HR Decision Intelligence Platform")
st.sidebar.success("✅ Model Loaded")

try:
    df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    model = joblib.load("models/attrition_model.pkl")
except FileNotFoundError as e:
    st.error(f"Required file not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading application files: {e}")
    st.stop()

total=len(df)
left=(df["Attrition"]=="Yes").sum()
stayed=(df["Attrition"]=="No").sum()
attrition_rate=left/total*100
avg_salary=df["MonthlyIncome"].mean()

st.title("📊 WorkInsight")
st.caption("AI-Powered HR Decision Intelligence Platform")

tab1,tab2,tab3=st.tabs(["📊 Dashboard","📈 Analytics","🤖 Prediction"])

with tab1:
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Employees",total)
    c2.metric("Left",left)
    c3.metric("Stayed",stayed)
    c4.metric("Attrition",f"{attrition_rate:.2f}%")
    c5.metric("Avg Salary",f"₹{avg_salary:,.0f}")
    st.subheader("Dataset Summary")
    st.dataframe(df.describe())
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    dept=df["Department"].value_counts()
    fig,ax=plt.subplots()
    ax.bar(dept.index,dept.values)
    st.pyplot(fig)
    gender=df["Gender"].value_counts()
    fig,ax=plt.subplots()
    ax.bar(gender.index,gender.values)
    st.pyplot(fig)
    attr=df["Attrition"].value_counts()
    fig,ax=plt.subplots()
    ax.pie(attr.values,labels=attr.index,autopct="%1.1f%%")
    st.pyplot(fig)

with tab2:
    dep=st.selectbox("Department",["All"]+sorted(df["Department"].unique()))
    filtered=df if dep=="All" else df[df["Department"]==dep]
    st.dataframe(filtered)
    st.subheader("Top 10 Highest Paid Employees")
    st.dataframe(df.sort_values("MonthlyIncome",ascending=False).head(10))
    st.subheader("Attrition by Department")
    st.bar_chart(pd.crosstab(df["Department"],df["Attrition"]))
    st.info("Reduce overtime, improve job satisfaction, review salaries and improve work-life balance.")

with tab3:
    age=st.number_input("Age",18,60,30)
    salary=st.number_input("Monthly Income",1000,50000,5000)
    distance=st.number_input("Distance From Home",1,30,5)
    job=st.selectbox("Job Satisfaction",[1,2,3,4])
    work=st.selectbox("Work Life Balance",[1,2,3,4])
    years=st.number_input("Years At Company",0,40,5)
    overtime=1 if st.selectbox("OverTime",["No","Yes"])=="Yes" else 0
    gender=1 if st.selectbox("Gender",["Female","Male"])=="Male" else 0
    if st.button("Predict"):
        X=[[age,salary,distance,job,work,years,overtime,gender]]
        pred=model.predict(X)[0]
        conf=max(model.predict_proba(X)[0])*100
        if pred==1:
            st.error("⚠ High Risk of Attrition")
        else:
            st.success("✅ Low Risk of Attrition")
        st.info(f"Confidence: {conf:.2f}%")

st.caption("© 2026 WorkInsight | Developed by Priya")
