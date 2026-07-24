import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------------------
# Navigation Tabs
# -----------------------------------------
tab1, tab2, tab3 = st.tabs(
    [
        "📈 Simple Linear Regression",
        "📊 Multiple Linear Regression",
        "🌳 Decision Tree"
    ]
)

# =====================================================
# TAB 1 - SIMPLE LINEAR REGRESSION
# =====================================================
with tab1:

    # Sample Data
    customers = np.array([20, 25, 30, 35, 40, 45, 50, 55]).reshape(-1, 1)
    sales = np.array([200, 250, 300, 340, 400, 450, 480, 550])

    # Train the model
    model = LinearRegression()
    model.fit(customers, sales)

    st.title("Simple Linear Regression")

    st.subheader("Daily Cafe Sales")

    st.markdown(
        "**Problem Statement:** Build a model to predict sales for 60 customers."
    )

    st.markdown(
        "**Model:** A Simple Linear Regression model is used to predict daily sales based on the number of customers."
    )

    st.write("")

    # User Input
    num_customers = st.number_input(
        "Enter the number of customers:",
        min_value=1,
        max_value=500,
        value=60,
        key="slr_customers"
    )

    predicted_sales = model.predict([[num_customers]])

    st.success(f"Predicted Daily Sales: **${predicted_sales[0]:.2f}**")

    st.write("### Model Information")
    st.write(f"**Slope (Coefficient):** {model.coef_[0]:.2f}")
    st.write(f"**Intercept:** {model.intercept_:.2f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(customers, sales,
               color="blue",
               label="Actual Data")

    ax.plot(customers,
            model.predict(customers),
            color="red",
            label="Regression Line")

    ax.scatter(num_customers,
               predicted_sales[0],
               color="green",
               s=120,
               label="Prediction")

    ax.set_xlabel("Number of Customers")
    ax.set_ylabel("Daily Sales ($)")
    ax.set_title("Customers vs Daily Sales")
    ax.legend()

    st.pyplot(fig)

    # Dataset
    st.write("### Training Dataset")

    data = {
        "Customers": customers.flatten(),
        "Sales ($)": sales
    }

    st.table(data)

    st.markdown("""
    ### Interpretation of Results

    - **Slope (Coefficient):** Indicates how much daily sales increase for every additional customer.
    - **Intercept:** Predicted sales when there are zero customers.
    - **Predicted Daily Sales:** Estimated sales for the entered number of customers.
    - **Blue Dots:** Historical observations.
    - **Red Line:** Best-fit regression line.
    - **Green Dot:** Predicted value.
    """)

# =====================================================
# TAB 2 - MULTIPLE LINEAR REGRESSION
# =====================================================
with tab2:

    st.title("Multiple Linear Regression")

    st.info("🚧 This section will contain the Multiple Linear Regression demo.")

    st.markdown("""
    **Suggested Example**

    Predict house price based on:
    - Area
    - Number of Bedrooms
    - Age of House

    *(You can add your Multiple Linear Regression model here.)*
    """)

# =====================================================
# TAB 3 - DECISION TREE
# =====================================================
with tab3:

    st.title("Decision Tree")

    st.info("🚧 This section will contain the Decision Tree demo.")

    st.markdown("""
    **Suggested Example**

    Predict whether a customer will purchase based on:
    - Age
    - Income
    - Spending Score

    *(You can add your Decision Tree model here.)*
    """)