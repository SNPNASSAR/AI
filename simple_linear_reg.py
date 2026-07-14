
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Sample Data
customers = np.array([20, 25, 30, 35, 40, 45, 50, 55]).reshape(-1, 1)
sales = np.array([200, 250, 300, 340, 400, 450, 480, 550])

# Train the model
model = LinearRegression()
model.fit(customers, sales)


# Streamlit UI
st.markdown("""<p style="text-align:left;font-size:40px;font-weight:bold;color:#1f77b4;margin-bottom:20px;">Daily Cafe Sales</p>""",unsafe_allow_html=True)

st.markdown("<h6> Problem statement: Build a model to predict sales for 60 customers.</h6>", unsafe_allow_html=True)
            
st.markdown("<h6> Model: A simple linear regression model is used in this scenario to predict sales for 60 customers.</h6>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# User input
num_customers = st.number_input("Enter the number of customers:",min_value=1,max_value=500, value=60)

st.markdown("<br>", unsafe_allow_html=True)

predicted_sales = model.predict([[num_customers]])
st.success(f"Predicted Daily Sales: **${predicted_sales[0]:.2f}**")
st.write("### Model Information")
st.write(f"**Slope (Coefficient):** {model.coef_[0]:.2f}")
st.write(f"**Intercept:** {model.intercept_:.2f}")

# Plot
fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(customers, sales, color="blue", label="Actual Data")
ax.plot(customers, model.predict(customers), color="red", label="Regression Line")
ax.scatter(num_customers, predicted_sales[0],
           color="green", s=120, label="Prediction")

ax.set_xlabel("Number of Customers")
ax.set_ylabel("Daily Sales ($)")
ax.set_title("Customers vs Daily Sales")
ax.legend()

st.pyplot(fig)

# Show dataset
st.write("### Training Dataset")

data = {
    "Customers": customers.flatten(),
    "Sales ($)": sales
}

st.table(data)

st.markdown("""
### Interpretation of Results

- **Slope (Coefficient):** Indicates how much the daily sales increase for every additional customer.
- **Intercept:** Represents the predicted sales when there are zero customers. It is mainly a mathematical parameter used to define the regression line.
- **Predicted Daily Sales:** The estimated sales for the number of customers entered by the user.
- **Blue Dots:** Actual historical sales data.
- **Red Line:** The best-fit regression line learned from the historical data.
- **Green Dot:** The predicted sales for the entered number of customers.
""")