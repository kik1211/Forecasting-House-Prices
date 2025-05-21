from sklearn.metrics import r2_score
import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

# Load model and data
model = joblib.load('indian_house_model.joblib')
viz_data = joblib.load('viz_data.joblib')

# App title
st.title('🏘️ Indian House Price Predictor')
st.write("Predict house prices in India using Plotly visualizations")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Price vs Area", "Bedrooms", "Locations"])

with tab1:
    st.header('Price Prediction')
    area = st.slider('Select Area (sq ft)', 500, 10000, 1500, 100)
    if st.button('Predict Price'):
        price = model.predict([[area]])[0]
        st.success(f'Predicted Price: ₹{price*100000:,.2f}')
    st.metric("Model R-squared Score", f"{r2_score(viz_data['y_test'], viz_data['y_pred']):.2f}")

with tab2:
    st.header('Area vs Price Relationship')
    
    # Plotly scatter plot
    fig1 = px.scatter(
        x=viz_data['X_test']['Area'],
        y=viz_data['y_test'],
        labels={'x': 'Area (sq ft)', 'y': 'Price'},
        title='Actual Prices'
    )
    st.plotly_chart(fig1)
    
    # Chennai data
    fig2 = px.scatter(
        viz_data['chennai_df'],
        x='Area',
        y='Price',
        title='Chennai Area vs Price'
    )
    st.plotly_chart(fig2)

with tab3:
    st.header('Bedroom Distribution')
    bedroom_df = viz_data['bedroom_counts'].reset_index()
    bedroom_df.columns = ['Bedrooms', 'Count']
    fig3 = px.pie(
        bedroom_df,
        names='Bedrooms',
        values='Count',
        title='Bedroom Distribution in Chennai'
    )
    st.plotly_chart(fig3)

with tab4:
    st.header('House Distribution by Location')
    fig4 = px.bar(
        viz_data['area_df'],
        x='Location',
        y='House Count',
        title='Houses by Location',
        color='House Count'
    )
    fig4.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig4, use_container_width=True)