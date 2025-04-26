# Install streamlit and pyngrok
!pip install streamlit pyngrok

# Write the Streamlit app code into a file
%%writefile ipl_score_app.py

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
import pickle

def train_model():
    data = {
        'overs': [5, 10, 15, 18, 20, 12, 7, 17],
        'runs': [40, 70, 110, 140, 160, 85, 50, 150],
        'wickets': [1, 2, 3, 5, 4, 2, 1, 6],
        'final_score': [150, 180, 200, 220, 240, 190, 160, 230]
    }
    df = pd.DataFrame(data)
    X = df[['overs', 'runs', 'wickets']]
    y = df['final_score']

    model = RandomForestRegressor()
    model.fit(X, y)

    with open('ipl_score_model.pkl', 'wb') as f:
        pickle.dump(model, f)

def load_model():
    with open('ipl_score_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def predict_score(overs, runs, wickets):
    model = load_model()
    input_df = pd.DataFrame({
        'overs': [overs],
        'runs': [runs],
        'wickets': [wickets]
    })
    prediction = model.predict(input_df)
    return prediction[0]

def main():
    st.title("\ud83c\udfcf IPL Final Score Predictor")
    st.write("Enter the current match status:")

    overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, step=0.1)
    runs = st.number_input("Current Runs", min_value=0)
    wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10)

    if st.button("Predict Final Score"):
        predicted_score = predict_score(overs, runs, wickets)
        st.success(f"\ud83c\udfc6 Predicted Final Score: {int(predicted_score)} runs")

if __name__ == "__main__":
    train_model()
    main()

# Launch the Streamlit app with ngrok
def start_streamlit():
    from pyngrok import ngrok
    !streamlit run ipl_score_app.py &>/content/log.txt &
    public_url = ngrok.connect(port='8501')
    print(f"Streamlit app is live at: {public_url}")

start_streamlit()
