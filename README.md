# GetAround delay and pricing analysis using Streamlit, MLFlow and FastAPI

The purposes of this repository are :
- analyzing delays and their impact on rentals and delivering a thorough EDA via a dashboarding application such as Streamlit
- analyzing rental fees and training a machine learning model to predict optimal rental price for
- serving this model via an API containing a /predict endpoint

This repositoryb is my Deployment project for Jedha's Machine Learning Engineer certificate.

## Requirements 
- To run EDA notebook : `pip install -r requirements.txt`
- To run Streamlit app : `sh heroku_push.sh` with your own app name in place
- To run MLFlow UI : `sh heroku_push.sh` with your own app name in place <br>
  To train model and log training data on MLFlow server : `python train.py`
- To run FastAPI : `sh heroku_push.sh` with your own app name in place <br>
  To make a prediction request : <br>
  `r = requests.post("https://fastapi-heroku-app-cd532d0f29eb.herokuapp.com/predict", json=dict)` <br>
  where dict is a properly formatted dictionary variable

## Web Dashboard
The EDA dashboard can be viewed [here](https://streamlit-heroku-app-0980a0716d71.herokuapp.com/)

## MLFLow server
The MLFlow UI and registry can be found [here](https://mlflow-heroku-app-014f2f811fa9.herokuapp.com/)

## FastAPI
The FastAPI documentation is available [here](https://fastapi-heroku-app-cd532d0f29eb.herokuapp.com/docs)
