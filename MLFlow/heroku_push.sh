# Image build Widows
docker build . -t heroku-mlflow-image

# Login to Heroku
heroku container:login

# Create Heroku app
# heroku create mlflow-heroku-app

# Tag the image
docker tag heroku-mlflow-image registry.heroku.com/mlflow-heroku-app/web

# Push the image
docker push registry.heroku.com/mlflow-heroku-app/web

# Release the image (activate container container)
heroku container:release web -a mlflow-heroku-app

# Open the app
heroku open -a mlflow-heroku-app