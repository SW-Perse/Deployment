# Image build Widows
 docker build . -t heroku-streamlit-image

# Login to Heroku
heroku container:login

# Create Heroku app
# heroku create streamlit-heroku-app

# Tag the image
 docker tag heroku-streamlit-image registry.heroku.com/streamlit-heroku-app/web

# Push the image
 docker push registry.heroku.com/streamlit-heroku-app/web

# Release the image (activate container container)
 heroku container:release web -a streamlit-heroku-app

# Open the app
heroku open -a streamlit-heroku-app
