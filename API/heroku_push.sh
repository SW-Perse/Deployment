# Build image 
docker build . -t heroku-api-image

# Create heroku app
#heroku create fastapi-heroku-app

# Connect to heroku container service to be able to push a container
heroku container:login

# Tag docker image to heroku app
docker tag heroku-api-image registry.heroku.com/fastapi-heroku-app/web

# Push image to heroku
docker push registry.heroku.com/fastapi-heroku-app/web

# Activate heroku app and run container
heroku container:release web -a fastapi-heroku-app

# Open heroku app
heroku open -a fastapi-heroku-app