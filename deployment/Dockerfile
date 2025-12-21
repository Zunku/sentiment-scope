FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

COPY ./requirements.txt /app/requirements.txt
COPY ./app /app/app

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
# Install the app custom utils
RUN pip install /app/app/.


# Remember to consider routes, docker containers have their own file system.
# Inside the /deployment directory:
# docker build -t sentiment-analysis . To build Docker image
# docker run -p 80:80 sentiment-analysis To run the Docker container