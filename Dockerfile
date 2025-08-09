# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# Install the packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of your application's code into the container
COPY . /code/

# Make port 7860 available to the world outside this container
# Hugging Face Spaces expects the app to run on this port
EXPOSE 7860

# Run the Gunicorn server when the container launches
# It will run the 'app' object from your 'app.py' file
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]