FROM python:3.7-buster
LABEL maintainer "Sven Thies <sven_thies@web.de>"

# Upgrade pip
RUN python -m pip install --upgrade pip

# Create app folder
RUN mkdir /app/
WORKDIR /app/

# Install tensorflow 2 for rpi
RUN pip install https://storage.googleapis.com/tensorflow/raspberrypi/tensorflow-2.3.0-cp35-none-linux_armv7l.whl

# Install rest of requierements
COPY requirements.txt .
RUN pip install --requirement requirements.txt

# Copy app
COPY app .

# Add non-root user
RUN useradd -ms /bin/bash duser
USER duser

# Run app
CMD ["python","make_prediction.py"]
