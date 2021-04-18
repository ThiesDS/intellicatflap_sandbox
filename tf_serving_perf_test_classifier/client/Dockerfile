# Not an official base image, but it runs tf2 on arm architectures
FROM francoisgervais/tensorflow:2.1.0-cp35
LABEL maintainer "Sven Thies <sven_thies@web.de>"

# Install python>3.6 (required for object detection api)
RUN apt-get -qq update && apt-get -qq upgrade \
    && apt-get install -y make build-essential libssl-dev zlib1g-dev
RUN apt-get install -y libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm
RUN apt-get install -y libncurses5-dev  libncursesw5-dev xz-utils tk-dev
RUN wget https://www.python.org/ftp/python/3.6.4/Python-3.6.4.tgz \
    && tar xvf Python-3.6.4.tgz \
    && cd Python-3.6.4 \
    && ./configure --enable-optimizations \
    && make -j8 \
    && make altinstall

RUN python3.6 --version

# Install python and  requirements
COPY requirements.txt .
RUN pip3 install --requirement requirements.txt

# Create app folder
RUN mkdir /app/
WORKDIR /app/

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
