FROM python:3

WORKDIR /chatapp
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools 

COPY requirements.txt . 
RUN pip install -r requirements.txt
COPY ./app ./app

CMD ["python3", "./main.py", "./train.py"]