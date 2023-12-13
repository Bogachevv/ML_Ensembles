FROM python:3.11-slim
LABEL authors="bogachevv"

COPY requirements.txt /root/Ensembles/requirements.txt
RUN chown -R root:root /root/Ensembles
WORKDIR /root/Ensembles
RUN pip3 install -r requirements.txt

COPY ./src ./
RUN chown -R root:root ./


ENV SECRET_KEY hello
ENV FLASK_APP ml_server

RUN chmod +x run.py

CMD ["python3", "run.py"]