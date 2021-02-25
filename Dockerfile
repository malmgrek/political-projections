FROM python:3.8.0-slim
WORKDIR /code
COPY requirements.txt .
RUN pip install -r requirements.txt
# RUN pip install jupyter
COPY ./ ./
RUN pip install -e .
CMD ["bash", "./bin/run_services.sh"]
