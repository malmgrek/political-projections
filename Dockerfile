FROM python:3.8
WORKDIR /code
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install jupyter
COPY ./ ./
RUN pip install -e .
CMD ["jupyter", "notebook", "--port", "8888", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
