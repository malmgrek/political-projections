# TODO: Lightest docker image is Alpine Linux
Map to zero mean and unit variance#
# Millions users: load balancing,
# multiple machines, Kubernetes
#
# - Load balancing:
#   reverse proxy distributes the requests
# - defines resources before hand
# - master node -> worker nodes
# - in principle, scales w.r.t. load
# - does the master proxy know loads of subjects?
# - Google Kubernetes Engine out of the box solution
#
FROM python:3.8
WORKDIR /code
COPY requirements.txt .
RUN pip install -r requirements.txt
# RUN pip install jupyter
COPY ./ ./
RUN pip install -e .
CMD ["bash", "./bin/run_services.sh"]
