#!/bin/bash


echo "========================================"
echo "== Setting up development environment =="
echo "========================================"
echo
read -p "Please STOP If your not in Python virtual environment. Stop? [y/n]" -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]
then
    pip install -r requirements.txt
    pip install jupyter
    pip install -e .
fi
