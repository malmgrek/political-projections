#!/bin/bash


echo "========================================"
echo "== Setting up development environment =="
echo "========================================"
echo
read -p "Please STOP If your not in Python virtual environment. Stop? [y/n]" -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]
then
    pip install --user -r requirements.txt
    pip install --user jupyter
    pip install --user -e .
fi
