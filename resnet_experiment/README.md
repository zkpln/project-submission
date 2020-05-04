# CSE 4/574 RESNET experiment

A RESNET experiment for CSE 4/574

## Getting Started

### Ubuntu Linux

Make sure pip and venv are installed:
    $ sudo apt install python3-pip python3-venv

Create a new venv:
    $ python3 -m venv env

Activate the new venv (do this every time you want to use it):
    $ source env/bin/activate

Upgrade pip and setuptools (otherwise tensorflow will not install)
    $ pip3 install --upgrade pip
    $ pip3 install --upgrade setuptools

Install the project's required packages into the active venv:
    $ pip3 install -r requirements.txt

Then you should be able to run it:

    $ ./resnet_experiment/main.py

