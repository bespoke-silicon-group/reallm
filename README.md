# Chiplet Cloud Design Methodology

Chiplet Cloud is a chiplet-based ASIC supercomputer architecture that optimizes total cost of ownership (TCO) per generated token for serving large generative language models to reduce the overall cost to deploy and run these applications in the real world.

To explore the software-hardware co-design space and perform software mapping-aware Chiplet Cloud optimizations across the architectural design space, we propose a comprehensive design methodology, which is this repository.
This methodology not only accurately explores a spectrum of major design trade-offs in the joint space of hardware and software, but also generates a detailed performance-cost analysis on all valid design points and then outputs the Pareto frontier.

## Python Setup

This infrastructure uses a python virtual environment to manage all of the required packages. To create this virtual environment, simply run `make setup`. This will use the system default python3 interpreter to create the virtual environment, however we check to make sure that the version is 3.10.\*. If the system python3 version is not 3.10.\* then it will throw an error. There are 2 primary ways to fix this: give the makefile the path to a python3.10 interpreter or override the version check. We highly suggest using python3.10 as we cannot ensure the infrastructure is stable on other versions.

To specify a different python interpreter, set the SYSTEM_PYTHON3 variable when you run setup (ie. `make setup SYSTEM_PYTHON3=<path to python3.10>`). This only needs to be set during first time setup, you do not need to set SYSTEM_PYTHON3 after.

To override the version check, you can set the SYSTEM_PYTHON3_VERSION variable to `3.*` when you run setup (ie. `make setup SYSTEM_PYTHON3_VERSION=3.*`). This only needs to be set during first time setup, you do not need to set SYSTEM_PYTHON3_VERSION after.

NOTE: we do not claim support for python versions other than 3.10. Newer version of python are safer than older. However, when using different versions of python, the package versions might also need to be changed. The packages that we install into the virtual environment are found in "requirements.txt". These packages also have a specified version. If you are using a different version of python and pip cannot resolve the packages, you can try removing the specific versions for the packages by modifying requirements.txt and removing the version for each package.

## Phases

* Hardware Exploration

The first phase is the hardware exploration flow which performs a bottom-up, LLM agnostic design space exploration generating thousands of realizable Chiplet Cloud server designs.

* Software Evaluation

The second phase is the software evaluation flow which takes the realizable server design points along with a generative LLM specification to perform software optimized inference simulations and TCO estimations to find Pareto optimal Chiplet Cloud design points.
