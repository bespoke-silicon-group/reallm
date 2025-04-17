# ReaLLM:  Trace-Driven Framework for Rapid Simulation of Large-Scale LLM Inference

## Overview

This project provides a simulation pipeline for modeling LLM serving system with customized hardware. The framework supports:

- Kernel generation and performance simulation
    - Generates kernel sizes for the specified model and system
    - Simulates kernel performance using kernel-level simulator such as [LLMCompass](https://github.com/PrincetonUniversity/LLMCompass)
- Trace generation for realistic LLM workloads
    - Generates synthetic traces for code and conversation tasks with specified request rates
    - Based on [Azure LLM inference trace](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md)
- System-level simulation with configurable parallelism and batching strategies
    - Simulates full system performance using traces
    - Supports various parallelism strategies 
        - Tensor parallelism
        - Pipeline parallelism 
        - Context parallelism
        - Expert parallelism (for MoE model)
    - Supports various batching strategies
        - Continuous batching
        - Mixed continuous batching
        - Chunked mixed continuous batching
- Multiple LLM architectures
    - Multi Query Attention + FFN
        - Llama3-70B
        - Llama3-405B
    - Multi-head Latent Attention + MoE
        - DeepSeek-V2
        - DeepSeek-V3


## Python Setup

This infrastructure uses a python virtual environment to manage all of the required packages. To create this virtual environment, simply run `make setup`. This will use the system default python3 interpreter to create the virtual environment, however we check to make sure that the version is 3.10.\*. If the system python3 version is not 3.10.\* then it will throw an error. There are 2 primary ways to fix this: give the makefile the path to a python3.10 interpreter or override the version check. We highly suggest using python3.10 as we cannot ensure the infrastructure is stable on other versions.

To specify a different python interpreter, set the SYSTEM_PYTHON3 variable when you run setup (ie. `make setup SYSTEM_PYTHON3=<path to python3.10>`). This only needs to be set during first time setup, you do not need to set SYSTEM_PYTHON3 after.

To override the version check, you can set the SYSTEM_PYTHON3_VERSION variable to `3.*` when you run setup (ie. `make setup SYSTEM_PYTHON3_VERSION=3.*`). This only needs to be set during first time setup, you do not need to set SYSTEM_PYTHON3_VERSION after.

NOTE: we do not claim support for python versions other than 3.10. Newer version of python are safer than older. However, when using different versions of python, the package versions might also need to be changed. The packages that we install into the virtual environment are found in [requirements.txt](./requirements.txt). These packages also have a specified version. If you are using a different version of python and pip cannot resolve the packages, you can try removing the specific versions for the packages by modifying requirements.txt and removing the version for each package.

To create a new environment with Python 3.10 using conda, run the following commands:
```sh
conda create --name py310 python=3.10
conda activate py10
```

## Quickstart

### Configuration

Simulation parameters are configured through YAML files located in the [configs/system](./configs/system/) directory.
Please refer to the default system configuration [configs/system/default_homo.yaml](./configs/system/default_homo.yaml) for details.

The device is configured through JSON files located in the [configs/device](./configs/device/) directory.


### Using `make`
The easiest way to use the tool is through make, which automatically handles all library dependencies.

#### Basic Usage
```sh
# Run the entire simulation pipeline
make all

# Run kernel generation and simulation
make kernel

# Download and generate traces
make traces

# Run system simulation
make sim
```

#### Advanced Usage
You can set a persistent configuration file that will be used by all subsequent commands:

```sh
# Set a specific configuration file
make set_config CONFIG=configs/system/my_custom_config.yaml

# Check current active configuration
make current_config
```
Once set, all make commands will use this configuration until you specify a different one.

To simulate specific traces:
```sh
# Run simulation with specific trace file
make sim TRACE=workspace/traces/rr_code_3.csv

# Run simulation for specific tasks and requrest rates
make sim TASK="code conv" RATE=5
make sim TASK=code RATE="1 3 5 7"

```