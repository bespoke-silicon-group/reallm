## Overview

As large language models (LLMs) continue to revolutionize natural language processing tasks, the need for efficient hardware designs becomes essential. However, understanding the intricate relationship between hardware architectures, workloads, and optimization targets remains a challenge. To address this gap, this repository provides a comprehensive hardware evaluation and exploration framework for LLM inference.

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

### Using `make`
The easiest way to use the tool is through make, which automatically handles all library dependencies.

For example, to evaluate TPUv4 running GPT-3, simply type:
```sh
make tpuv4.sw.gpt3
```

To evaluate your custom designs, add the hardware configuration to [inputs/hardware/config](inputs/hardware/config/), the system-level configuration to [inputs/software/system](inputs/software/system/) and the LLM configuration to [inputs/software/model](inputs/software/model/). 
For more details, see [Input Configurations](#input-configurations).

| Make Target  | Explanantion|
|----------------|-|
| `make xx.hw`  | Explore and evaluate the hardware designs `xx` and generate all valid server designs. |
| `make xx.sw.%%`  | Evaluate the end-to-end performance and TCO of hardware `xx` running the model `%%` (e.g. gpt3). This depends on the hardware evaluation flow. |
| `make xx.clean`  | Remove all outputs of hardware `xx`. |

### Running the Main Function
You can also run the main function directly from the command line. For example, to evaluate TPUv4 running GPT-3, run:
```sh
python main.py -hw inputs/hardware/config/tpuv4.yaml -m inputs/software/model/gpt3.yaml
```

To check all options, run
```sh
python main.py -h
```

## Input Configurations

User can specify a wide variety of hardware and software characteristics as inputs to the tool.
[inputs](./inputs/) provides examples, where correspoding input configurations for each directory shown in the figure below.

![Flow](docs/flow.png)

- [inputs/hardware/config](./inputs/hardware/config/) includes hardware architecture configurations. 
- [inputs/hardware/constant](./inputs/hardware/constant/) includes technology node constants and constraints used for guiding hardware evaluation.
- [inputs/software/model](./inputs/software/model/) includes the model specifications of some popular LLMs.
- [inputs/software/system](./inputs/software/system/) includes files that specify the system-level configuration, including mapping and workload.

[hw_example](./inputs/hardware/config/hw_example.yaml) and [sys_default](./inputs/software/system/sys_default.yaml) can be used as templates for hardware archtecture and system configurations.
All configurations can be set to specific values or a list of values to enable large design space exploration.

## Outputs
Results are generated in the `outputs/HW_NAME/` directory, including hardware specs (`HW_NAME.csv` and `HW_NAME.pkl`) and performance of running the model (`MODEL_NAME.csv` and `MODEL_NAME.pkl`).