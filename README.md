# Chiplet Cloud Design Methodology

Chiplet Cloud is a chiplet-based ASIC supercomputer architecture that optimizes total cost of ownership (TCO) per generated token for serving large generative language models to reduce the overall cost to deploy and run these applications in the real world. 

To explore the software-hardware co-design space and perform software mapping-aware Chiplet Cloud optimizations across the architectural design space, we propose a comprehensive design methodology, which is this repository.
This methodology not only accurately explores a spectrum of major design trade-offs in the joint space of hardware and software, but also generates a detailed performance-cost analysis on all valid design points and then outputs the Pareto frontier.


## Phases

* Hardware Exploration

The first phase is the hardware exploration flow which performs a bottom-up, LLM agnostic design space exploration generating thousands of realizable Chiplet Cloud server designs.

To kick off the hardware exploration follow these steps:

1. Create a hardware configuration YAML file in the `configs/hardware` directory. We've provided two examples for reference: cc.yaml for Chiplet Cloud style exploration and tpuv4i.yaml for Google's TPU v4i.
2. Add the hardware target name to the Makefile.
3. Run the command `make <HW_TARGET_NAME>.hw` to initiate the hardware exploration.


* Software Evaluation

The second phase is the software evaluation flow which takes the realizable server design points along with a generative LLM specification to perform software optimized inference simulations and TCO estimations to find Pareto optimal Chiplet Cloud design points.

Create the target model TAML file in the `configs/models` directory and add the model name to the Makefile. We have provided 8 popular models on the market.

Run the command `make <HW_TARGET_NAME>.sw.<SW_MODEL_NAME>` to start the software evaluation.
