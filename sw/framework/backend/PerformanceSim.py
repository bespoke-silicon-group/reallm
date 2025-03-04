import numpy as np
from framework.Pattern import *
from framework.operators import *
from dataclasses import fields
from LLMCompass.design_space_exploration.dse import read_architecture_template, template_to_system
import json, os

_roofline_simulators = {}
def register_roofline_simulator( op_type ):
    def decorator( func ):
        _roofline_simulators[op_type] = func
        return func
    return decorator

_uarch_simulators = {}
def register_uarch_simulator( op_type ):
    def decorator( func ):
        _uarch_simulators[op_type] = func
        return func
    return decorator


class PerformanceSim:
    def __init__( self, network, system, llmcompass_json, ops_to_sim = None):
        self.network = network
        self.system = system
        self.num_nodes = system.num_packages
        self.T = system.default_mapping['t']
        self.P = system.default_mapping['p']
        self.C = system.default_mapping['c']
        self.node_perf = system.server.package.perf
        self.node_bw = system.server.package.dram_bw_TB_per_sec * 1e12
        if ops_to_sim is None:
            self.ops_to_sim = [op.type for op in network.iter()]
            self.ops_to_sim = list(set(self.ops_to_sim))
        else:
            if not isinstance(ops_to_sim, list):
                self.ops_to_sim = [ops_to_sim]
            else:
                self.ops_to_sim = ops_to_sim

        hardware_spec = read_architecture_template(llmcompass_json)
        self.llmcompass_node = template_to_system(hardware_spec)
    
    def run( self, method, saved_results_dir = None, debug = False):
        if saved_results_dir is None:
            saved_results_dir = f"tmp_{method}_sim"
            os.makedirs(saved_results_dir, exist_ok=True)

        # find unique input shapes for each operator
        operators = {}
        for op in self.ops_to_sim:
            operators[op] = {} # key: input shapes str, val: [dict, num_occurrences, Total_latency]
        print(operators)
        for E in self.network.iter():
            if E.type in self.ops_to_sim:
                input_shapes = {}
                saved_results_file_path = os.path.join(saved_results_dir, f"{E.type}.csv")
                for f in fields(E):
                    if "ArgIn" in str(f.type):
                        tensor_id = getattr(E, f.name)
                        T = self.network.lookup_tensor(tensor_id)
                        if T is None:
                            continue
                        input_shapes[f.name] = T.shape
                inputs =  json.dumps(input_shapes, sort_keys=True)
                if inputs not in operators[E.type]:
                    operators[E.type][inputs] = [input_shapes, 0, 0]
                operators[E.type][inputs][1] += 1
        
        total_latency = 0
        for op in operators:
            for input_shapes in operators[op]:
                if method == "uarch":
                    latency = _uarch_simulators[op](self.llmcompass_node.device, operators[op][input_shapes][0], debug=debug)
                elif method == "roofline":
                    latency = _roofline_simulators[op](self.node_perf, self.node_bw, operators[op][input_shapes][0],
                                                       saved_results_file_path=saved_results_file_path, debug=debug)
                operators[op][input_shapes][2] = latency * operators[op][input_shapes][1] 
                total_latency += operators[op][input_shapes][2]
        if debug:
            print("=========================================")
            print(f"Total latency: {total_latency}")
            for op in operators:
                print(f"Operator: {op}")
                for input_shapes in operators[op]:
                    print(f"Input shapes: {input_shapes}, Num occurrences: {operators[op][input_shapes][1]}, Total latency: {operators[op][input_shapes][2]}")
        return total_latency
        
class Performance:
    def __init__(self) -> None:
        # TODO: Implement this later
        pass
    
     
    
    
    