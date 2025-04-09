
from __future__ import annotations

from bsg.framework.Network import *
from bsg.utils.BSGBaseDataclass import BSGBaseDataclass
from typing import Any, List
from dataclasses import dataclass
import numpy as np
import copy


@dataclass
class LayerMemoryProfile( BSGBaseDataclass ):
    layer_name      : str = None
    layer_type      : str = None
    pre_layer_size  : int = None
    mid_layer_size  : int = None
    post_layer_size : int = None
    allocated       : List[str] = None
    deallocated     : List[str] = None



class MemoryProfiler:

    def __init__( self, network, arch ) -> None:
        self.network = network
        self.arch    = arch
        self.alloc   = Allocator(self.network, self.arch)

    def run( self, in_dict ) -> List[LayerMemoryProfile]:
        sinks = copy.deepcopy(self.network.sinknodes)
        
        layers = []

        for k in in_dict.keys():
            self.alloc.alloc(k)

        for E in self.network.iter():
            if E.type == "Param":
                continue
            
            prof = LayerMemoryProfile()
            prof.layer_name = E.id
            prof.layer_type = E.type
            prof.pre_layer_size = self.alloc.size()

            inputs  = [i for i in E.get_inputs().values() if i is not None]
            outputs = [o for o in E.get_output().values() if o is not None]

            for i in inputs:
                assert i in self.alloc, f"{i} not allocated..."
            
            prof.allocated = []
            for o in outputs:
                if self.alloc.alloc(o):
                    prof.allocated.append(o)

            prof.mid_layer_size = self.alloc.size()
            
            prof.deallocated = []
            for i in inputs:
                sinks[i].remove(E.id)
                if len(sinks[i]) == 0:
                    if self.alloc.dealloc(i):
                        prof.deallocated.append(i)

            prof.post_layer_size = self.alloc.size()
            layers.append(prof)
            
        return layers


class Allocator:
    def __init__( self, network, arch ) -> None:
        self.network = network
        self.arch    = arch
        self.memory  = {}

    def __contains__( self, key ) -> bool:
        E : Expr = self.network.lookup_src(key)
        if E and E.type == "Param":
            return True
        return key in self.memory
        
    def alloc( self, key ) -> bool:
        if key in self.memory:
            return False

        E : Expr = self.network.lookup_src(key)
        if E and E.type == "Param":
            return False

        T : Tensor = self.network.lookup_tensor(key)
        self.memory[key] = T.shape
        return True

    def dealloc( self, key ) -> bool:
        if key in self.memory:
            del self.memory[key]
            return True
        return False
    
    def size( self ):
        return sum([np.prod(s) for s in self.memory.values()])
