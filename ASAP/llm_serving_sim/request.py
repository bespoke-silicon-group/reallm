
from model import Model

class Request:
    def __init__(self, 
                 id: int,
                 model: Model,
                 arrival_time: float,
                 input_len: int,
                 output_len: int):
        self.id = id
        self.model = model
        self.input_len = input_len
        self.output_len = output_len

        self.t_start = arrival_time

        self.t_end = [None] * output_len
        self.current_output = 0

    def __repr__(self):
        return f"Request({self.time}, {self.size})"

    def __lt__(self, other):
        return self.time < other.time

    def __eq__(self, other):
        return self.time == other.time

    def __hash__(self):
        return hash(self.time)

    def output_one_token(self, time: int) -> bool:
        # if self.current_output >= self.output_len:
        #     print(f"Request {self.id} already finished, output {self.current_output} >= {self.output_len}")
        #     raise ValueError("Request already finished")
        self.t_end[self.current_output] = time
        # print(f'request {self.id} output {self.current_output} at time {time}')
        self.current_output += 1
        if self.current_output >= self.output_len:
            # request finished
            return True
        return False