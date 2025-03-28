from request import Request

class Task:
    def __init__(self, request: Request, arrival_time: int):
        self.req = request
        self.t_arrival = arrival_time

class PrefillTask(Task):
    def __init__(self, request: Request, arrival_time: int):
        super().__init__(request, arrival_time)
        self.n = request.input_len # prompt length

class DecodeTask(Task):
    def __init__(self, request: Request, arrival_time: int, n_cur: int):
        super().__init__(request, arrival_time)
        self.n = request.output_len - 1 # decode length = output length - 1
        # if n_cur > self.n:
        #     raise ValueError("decode length exceeds output length")
        self.n_cur = n_cur
