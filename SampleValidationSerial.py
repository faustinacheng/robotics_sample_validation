import math
import random
import time


class SampleValidationSerial:

    def __init__(self):
        self.current_obstacle = "NONE"
        random.seed(time.time())

    def is_state_valid(self, q_seg):
        return random.random() > 0.01

    def step(self, q_near, q_rand, step_size=0.1):
        direction = [qr - qn for qr, qn in zip(q_rand, q_near)]
        length = math.sqrt(sum([d**2 for d in direction]))
        if length == 0:
            return q_near
        step = [qn + (d / length) * step_size for qn, d in zip(q_near, direction)]
        # joint limits
        return [max(min(q, math.pi), -math.pi) for q in step]

    def is_segment_valid(self, q_start, q_end):
        if self.current_obstacle == "NONE":
            step_size = 0.05
        elif self.current_obstacle == "SIMPLE":
            step_size = 0.02
        elif self.current_obstacle == "HARD":
            step_size = 0.01
        elif self.current_obstacle == "SUPER":
            step_size = 0.005
        else:
            # default if not specified
            step_size = 0.05

        direction = [qr - qn for qr, qn in zip(q_end, q_start)]
        length = math.sqrt(sum([d**2 for d in direction]))
        num_segs = int(length / step_size)
        for i in range(num_segs + 1):
            q_seg = self.step(q_start, q_end, step_size=i * step_size)
            if not self.is_state_valid(q_seg):
                return False
        return True
