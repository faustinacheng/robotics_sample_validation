import math
import random
import time


class SampleValidationSerial:

    def __init__(self):
        self.current_obstacle = "NONE"
        random.Random(time.time())

    def is_state_valid(self, q_seg):
        return random.random() > 0.015

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

        direction = [qe - qs for qs, qe in zip(q_start, q_end)]
        length = math.sqrt(sum(d**2 for d in direction))
        if length == 0:
            return self.is_state_valid(q_start)

        num_segs = max(1, int(length / step_size))
        for i in range(num_segs + 1):
            q_seg = [qs + i / num_segs * (qe - qs) for qs, qe in zip(q_start, q_end)]
            if not self.is_state_valid(q_seg):
                return False
        return True
