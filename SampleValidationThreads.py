import threading
import math
import random
import time


class SampleValidationThreads:

    def __init__(self):
        self.NUM_THREADS = 2
        self.current_obstacle = "NONE"
        random.seed(time.time())

    def is_state_valid(self, q_seg):
        return random.random() > 0.015

    def validation_worker(
        self, start_index, end_index, q_start, q_end, num_segs, stop_event
    ):
        for i in range(start_index, end_index):
            if stop_event.is_set():
                return
            t = i / num_segs
            q_seg = [qs + t * (qe - qs) for qs, qe in zip(q_start, q_end)]
            if not self.is_state_valid(q_seg):
                stop_event.set()
                return

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
            step_size = 0.05

        direction = [qe - qs for qs, qe in zip(q_start, q_end)]
        length = math.sqrt(sum(d**2 for d in direction))
        if length == 0:
            return self.is_state_valid(q_start)

        num_segs = max(1, int(length / step_size))

        num_threads = min(num_segs, self.NUM_THREADS)
        segment_per_thread = math.ceil(num_segs / num_threads)

        threads = []
        stop_event = threading.Event()

        for t_id in range(num_threads):
            start_index = t_id * segment_per_thread
            end_index = min((t_id + 1) * segment_per_thread, num_segs + 1)
            thread = threading.Thread(
                target=self.validation_worker,
                args=(start_index, end_index, q_start, q_end, num_segs, stop_event),
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return not stop_event.is_set()
