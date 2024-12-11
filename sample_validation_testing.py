import time
import sys
import random
import math
import csv
from SampleValidationThreads import SampleValidationThreads
from SampleValidationSerial import SampleValidationSerial
from SampleValidationCUDA import SampleValidationCUDA

# import SampleValidationCUDA
num_joints = int(sys.argv[1])
current_obstacles = ["NONE", "SIMPLE", "HARD", "SUPER"]
serial = SampleValidationSerial()
multithreads = SampleValidationThreads()
cuda = SampleValidationCUDA()
modules = [serial, multithreads, cuda]


def trajectory_sample():
    q_rand = [random.uniform(-math.pi, math.pi) for _ in range(num_joints)]
    return q_rand


with open("sample_validation_testing_output.txt", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Module", "Obstacle Type", "Time Taken to Find Valid Sample (s)"])
    for i in range(int(sys.argv[2])):
        for module in modules:
            print("-" * 50)
            random.seed(0xDEADBEEF)
            for obstacle in current_obstacles:
                module.current_obstacle = obstacle

                start_time = time.time()
                valid = False
                while not valid:
                    q_start = trajectory_sample()
                    q_end = trajectory_sample()
                    valid = module.is_segment_valid(q_start, q_end)
                end_time = time.time()

                time_taken = end_time - start_time
                print(
                    f"Module: {module.__class__.__name__}, Obstacle: {obstacle}, Time: {time_taken:.6f} s"
                )
                writer.writerow([module.__class__.__name__, obstacle, time_taken])


# class RRT:
#     def __init__(self):
#         self.current_obstacle = "NONE"

#     def step(self, start, end, step_size):
#         return 1

#     def is_state_valid(self, q_seg):
#         return False

#     def validation_worker(self, i, q_start, q_end, step_size, stop_event):
#         if stop_event.is_set():
#             return
#         q_seg = self.step(q_start, q_end, step_size=i * step_size)
#         if not self.is_state_valid(q_seg):
#             stop_event.set()


#     def is_segment_valid(self, q_start, q_end):

#         threads = []
#         stop_event = threading.Event()

#         if self.current_obstacle == 'NONE':
#             step_size = 0.05
#         elif self.current_obstacle == 'SIMPLE':
#             step_size = 0.02
#         elif self.current_obstacle == 'HARD':
#             step_size = 0.01
#         elif self.current_obstacle == 'SUPER':
#             step_size = 0.005

#         direction = [qr - qn for qr, qn in zip(q_end, q_start)]
#         length = math.sqrt(sum([d**2 for d in direction]))
#         num_segs = int(length / step_size)
#         for i in range(num_segs + 1):
#             thread = threading.Thread(target=self.validation_worker, args=(i, q_start, q_end, step_size, stop_event))
#             threads.append(thread)
#             thread.start()
#             if stop_event.is_set():
#                 return False

#         for thread in threads:
#             thread.join()
#         return not stop_event.is_set()

# a = RRT()
# print(a.is_segment_valid([1], [2]))
