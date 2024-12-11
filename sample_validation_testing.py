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


def trajectory_sample(seed):
    q_rand = [seed.uniform(-math.pi, math.pi) for _ in range(num_joints)]
    return q_rand


with open("sample_validation_testing_output.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "Module",
            "Obstacle Type",
            "Time Taken to Find Valid Sample (s)",
            "Number of Samples Checked",
        ]
    )
    for i in range(int(sys.argv[2])):
        for module in modules:
            print("-" * 50)
            seeds = [
                random.Random(0xDEADBEEF),
                random.Random(0xBEEFBEEF),
                random.Random(0xDEADBABA),
                random.Random(0xBEEFDADA),
            ]
            for o_i, obstacle in enumerate(current_obstacles):
                module.current_obstacle = obstacle
                num_runs = 0

                start_time = time.time()
                valid = False
                while not valid:
                    q_start = trajectory_sample(seeds[o_i])
                    q_end = trajectory_sample(seeds[o_i])
                    valid = module.is_segment_valid(q_start, q_end)
                    num_runs += 1
                end_time = time.time()

                time_taken = end_time - start_time
                print(
                    f"Module: {module.__class__.__name__}, Obstacle: {obstacle}, Time: {time_taken:.6f} s, Num Runs: {num_runs}"
                )
                writer.writerow(
                    [module.__class__.__name__, obstacle, time_taken, num_runs]
                )
