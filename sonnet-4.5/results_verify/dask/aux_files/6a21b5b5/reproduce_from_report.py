from multiprocessing import current_process

tracker_pid_wrong = current_process()
child_pid = 12345

print(f"Buggy: {child_pid} != {tracker_pid_wrong}")
print(f"Result: {child_pid != tracker_pid_wrong}")
print(f"Type: int != Process always evaluates to True")

tracker_pid_correct = current_process().pid
print(f"\nCorrect: {child_pid} != {tracker_pid_correct}")
print(f"Result: {child_pid != tracker_pid_correct}")
print(f"Type: int != int correctly compares PIDs")