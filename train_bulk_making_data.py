import multiprocessing
import os

exitFlag = 0

class myProcess (multiprocessing.Process):
    def __init__(self, processID, name, start_idx, end_idx):
        multiprocessing.Process.__init__(self)
        self.processID = processID
        self.name = name
        self.start_idx = start_idx
        self.end_idx = end_idx
    def run(self):
        print("start process: " + self.name)
        generate_data(self.name,  self.start_idx, self.end_idx)
        print("exit process: "+ self.name)
def generate_data(processName, idx, end_idx):
    while idx <= end_idx:
        try:
            print(f"{processName} processing index: {idx}")
            os.system(f'python3 making_data_multi.py {idx}')
            idx += 1
        except KeyboardInterrupt:
            print('keyboard catched')
            break
        
process_number = 30
particles_to_be_done = 49954
particles_for_each_process = particles_to_be_done // process_number
print(particles_for_each_process)
processes = []
for i in range(process_number):
    start_idx = i*particles_for_each_process
    end_idx = start_idx+particles_for_each_process - 1
    if i == process_number -1 :
        end_idx = particles_to_be_done -1
    print(i)
    print(start_idx)
    print(end_idx)
    processes.append( myProcess(i, f"Process-{i}", start_idx, end_idx))

for process in processes:
    process.start()
for process in processes:
    process.join()

print("exit")