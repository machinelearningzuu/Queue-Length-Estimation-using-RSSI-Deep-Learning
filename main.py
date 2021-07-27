import os

from variables import*
from util import*
from queue_length import QueueLengthEstimation

if not os.path.exists(os.path.join(os.getcwd(),'data/csv_files')):
    os.makedirs(os.path.join(os.getcwd(),'data/csv_files'))

if not os.path.exists(os.path.join(os.getcwd(),'data/weights')):
    os.makedirs(os.path.join(os.getcwd(),'data/weights'))

if not os.path.exists(os.path.join(os.getcwd(),'data/visualization')):
    os.makedirs(os.path.join(os.getcwd(),'data/visualization'))

if __name__ == "__main__":
    queue_len = QueueLengthEstimation()
    queue_len.run_queuelength()