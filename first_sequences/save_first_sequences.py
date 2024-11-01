from Bio import SeqIO
import glob
import constants.constants as CONST
import datetime
import multiprocessing as mp
from multiprocessing import current_process, Value
import os
import time

dir_save = CONST.FRST_DIR
if not os.path.exists(dir_save):
    os.makedirs(dir_save)
    
def save_seq(sequence, filename):
    with open(filename, "w") as handle:
          write_seq = SeqIO.write(sequence, handle, "fasta")


def read_seq(file_path):
    worker_name = current_process().name    
    sequences = SeqIO.parse(file_path, "fasta")    
    
    for sequence in sequences:    
        id_seq = sequence.id.split('|')
        
        if id_seq[0]=='EPI_ISL_601443':
            filename = f"{dir_save}/alpha.fasta"
            save_seq(sequence, filename)
            print(f"{id_seq[0]} is found")

        if id_seq[0]=='EPI_ISL_712073':            
            filename = f"{dir_save}/beta.fasta"
            save_seq(sequence, filename)
            print(f"{id_seq[0]} is found")

        if id_seq[0]=='EPI_ISL_2095177':
            filename = f"{dir_save}/gamma.fasta"
            save_seq(sequence, filename)
            print(f"{id_seq[0]} is found")

        if id_seq[0]=='EPI_ISL_3148365':
            filename = f"{dir_save}/delta.fasta"
            save_seq(sequence, filename)
            print(f"{id_seq[0]} is found")

        if id_seq[0]=='EPI_ISL_6640916':
            filename = f"{dir_save}/omicron.fasta"
            save_seq(sequence, filename)
            print(f"{id_seq[0]} is found")


if __name__ == "__main__":

    start = time.time()
    
    if CONST.NM_CPU == None:
        num_cores = mp.cpu_count()  # Get the number of CPU cores
    else:
        num_cores = CONST.NM_CPU
        
    filenames = glob.glob(f"{CONST.BATCH_DIR}/group_*.fasta")
    with mp.Pool(processes=num_cores) as pool:
        found_sequence = pool.map(read_seq, filenames)
            
    end = time.time()
    
    print(end - start)