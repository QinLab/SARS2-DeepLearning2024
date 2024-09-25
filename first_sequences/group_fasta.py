import argparse
from Bio import SeqIO
import constants.constants as CONST
import os


arg_parser = argparse.ArgumentParser(description="Split a large FASTA file into smaller batches.")
arg_parser.add_argument("-b", "--batch_size", type=int, default=10000,
                        help="Batch size for splitting the large FASTA file into sub FASTA files")
args = arg_parser.parse_args()
batch_size = args.batch_size

"from https://biopython.org/wiki/Split_large_file"
def batch_iterator(iterator, batch_size):
    """Returns lists of length batch_size.

    This can be used on any iterator, for example to batch up
    SeqRecord objects from Bio.SeqIO.parse(...), or to batch
    Alignment objects from Bio.Align.parse(...), or simply
    lines from a file handle.

    This is a generator function, and it returns lists of the
    entries from the supplied iterator.  Each list will have
    batch_size entries, although the final list may be shorter.
    """
    batch = []
    for entry in iterator:
        batch.append(entry)
        if len(batch) == batch_size:
            yield batch
            batch = []


batch_path = CONST.BATCH_DIR
if not os.path.exists(batch_path):
    os.makedirs(batch_path)
    
record_iter = SeqIO.parse(open(CONST.SEQ_DIR), "fasta")
for i, batch in enumerate(batch_iterator(record_iter, batch_size)):
    filename = f"{batch_path}/group_{i + 1}.fasta"
#     print(f"Creating file: {filename}")
    with open(filename, "w") as handle:
        count = SeqIO.write(batch, handle, "fasta")
        
    print("Wrote %i records to %s" % (count, filename))
    # Check if the file now exists
#     if os.path.exists(filename):
#         print(f"File {filename} successfully created.")
#     else:
#         print(f"Failed to create file {filename}.")
#     break