from Bio import SeqIO
import glob
import pandas as pd
import sars.constants as CONST
from sars.training.data_prep import read_labels


if __name__ == "__main__":
    
    pathes = glob.glob(f"{CONST.FRST_DIR}/*.fasta")

    id_dict = {}
    for path in pathes: 
        sequence = next(SeqIO.parse(path, "fasta"))
        id_seq = sequence.id.split('|')
        id_dict.setdefault(id_seq[0], []).append(str(sequence.seq))

    id_seq = pd.DataFrame({'ID':list(id_dict.keys())
                            ,'sequence':[''.join(seqs) for seqs in id_dict.values()]})

    id_label_map = read_labels(CONST.LABEL_DIR, CONST.VOC_WHO)

    merged_df = pd.merge(id_seq, id_label_map, on='ID')
    merged_df[["ID",
               'sequence', 
               'Variant_VOC']].to_csv(f'{CONST.FRST_DIR}/first_detected.csv')