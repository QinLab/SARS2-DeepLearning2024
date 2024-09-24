# Finding Initial Sequences in Each VOCs

Save the original FASTA file you downloaded from GISAID in a path, and specify this path in `SEQ_DIR` inside `constants/constants.py`.

## 1. Create Subfiles from original file
To create sub-FASTA files from the original file, run:
```
python3 ./first_sequences/group_fasta.py
```
The sub-FASTA files will be saved in the following directory:`decision_tree/batched_sequences/group_<#>.fasta`.

## 2. Save Initial Sequences
First, create a folder named `first_seq`.Then, run:
```
python3 ./first_sequences/save_first_sequences.py
```
The initial sequences from each VOC will be saved in the directory:`decision_tree/first_seq/<Variant_name>.fasta`.

## 3. Label Initial Sequences
To label the initial sequences and save all of them in a single file named `first_seq/first_detected.csv`, run:
```
python3 ./first_sequences/label_first_sequences.py
```
This will save the labeled initial sequences in a consolidated CSV file.