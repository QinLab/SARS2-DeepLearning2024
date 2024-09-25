# Finding Initial Sequences in Each VOCs

Save the original FASTA file you downloaded from GISAID in a path, and specify this path in `SEQ_DIR` inside `constants/constants.py`.

**1. Create Subfiles from original file**
To create sub-FASTA files from the original file, run:
```
python3 <root_repository>/first_sequences/group_fasta.py
```
We saved 10,000 sequences in each sub-FASTA files.

The sub-FASTA files will be saved in the following directory:`<root_repository>/decision_tree/batched_sequences/group_<#>.fasta`.

\* if you want to change the number of sequences saved in each FASTA file, run:
```
python3 <root_repository>/first_sequences/group_fasta.py -b <Your_preferred_number>
```

**2. Save Initial Sequences**
Run:
```
python3 <root_repository>/first_sequences/save_first_sequences.py
```
The initial sequences from each VOC will be saved in the directory:`<root_repository>/decision_tree/first_seq/<Variant_name>.fasta`.

**3. Label Initial Sequences**
To label the initial sequences and save all of them in a single file named `<root_repository>/first_seq/first_detected.csv`, run:
```
python3 <root_repository>/first_sequences/label_first_sequences.py
```
This will save the labeled initial sequences in a consolidated CSV file.