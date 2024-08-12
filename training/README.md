# Steps for training
1. Download ```alignment-and-proteins/msa_2022-06-16/2022-06-16_unmasked.fa``` for having sequences and ```variant_surveillance_tsv_2022_06_16/variant_surveillance.tsv``` for having labels of sequences from [GISAID](https://gisaid.org/)
2. Save data in these directories:
    - ```./alignment-and-proteins/msa_2022-06-16/2022-06-16_unmasked.fa```
    - ```./variant_surveillance_tsv_2022_06_16/variant_surveillance.tsv```
3. Create these folders:
```
data --- train
    |
     ---test
    
```
4. To balance data: Run ```python3 data_prep.py```
5. To make train and test dataset: Run ```python3 split_data_train_val_test.py```
6. To train the model: Run ```python3 train.py 64 15```

batch_size = 64, epoches=15.
