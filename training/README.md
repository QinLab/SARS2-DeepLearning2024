# Train model

## Steps for training
1. Download `alignment-and-proteins/msa_2022-06-16/2022-06-16_unmasked.fa` for having sequences and `variant_surveillance_tsv_2022_06_16/variant_surveillance.tsv` for having labels of sequences from [GISAID](https://gisaid.org/).
2. Save data in these directories:
    - `<root_repository>/GISAID/alignment-and-proteins/msa_2022-06-16/2022-06-16_unmasked.fa`
    - `<root_repository>/GISAID/variant_surveillance_tsv_2022_06_16/variant_surveillance.tsv`
3. Create these folders:
```
data 
│
├── train
└── test   
```
4. Ensure the root directory is in `PYTHONPATH`
```
export PYTHONPATH="<ROOT_DIRECTORY>:$PYTHONPATH"
```
5. To balance data: Run 
```python3 <root_repository>/training/data_prep.py```
6. To make train and test dataset: Run 
```python3 <root_repository>/training/split_data_train_val_test.py```
7. To train the model: Run 
```python3 <root_repository>/training/main.py 64 15```
-batch_size = 64, epoches=15.

## Data curation and preprocessing details

All data were sourced from GISAID on 2022-06-16 using two files: (1) the multiple-sequence alignment (MSA) FASTA file `alignment-and-proteins/msa_2022-06-16/2022-06-16_unmasked.fa` and (2) the metadata file `variant_surveillance_tsv_2022_06_16/variant_surveillance.tsv`. The workflow first filters metadata records to WHO VOC labels (`Alpha`, `Beta`, `Gamma`, `Delta`, `Omicron`) by extracting these names from the `Variant` column and dropping non-matching entries.

To address class imbalance, the pipeline performs class-wise random downsampling at the label table stage using the minority VOC count as the target sample size (`training/data_prep.py`). In this snapshot, the limiting class is Beta (~45k sequences), so each VOC is downsampled to this size before sequence matching. IDs are then matched against the aligned FASTA file, and exact duplicate sequence strings are removed. Because some sampled IDs are not retained after sequence matching/deduplication, the final balanced pool used for model construction is capped by fixed per-class sampling in the next step.

For train/test construction (`training/split_data_train_val_test.py`), the code samples 32,000 sequences per VOC for training (160,000 total) and 8,000 per VOC for test (40,000 total), giving 200,000 sequences overall. A duplicate check is run after concatenating train and test and reports zero cross-split duplicates. Validation data are created from the training portion via the configurable split ratio in `constants/constants.py`.

Alignment handling is inherited directly from the GISAID MSA input: all sequences are already aligned to a common length (29,891 positions), and no additional realignment is performed in this repository. Gap characters (`-`) are preserved as an explicit state during encoding so that insertions/deletions remain model-visible.

For model input, each aligned nucleotide string is converted position-wise into a 7-channel one-hot representation (`one_hot/one_hot.py`) with the fixed symbol order `['-', 'a', 'c', 'g', 'i', 'n', 't']`: canonical bases (A/C/G/T), gap (`-`), ambiguous base (`N`), and an `i` bucket for any non-standard/other character not in the predefined alphabet. Concatenating 29,891 positions yields an input tensor of shape `(N, 29,891, 7)`, where `N` is the number of sequences in the corresponding split.
 

## Summary of our model


| Layer (type)                |Output Shape            |Param  |   
| --------------------------- |:----------------------:|:-----:|
|conv1d (Conv1D)              | (None, 9958, 196)      | 26264 |                                                                    
|max_pooling1d (MaxPooling1D) | (None, 1991, 196)      | 0     |                                                
|conv1d_1 (Conv1D)            | (None, 658, 196)       |730100 |                                                                   
|max_pooling1d_1(MaxPooling1D)| (None, 131, 196)       | 0     |                                              
|flatten (Flatten)            | (None, 25676)          | 0     |           
|dense (Dense)                | (None, 164)            |4211028|                                                        
|dense_1 (Dense)              | (None, 42)             | 6930  |                                                               
|dense_2 (Dense)              | (None, 20)             | 860   |                                                                     
|dropout (Dropout)            | (None, 20)             | 0     |                                                                    
|dense_3 (Dense)              | (None, 5)              | 105   |    
 
- Total params: 4,975,287
- Trainable params: 4,975,287
- Non-trainable params: 0

## Results
* Accuracy in training and validation dataset

![Accuracy](/results/Training_validation_accuracy.jpg)

* Loss in training and validation dataset

![Loss](/results/Training_validation_loss.jpg)
