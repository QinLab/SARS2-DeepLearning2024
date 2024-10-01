# Count MSA (Multiple sequence alignment)

**To count MSA in each VOCs, run:**
```
python3 ./msa_gene/msa.py -var <NAME_VARIANT> -num <NUM_SEQ> -thr <FREQ_MUTATION>
```
Replace `<NAME_VARIANT>` with one of the following variant names: Alpha, Beta, Gamma, Delta, or Omicron.

`<NUM_SEQ>` is the number of genetic sequences you want to consider.

`<FREQ_MUTATION>` is the preferred number of frequency of a mutaion.

**Venn Diagram of mutations between VOCs**
Run:
```
python3 ./msa_gene/venn_dna.py -freq <FREQ_MUTATION>
```
`<FREQ_MUTATION>` is the preferred number of frequency of a mutaion.