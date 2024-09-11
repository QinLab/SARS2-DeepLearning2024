from Bio.Seq import Seq
import re

def translate_with_gaps(seq):
    proteins = []
    for i in range(0, len(seq), 3):
        proteins.append(str(Seq(seq[i:i+3]).translate()))
    return ''.join(proteins) 


def get_protein_genes(gene, df_orfs, converted_sequence):
    start = int(df[df['Gene']==gene]['Start'])
    end = int(df[df['Gene']==gene]['End'])
    protein = translate_with_gaps(converted_sequence[start-1:end])
    return protein


def find_matching_condons_with_single_difference(aa1_input, aa2_input):
    """
    Find and return pairs of 3-letter words from two groups (Condons) where each condon 
    in the pair differs by exactly one letter, along with the position of the difference.
    Position is 1-based.
    """
    matching_pairs = []
    seen_differences = []
    differences = []
    
    for aa1 in aa1_input:        
        for aa2 in aa2_input:            
            for i, (ch1, ch2) in enumerate(zip(aa1, aa2)):
                if ch1 != ch2:
                    differences.append((i, ch1, ch2))
        
                    
    tuple_counts = Counter(differences)
    # Find the most common tuple
    most_common_tuple = tuple_counts.most_common(3)
    print(f"Most common tuple: {most_common_tuple}")

    i, ch1, ch2 = tuple_counts.most_common(1)[0][0]
    seen_differences.append((i, ch1, ch2))
    matching_pairs.append((aa1, aa2, i, ch1, ch2))
    
    most_common_tuple = tuple_counts.most_common(16)
    for a in range(len(most_common_tuple)-1):
        if most_common_tuple[a+1][1] >= 4:
            i, ch1, ch2 = most_common_tuple[a+1][0]
            seen_differences.append((i, ch1, ch2))
            matching_pairs.append((aa1, aa2, i, ch1, ch2))

    return matching_pairs, seen_differences


def get_position_from_codon(codon, ORF, df_ORFs, codontab):
    
    '''
    H49Y mutant is produced by C/T change at position 21,707
    example: get_position_from_codon('H49Y', 'ORF2 (S)')
    output: Most common tuple: [((0, 'C', 'T'), 4), ((2, 'T', 'C'), 1), ((2, 'C', 'T'), 1)]
[21707]
    '''
    
    real_position = []
    if ORF == 'ORF2 (S)' or 'ORF8' or 'ORF1ab' or 'ORF5 (M)' or 'ORF9 (N)':
        characters = [char for char in codon]
        aa1 = characters[0]
        aa2 = characters[-1]

        matching_pairs, seen_differences = find_matching_condons_with_single_difference(codontab[aa1],
                                                                                        codontab[aa2])
        #extract the position number of codon
        integer_match = re.search(r'\d+', codon)
        extracted_integer = int(integer_match.group())

        protein = get_protein_genes(ORF)

        if ORF in list(df_ORFs['Gene']):
            start = int(df_ORFs[df_ORFs['Gene']==ORF]['Start'])
            D = protein[extracted_integer -1]
            start_D = (extracted_integer-1)*3 + start -1
            start_real = start_D + 1

            pos_real_list = []
            for i in range(len(seen_differences)):
                pos_cond = seen_differences[i][0]
                pos_real_list.append(start_real + pos_cond)

            real_position = list(set(pos_real_list))

        return real_position
    
    
def extract_positions(df_mutation):
    """
    Extracts positions and condon,and specified genes from the mutation DataFrame.
    """
    genes = ['ORF2 (S)', 'ORF8', 'ORF1ab', 'ORF5 (M)', 'ORF9 (N)']
    position_spike = []

    for gene in genes:
        for index, row in df_mutation.iterrows():
            if row['Genes'] == gene:
                if row["Nucleotides and Positions"] != 'Unknown':
                    # Extract the integer position from the nucleotides and positions column
                    integer_match = re.search(r'\d+', row["Nucleotides and Positions"])
                    extracted_integer = int(integer_match.group())
                    position_spike.append((row['Nucleotides and Positions'],
                                                 row['Condon'],
                                                 row['Genes'],
                                                 [extracted_integer]))
                else:
                    # Use a fallback function to get the position from the codon
                    pos = get_position_from_codon(row['Condon'], gene)
                    position_spike_alpha.append((row['Nucleotides and Positions'],
                                                 row['Condon'],
                                                 row['Genes'],
                                                 pos))
    return position_spike
