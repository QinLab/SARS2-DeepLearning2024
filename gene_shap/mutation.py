from collections import Counter

# Function to compare sequences and find mutations 
def find_mutations(reference_seq, sequence):
    mutations = []
    for i in range(len(sequence)):
        if reference_seq[i] != sequence[i] and sequence[i] not in ["-", "n"]:
            mutations.append((i + 1, reference_seq[i], sequence[i]))
    return mutations
    
def find_most_frequent_mutations(self, df, ref_converted_sequence):
    
    # Counter to store the frequency of mutations
    mutation_counter = Counter()

    # Iterate over DataFrame rows and count mutations
    for _, row in df.iterrows():
        sequence = row['sequence']
        mutations = find_mutations(ref_converted_sequence, sequence)
        for mutation in mutations:
            mutation_counter[mutation] += 1

    # Find the mutations with the highest frequency
    most_frequent_mutations = mutation_counter.most_common()

    # Extract position, reference base, mutated base, and frequency as separate lists
    positions = []
    ref_bases = []
    mut_bases = []
    frequencies = []

    for mutation, frequency in most_frequent_mutations:
        position, ref_base, mut_base = mutation
        positions.append(position)
        ref_bases.append(ref_base)
        mut_bases.append(mut_base)
        frequencies.append(frequency)

    return positions, ref_bases, mut_bases, frequencies, most_frequent_mutations
    