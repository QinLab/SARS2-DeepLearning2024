import pandas as pd

'''
reference = https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7245191/#:~:text=The%20genes%20expressed%20by%20SARS%20CoV%2D2%20(NC_045512.2)
'''
data_ORFs = {
    'Number(#)': ["1(7,096)", "1(4,405)", "2(1,273)", "3(275)", "4(75)", "5(222)", "6(61)", "7(121)", "8(43)", "9(121)", "10(419)", "11(38)"],
    'Gene': ["ORF1ab", "ORF1a", "ORF2 (S)", "ORF3a", "ORF4 (E)", "ORF5 (M)", "ORF6", "ORF7a", "ORF7b", "ORF8", "ORF9 (N)", "ORF10"],
    'GeneID': ["43,740,578", "43,740,578", "43,740,568", "43,740,569", "43,740,570", "43,740,571", "43,740,572", "43,740,573", "43,740,574", "43,740,577", "43,740,575", "43,740,576"],
    'Location': ["266–21,555", "266–13,483", "21,563–25,384", "25,393–26,220", "26,245–26,472", "26,523–27,191", "27,202–27,387", "27,394–27,759", "27,756–27,887", "27,894–28,259", "28,274–29,533", "29,558–29,674"],
    'Protein': ["ORF1ab polyprotein", "ORF1a polyprotein", "Spike protein (S protein)", "ORF3a protein", "Envelope protein (E protein)", "Membrane protein (M protein)", "ORF6 protein", "ORF7a protein", "ORF7b protein", "ORF8 protein", "Nucleocapsid phosphoprotein (N protein)", "ORF10 protein"],
    '[LOCUS]': ["[BCB15089.1/BCB97900.1]", "[YP_009725295.1]", "[BCA87361.1]", "[BCA87362.1]", "[BCA87363.1]", "[BCA87364.1]", "[BCA87365.1]", "[BCA87366.1]", "[BCB15096.1]", "[BCA87367.1]", "[BCA87368.1]", "[BCA87369.1]"]
}

df_ORFs = pd.DataFrame(data_ORFs)

# Split the 'Location' column into 'Start' and 'End' columns based on the '–' separator
df_ORFs[['Start', 'End']] = df_ORFs['Location'].str.split('–', expand=True)

# Remove commas and convert 'Start' and 'End' columns to integers
df_ORFs['Start'] = df_ORFs['Start'].str.replace(',', '').astype(int)
df_ORFs['End'] = df_ORFs['End'].str.replace(',', '').astype(int)

# Clean dataframe
df_ORFs = df_ORFs[["Gene", "Protein", "Start", "End"]]

# Save the ORF as csv
df_ORFs.to_csv('./ORF.csv', index=False)
