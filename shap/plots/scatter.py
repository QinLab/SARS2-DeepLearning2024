import sars.constants as CONST
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import warnings 


def create_legend(color_map):
    """Create a legend for the scatter plot based on the color map."""
    return [
        plt.Line2D([0], [0], marker='o', color='w', label=region,
                   markerfacecolor=color, markersize=8)
        for region, color in color_map.items()
    ]

def scatter_plot_var(ID, shap_values, var, as_var):
    
    df = pd.read_csv(CONST.ORf_DIR)

    # Create the dictionary using a dictionary comprehension
    regions = {row['Gene']: (row['Start'], row['End']) for _, row in df.iterrows()}

    color_map = {
    'ORF1ab': 'red',
    'ORF1a': 'blue',
    'ORF2 (S)': 'green',
    'ORF3a': 'orange',
    'ORF4 (E)': 'purple',
    'ORF5 (M)': 'cyan',
    'ORF6': 'magenta',
    'ORF7a': 'yellow',
    'ORF7b': 'brown',
    'ORF8': 'pink',
    'ORF9 (N)': 'gray',
    'ORF10': 'black',
}

    var_name = CONST.VOC_WHO
    
    if var not in var_name:
        warnings.warn(f"{as_var} is not found in var_name.")
        return
    
    index = var_name.index(as_var) if as_var in var_name else None
    if index is None:
        warnings.warn(f"{as_var} is not found in var_name.")
        return
    
    shap_scatter = np.sum(shap_values[index],axis=-1)
    indices = np.arange(len(shap_scatter[0]))
        
    # Assign colors based on regions
    colors = []
    for index in indices:
        color_assigned = False
        for region, (start, end) in regions.items():
            if start <= index <= end:
                colors.append(color_map[region])
                color_assigned = True
                break
        if not color_assigned:
            colors.append('black')

    # Create a scatter plot
    plt.figure(figsize=(12, 8))  
    plt.subplot(2, 2, 1)
    plt.scatter(indices, shap_scatter[0], alpha=0.5, color = colors)
    plt.legend(handles=create_legend(color_map), bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.title(f'Scatter Plot of SHAP value for {var} as {as_var}')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.tight_layout()
    
# ORF-specific plots
    orfs = ['ORF2 (S)', 'ORF8', 'ORF9 (N)']
    for i, orf in enumerate(orfs, start=2):
        plt.subplot(2, 2, i)
        plt.scatter(indices, shap_scatter[0], alpha=0.5, color=colors)
        plt.legend(handles=create_legend(color_map), bbox_to_anchor=(1.04, 0.5), loc='center left')
        plt.title(f'Scatter Plot of SHAP value for {var} as {as_var} in {orf}')
        plt.xlabel('Index')
        plt.ylabel('Value')
        start, end = regions[orf]
        plt.xlim(start - 1000, end + 1000) # To compare with its neigbours
        plt.tight_layout()

    # Save all plots to a single JPEG file
    plt.savefig(f'{CONST.SCAT_DIR}/{ID}_{var}_{as_var}_scat.jpg', format='jpg', bbox_inches='tight')
    plt.close()