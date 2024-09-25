from collections import Counter
import constants.constants as CONST
import matplotlib
import matplotlib.patches
import matplotlib.pyplot as plt
from msa.utils_mutations import find_mutations
import numpy as np
import os

   
def plot__(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base], 
                                              width=1.5*height, 
                                              height=height/4,
                                              facecolor=color, 
                                              edgecolor=color, 
                                              fill=True))


def plot_a(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
           [0.0, 0.0],
           [0.5, 1.0],
           [0.5, 0.8],
           [0.2, 0.0],
        ]),
        np.array([
           [1.0, 0.0],
           [0.5, 1.0],
           [0.5, 0.8],
           [0.8, 0.0],
        ]),
        np.array([
           [0.225, 0.45],
           [0.775, 0.45],
           [0.85, 0.3],
           [0.15, 0.3],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1,height])[None,:]*polygon_coords
                                                 + np.array([left_edge,base])[None,:]),
                                                facecolor=color, 
                                                edgecolor=color))
        

def plot_c(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], 
                                            width=1.3, 
                                            height=height,
                                            facecolor=color, 
                                            edgecolor=color))
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], 
                                            width=0.7*1.3, 
                                            height=0.7*height,
                                            facecolor='white', 
                                            edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], 
                                              width=1.0, 
                                              height=height,
                                              facecolor='white', 
                                              edgecolor='white', 
                                              fill=True))

    
    
def plot_g(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], 
                                            width=1.3, 
                                            height=height,
                                            facecolor=color, 
                                            edgecolor=color))
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], 
                                            width=0.7*1.3, 
                                            height=0.7*height,
                                            facecolor='white', 
                                            edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], 
                                              width=1.0, 
                                              height=height,
                                              facecolor='white', 
                                              edgecolor='white', 
                                              fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.825, base+0.085*height], 
                                              width=0.174, 
                                              height=0.415*height,
                                              facecolor=color, 
                                              edgecolor=color, 
                                              fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.625, base+0.35*height], 
                                              width=0.374, 
                                              height=0.15*height,
                                              facecolor=color, 
                                              edgecolor=color, 
                                              fill=True))
    
    
    
def plot_i(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base], 
                                              width=0.15*height, 
                                              height=height,
                                              facecolor=color, 
                                              edgecolor=color, 
                                              fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base+0.85*height], 
                                              width=0.15*height, 
                                              height=0.15*height,
                                              facecolor=color, 
                                              edgecolor=color, 
                                              fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base], 
                                              width=0.15*height, 
                                              height=0.15*height,
                                              facecolor=color, 
                                              edgecolor=color, 
                                              fill=True))
    
    
def plot_n(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base], 
                                              width=0.15*height,
                                              height=height,
                                              facecolor=color, 
                                              edgecolor=color, 
                                              fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.85*height, base],
                                              width=0.15*height,
                                              height=height,
                                              facecolor=color,
                                              edgecolor=color,
                                              fill=True))
    ax.add_patch(matplotlib.patches.Polygon(xy=[[left_edge, base+height],
                                                [left_edge+0.85*height, base],
                                                [left_edge+height, base],
                                                [left_edge+0.15*height, base+height]],
                                                facecolor=color, 
                                                edgecolor=color, 
                                                fill=True))
    
def plot_t(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.4, base],
                                              width=0.2, 
                                              height=height, 
                                              facecolor=color, 
                                              edgecolor=color, 
                                              fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base+0.8*height],
                                              width=1.0, 
                                              height=0.2*height, 
                                              facecolor=color, 
                                              edgecolor=color, 
                                              fill=True))
    
    
    
    
default_colors = {0:'purple', 1:'green', 2:'orange', 3:'red', 4:'pink', 5:'olive', 6:'cyan'}
default_plot_funcs = {0:plot__, 1:plot_a, 2:plot_c, 3:plot_g, 4:plot_i, 5:plot_n, 6:plot_t}


"https://github.com/kundajelab/deeplift/blob/master/deeplift/visualization/viz_sequence.py"
class plot_DNA(): 
    def __init__(self, var, as_var, Id, num_shap, ref=None,
                 mut= None, freq= None, indices = None, sorted_shap = None):
        self.var = var
        self.as_var = as_var
        self.Id = Id
        self.ref = ref
        self.mut = mut
        self.freq = freq
        self.indices = indices
        self.num_shap = num_shap
        self.sorted_shap = sorted_shap
    
    def plot_weights_given_ax(self, ax, array,start,
                     height_padding_factor,
                     length_padding,
                     subticks_frequency,
                     highlight,
                     colors=default_colors,
                     plot_funcs=default_plot_funcs):
        assert array.shape[1]==7
        if np.all(array == 0):
            return
        max_pos_height = 0.0
        min_neg_height = 0.0
        heights_at_positions = []
        depths_at_positions = []
        for i in range(array.shape[0]):

            #sort from smallest to highest magnitude
            _acgint_vals = sorted(enumerate(array[i,:]), key=lambda x: abs(x[1]))
            positive_height_so_far = 0.0
            negative_height_so_far = 0.0
            for letter in _acgint_vals:
                plot_func = plot_funcs[letter[0]]
                color=colors[letter[0]]
                if (letter[1] > 0):
                    height_so_far = positive_height_so_far
                    positive_height_so_far += letter[1]                
                else:
                    height_so_far = negative_height_so_far
                    negative_height_so_far += letter[1]
                plot_func(ax=ax, 
                          base=height_so_far, 
                          left_edge=i, 
                          height=letter[1], 
                          color=color)
            max_pos_height = max(max_pos_height, positive_height_so_far)
            min_neg_height = min(min_neg_height, negative_height_so_far)
            heights_at_positions.append(positive_height_so_far)
            depths_at_positions.append(negative_height_so_far)

        #now highlight any desired positions; the key of
        #the highlight dict should be the color
        for color in highlight:
            for start_pos, end_pos in highlight[color]:
                assert start_pos >= 0.0 and end_pos <= array.shape[0]
                min_depth = np.min(depths_at_positions[start_pos:end_pos])
                max_height = np.max(heights_at_positions[start_pos:end_pos])
                ax.add_patch(
                    matplotlib.patches.Rectangle(xy=[start_pos,min_depth],
                        width=end_pos-start_pos,
                        height=max_height-min_depth,
                        edgecolor=color, fill=False))

        ax.set_xlim(-length_padding, array.shape[0]+length_padding)
        ax.xaxis.set_ticks(np.arange(0.0, array.shape[0]+1, subticks_frequency))
        ax.xaxis.set_ticklabels(np.arange(start+1, start+1+array.shape[0]+1, 
                                          subticks_frequency),fontsize=90,rotation=45)

        height_padding = max(abs(min_neg_height)*(height_padding_factor),
                             abs(max_pos_height)*(height_padding_factor))
        ax.set_ylim(min_neg_height-height_padding, max_pos_height+height_padding)
        ax.tick_params(axis='y', labelsize=50)


    def plot_ref_and_sequence_mutation_positions(self, df, shap_values,
                                                 features_test,
                                                 seq_num,
                                                 ref_seq,
                                                 ref_seq_oneHot):

        var_name = CONST.VOC_WHO

        if self.var not in var_name:
            warnings.warn(f"{self.var} is not found in var_name.")

        if self.as_var in var_name:
            index = var_name.index(self.as_var)
        else:
            warnings.warn(f"{self.as_var} is not found in var_name.")
        
        sequence = df['sequence'].values[0]
        mutation = find_mutations(ref_seq, sequence)

        positions = [m[0] for m in mutation]
        ref_bases = [m[1] for m in mutation]
        mut_bases = [m[2] for m in mutation]
        
        positions_array = np.array(positions)

        shap_sum_last_axis = np.sum(shap_values[index],axis=-1)

        dinuc_shuff_explanations = shap_sum_last_axis[:,:,None] * features_test
        array = dinuc_shuff_explanations[seq_num]

        # Get the indices that would sort the array
        sort_indices = np.argsort(shap_sum_last_axis[seq_num])
        # Reverse for descending order
        sort_indices = sort_indices[::-1]

        # Sorting by largest absolute negative SHAP values when a false classes are considered for a sequence
        def sorting_key(index):
            value = shap_sum_last_axis[seq_num][index]
            if value < 0:
                return (abs(value), -value)  # Sort negatives by descending magnitude
            else:
                return (0, 0)  # Non-negatives come last

        if self.var == self.as_var:
            sort_indices = sort_indices

        else:
            sort_indices = sorted(sort_indices, key=sorting_key, reverse=True)
            sort_indices = np.array(sort_indices)



        self.sorted_shap = sort_indices[:self.num_shap]
        print(f"the first {self.num_shap} important SHAPs for {self.var}:{sort_indices[:self.num_shap]+1}")
        # Sort the array
        sorted_array = array[sort_indices]
        
        for i in range(self.num_shap):

            mask = np.isin(positions_array, sort_indices[i]+1)
            contains_true = np.any(mask)

            if contains_true == True:
                # Apply the mask to the lists to get sorted elements
                ref_bases_sorted = np.array(ref_bases)[mask]
                mut_bases_sorted = np.array(mut_bases)[mask]
                
                self.ref = ref_bases_sorted[0]
                self.mut = mut_bases_sorted[0]
                self.indices = sort_indices[i] + 1
            
            else:
                self.ref = None
                self.mut = None
                self.indices = sort_indices[i] + 1

            start = sort_indices[i]-3 + 1 
            end = sort_indices[i]+3 + 1
            array_shap = dinuc_shuff_explanations[0][start:end]
            ref_seq_oneHot_compare = ref_seq_oneHot[start:start+6]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(30, 20))

            # Plot a section of the reference sequence on the first subplot
            self.plot_weights_given_ax(ax=ax1, array=ref_seq_oneHot_compare, start=start,
                                       height_padding_factor=0.05,
                                       length_padding=1,
                                       subticks_frequency=1,
                                       colors=default_colors,
                                       plot_funcs=default_plot_funcs,
                                       highlight={})

            # Plot a section of asserted sequence on the second subplot
            self.plot_weights_given_ax(ax=ax2, array=array_shap, start=start,
                                       height_padding_factor=0.05,
                                       length_padding=1,
                                       subticks_frequency=1,
                                       colors=default_colors,
                                       plot_funcs=default_plot_funcs,
                                       highlight={})
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.3)
            
            directory_path =f'{CONST.RSLT_DIR}/viz_plot/{self.var}'
    
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
                
            plt.savefig(f'{directory_path}/{self.Id}_{self.var}_as_{self.as_var}_{self.indices}.jpg', dpi=80)
            plt.close(fig)

                