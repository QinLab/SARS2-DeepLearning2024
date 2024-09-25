"a modified version of https://github.com/tctianchi/pyvenn/blob/master/venn.py"

# coding: utf-8
from itertools import chain, combinations
try:
    # since python 3.10
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import constants.constants as CONST
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
import math
import os

default_colors = [
    # r, g, b, a
    [92, 192, 98, 0.5],
    [90, 155, 212, 0.5],
    [246, 236, 86, 0.6],
    [241, 90, 96, 0.4],
    [255, 117, 0, 0.3],
    [82, 82, 190, 0.2],
]
default_colors = [
    [i[0] / 255.0, i[1] / 255.0, i[2] / 255.0, i[3]]
    for i in default_colors
]

def draw_ellipse(fig, ax, x, y, w, h, a, fillcolor):
    e = patches.Ellipse(
        xy=(x, y),
        width=w,
        height=h,
        angle=a,
        color=fillcolor)
    ax.add_patch(e)

def draw_triangle(fig, ax, x1, y1, x2, y2, x3, y3, fillcolor):
    xy = [
        (x1, y1),
        (x2, y2),
        (x3, y3),
    ]
    polygon = patches.Polygon(
        xy=xy,
        closed=True,
        color=fillcolor)
    ax.add_patch(polygon)

def draw_text(fig, ax, x, y, text, color=[0, 0, 0, 1], fontsize=14, ha="center", va="center"):
    ax.text(
        x, y, text,
        horizontalalignment=ha,
        verticalalignment=va,
        fontsize=fontsize,
        color="black")

def draw_annotate(fig, ax, x, y, textx, texty, text, color=[0, 0, 0, 1], arrowcolor=[0, 0, 0, 0.3]):
    plt.annotate(
        text,
        xy=(x, y),
        xytext=(textx, texty),
        arrowprops=dict(color=arrowcolor, shrink=0, width=0.5, headwidth=8),
        fontsize=14,
        color=color,
        xycoords="data",
        textcoords="data",
        horizontalalignment='center',
        verticalalignment='center'
    )


def all_subsets(lst):
    return chain.from_iterable(combinations(lst, r) for r in range(len(lst)+1))


def get_labels(data, elements=None, fill=["number"]):
    """
    get a dict of labels for groups in data

    @type data: list[Iterable]
    @rtype: dict[str, str]

    input
      data: data to get label for
      elements: list of element identifiers (e.g., ['A', 'B', 'D', 'G', 'O'])
      fill: ["number"|"logic"|"percent"]

    return
      labels: a dict of labels for different sets
    """
    N = len(data)
    sets_data = [set(data[i]) for i in range(N)]  
    s_all = set(chain(*data))  

    if elements is None:
        elements = ['A', 'B', 'C', 'D', 'E'][:N]  

    set_collections = {}

    for r in range(1, N+1):
        for subset in combinations(elements, r):
            key = ''.join(subset)
            included_indices = [elements.index(e) for e in subset]
            value = sets_data[included_indices[0]].copy()

            for idx in included_indices[1:]:
                value &= sets_data[idx]

            set_collections[key] = value

    labels = {k: "" for k in set_collections}
    if "logic" in fill:
        for k in set_collections:
            labels[k] = k + ": "
    if "number" in fill:
        for k in set_collections:
            labels[k] += str(len(set_collections[k]))
    if "percent" in fill:
        data_size = len(s_all)
        for k in set_collections:
            labels[k] += "(%.1f%%)" % (100.0 * len(set_collections[k]) / data_size)

    return labels


def venn5(labels, plot, names=CONST.VOC_WHO, elements=None, **options):
    """
    plots a 5-set Venn diagram

    @type labels: dict[str, str]
    @type names: list[str]
    @type elements: list[str]
    @rtype: (Figure, AxesSubplot)

    input
      labels: a label dict where keys are concatenated element identifiers (e.g., 'A', 'AB', 'ABDGO', ...)
      names:  display names for the groups (e.g., ['Alpha', 'Beta', 'Gamma', 'Delta', 'Omicron'])
      elements: internal identifiers used for computations (e.g., ['A', 'B', 'D', 'G', 'O'])
      more:   colors, figsize, dpi, fontsize

    return
      pyplot Figure and AxesSubplot object
    """
    colors = options.get('colors', [default_colors[i] for i in range(5)])
    figsize = options.get('figsize', (13, 13))
    dpi = options.get('dpi', 96)
    fontsize = options.get('fontsize', 14)

    if elements is None:
        elements = ['A', 'B', 'C', 'D', 'E'][:len(names)]  # default identifiers

    N = len(elements)

    fig = plt.figure(0, figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_axis_off()
    ax.set_ylim(bottom=0.0, top=1.0)
    ax.set_xlim(left=0.0, right=1.0)

    draw_ellipse(fig, ax, 0.428, 0.449, 0.87, 0.50, 155.0, colors[0])
    draw_ellipse(fig, ax, 0.469, 0.543, 0.87, 0.50, 82.0, colors[1])
    draw_ellipse(fig, ax, 0.558, 0.523, 0.87, 0.50, 10.0, colors[2])
    draw_ellipse(fig, ax, 0.578, 0.432, 0.87, 0.50, 118.0, colors[3])
    draw_ellipse(fig, ax, 0.489, 0.383, 0.87, 0.50, 46.0, colors[4])

    binary_to_element_key = {}
    for n in range(1, 2**N):
        binary_key = bin(n).split('0b')[-1].zfill(N)
        element_key = ''.join([elements[i] for i in range(N) if binary_key[i]=='1'])
        binary_to_element_key[binary_key] = element_key

    positions = [
        (0.27, 0.11, '00001'),
        (0.72, 0.11, '00010'),
        (0.55, 0.13, '00011'),
        (0.91, 0.58, '00100'),
        (0.78, 0.64, '00101'),
        (0.84, 0.41, '00110'),
        (0.76, 0.55, '00111'),
        (0.51, 0.90, '01000'),
        (0.39, 0.15, '01001'),
        (0.42, 0.78, '01010'),
        (0.50, 0.15, '01011'),
        (0.67, 0.76, '01100'),
        (0.70, 0.71, '01101'),
        (0.51, 0.74, '01110'),
        (0.64, 0.67, '01111'),
        (0.10, 0.61, '10000'),
        (0.20, 0.31, '10001'),
        (0.76, 0.25, '10010'),
        (0.65, 0.23, '10011'),
        (0.18, 0.50, '10100'),
        (0.21, 0.37, '10101'),
        (0.81, 0.37, '10110'),
        (0.74, 0.40, '10111'),
        (0.27, 0.70, '11000'),
        (0.34, 0.25, '11001'),
        (0.33, 0.72, '11010'),
        (0.51, 0.22, '11011'),
        (0.25, 0.58, '11100'),
        (0.28, 0.39, '11101'),
        (0.36, 0.66, '11110'),
        (0.51, 0.47, '11111'),
    ]

    for x, y, binary_key in positions:
        element_key = binary_to_element_key.get(binary_key, '')
        label_text = labels.get(element_key, '')
        draw_text(fig, ax, x, y, label_text, fontsize=fontsize)

    draw_text(fig, ax, 0.02, 0.72, names[0], colors[0], fontsize=fontsize, ha="right")
    draw_text(fig, ax, 0.72, 0.94, names[1], colors[1], fontsize=fontsize, va="bottom")
    draw_text(fig, ax, 0.97, 0.74, names[2], colors[2], fontsize=fontsize, ha="left")
    draw_text(fig, ax, 0.88, 0.05, names[3], colors[3], fontsize=fontsize, ha="left")
    draw_text(fig, ax, 0.12, 0.05, names[4], colors[4], fontsize=fontsize, ha="right")

    leg = ax.legend(names, loc='center left', bbox_to_anchor=(1.0, 0.5), fancybox=True)
    leg.get_frame().set_alpha(0.5)
    
    directory_path = f'{CONST.RSLT_DIR}/venn_plot'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        
    plt.savefig(f'{directory_path}/venn_diagram_{plot}.png', format='png', dpi=100, bbox_inches='tight')
    return fig, ax
