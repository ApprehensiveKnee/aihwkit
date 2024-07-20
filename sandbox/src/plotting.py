# -*- plotting.py -*-

# This file contains the functions to plot the distribution of the weights 
# of the tiles in the model, used specifically to explore the effect
# of the quantization on the weights of the tiles

# -*- coding: utf-8 -*-


import functools

from matplotlib import container
import matplotlib.pyplot as plt
import numpy as np
import torch
from math import log
from IPython.display import HTML



import matplotlib.animation as animation


# -------*-------- WEIGHTS EXTRACTION FUNCTIONS --------*--------

def extract_weights_from_tiles(model: torch.nn.Module, split_by_rows: bool = True):
    # Chek if the model is analog   
    if not hasattr(model, 'analog_tiles'):
        raise ValueError('The model does not have analog tiles')
    analog_tiles = model.analog_tiles()
    weights = {}
    for tile_number, tile in enumerate(analog_tiles):
        tile_weights = tile.get_weights()
        # Split the tensor by rows and store the rows into a list
        if split_by_rows:
            tile_rows = torch.split(tile_weights[0], 1, dim=0)
            weights[tile_number] = [tile_rows, tile_weights[1]]
        else:
            weights[tile_number] = [tile_weights[0]]

    return weights

def feed_to_plot(data, split_by_rows: bool = True):
    if split_by_rows:
        # generate a list of tensors out of the dictionary reuturned by extract_weights_from_tiles
        # for split_by_rows = True
        return [tile[0][rows] for tile in data.values() for rows in range(len(tile[0]))]    
    else:
        # generate a list of tensors out of the dictionary reuturned by extract_weights_from_tiles
        # for split_by_rows = False
        return [tile[0]for tile in data.values()]
    
# -------*-------- WEIGHTS EXTRACTION FUNCTIONS --------*--------

#  -------*-------- HISTOGRAPH FUNCTIONS --------*--------

def plot_tensor_values(tensor: torch.Tensor,bins:int, range: tuple, title: str, name_file: str, gaussian: dict = None, weight_max = None):
    # Create a new figure
    fig, ax = plt.subplots()
    # Create a histogram of the tensor values
    container = ax.hist(tensor.flatten().numpy(), bins=bins,range=range, alpha=0.7, color='darkorange', edgecolor=None)
    ax.set_xlabel('Weight Values')
    ax.set_xlim(left=range[0], right=range[1])
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    

    # if gaussian is provided, the dictionary should contain a list of means and stds.
    # The function will plot the gaussian distribution on top of the histogram
    # NOTE:The gaussian means and stds should be referred to the conductance values to wich the weights are mapped
    if gaussian:
        colors = plt.get_cmap('viridis')(np.linspace(range[0], range[1], len(gaussian['means'])))
        weight_max = max(abs(tensor.flatten().numpy())) if not weight_max else weight_max
        scale = gaussian['gmax'] / weight_max
        
        for i,(mean, std) in enumerate(zip(gaussian['means'], gaussian['stds'])):
            mean = mean / scale
            std = std / scale
            # Compute the bin height over the bins in the range [mean-3*std, mean+3*std]
            bin_range = [mean-0.5*std, mean+0.5*std]
            bin_low = np.digitize(bin_range[0], np.linspace(range[0], range[1], bins+1))
            bin_high = np.digitize(bin_range[1], np.linspace(range[0], range[1], bins+1))
            if bin_low == bin_high:
                gaussian_bins = container[0][bin_low-1]
            else:   
                gaussian_bins = container[0][bin_low-1 : bin_high-1] * np.exp(-0.5 * (( np.linspace(bin_range[0], bin_range[1], len(container[0][bin_low-1 : bin_high-1])) - mean) / std) ** 2)
            mean_height = np.mean(gaussian_bins) if bin_low != bin_high else gaussian_bins
            x = np.linspace(mean - 3*std, mean + 3*std, 100)
            y = (mean_height) * np.exp(-0.5 * ((x - mean) / std) ** 2)
            ax.plot(x, y, color=colors[i], linewidth=0.5, alpha=0.7, linestyle='--'), 
            # Plot the area under the curve
            ax.fill_between(x, y, color = colors[i], alpha=0.3)

    # Save the figure
    if name_file:
        plt.savefig(name_file)
    plt.close()
    # return the figure and axis
    return container, fig

def plot_conductances(tensor:torch.Tensor, bins:int, range: tuple, title: str, name_file: str):
    # Create a new figure
    _, ax = plt.subplots()
    # Create a histogram of the tensor values
    ax.hist(tensor.flatten().numpy(), bins=bins,range=range, alpha=0.7, color='darkorange', edgecolor=None)
    ax.set_xlabel('Conductance Values')
    ax.set_xlim(left=range[0], right=range[1])
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    # Save the figure
    if name_file:
        plt.savefig(name_file)
    plt.close()

def custom_hist(data: list, colors:list, num_bins:int, RANGE: tuple, alpha: float, edgecolor:str, extension: int,title:str, collaterals: dict = None, file_name:str = None):
    # Create a custom histogram used specifically to 
    # plot the distribution of the weights of the tiles
    fig, ax = plt.subplots()
    extension = extension
    one_plus = 1 if extension % 2 == 0 else 0
    # Determine the resolution of the bins
    res = ((RANGE[1]-RANGE[0])*0.5)/(num_bins)
    bins = np.linspace(RANGE[0]*extension,RANGE[1]*extension, num_bins*(extension**2)+ 1 + one_plus )
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bins_levels = [0]* (num_bins*(extension**2) + one_plus)
    
    # Loop over the data and fill the spaces 
    for d,c in zip(data, colors):
        # determine the bin the data point falls into
        bin = np.digitize(d, bins)
        # Fill the bin
        ax.fill_between([bin_centers[bin-1]-res, bin_centers[bin-1]+res], bins_levels[bin], bins_levels[bin]+1, color=c, alpha=alpha, edgecolor=edgecolor)
        bins_levels[bin] += 1

    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.set_xlim((RANGE[0], RANGE[1]))
    ax.set_ylim(bottom=0)
    # Add a box to the top right corner of the plot
    # containing the collaterals values
    if collaterals:
        i = 0
        for key, value in collaterals.items():
            ax.text(RANGE[0]+0.05, max(bins_levels)*0.99 - i*max(bins_levels)*0.05, f'{key}: {value}', fontsize=12, color='darkred')
            i += 1
    if file_name:
        plt.savefig(file_name)
    plt.close() 
    return

def update_hist(num, data, range: tuple, HIST_BINS:int, top: int, title: str):
    # Update the histogram with the new data
    plt.cla()
    n, bins, patches = plt.hist(data[num].flatten().numpy(), HIST_BINS, range = range, color = 'indigo', alpha=0.7 )
    plt.ylim(top=top)
    plt.xlim(left=range[0], right=range[1])
    plt.xlabel('Weight Values')
    plt.ylabel('Frequency')
    max_y = max(n)
    plt.text(range[0] + 0.1, max_y*0.9, f'Frame: {num}', fontsize=12, color='darkred')
    plt.title(title)
    return patches


def plot_hist_animation(datas, HIST_BINS:int,range: tuple , top: int , title:str = 'Distribution of Quantized Weight Values over the tiles', file_name:str = 'animation.gif'):
    fig = plt.figure()
    n, bin, _ =plt.hist(datas[0].flatten().numpy(), HIST_BINS, range = range, color = 'indigo', alpha=0.7 )
    plt.ylim(top=top)
    plt.xlim(left=range[0], right=range[1])
    # Use logaritmic scale for the y-axis
    # set  the x-axis to be fixed
    plt.xlabel('Weight Values')
    plt.ylabel('Frequency')
    # Get the maximum value of the y-axis
    max_y = max(n)
    plt.text(range[0], max_y*0.9, 'Frame: 0', fontsize=12, color='red')
    plt.title(title)
    ani = animation.FuncAnimation(fig, update_hist, len(datas),repeat=True, blit=True, fargs=(datas, range, HIST_BINS, top, title,))
    # Save the animation as a gif
    ani.save(file_name, writer='pillow', fps=1)
    plt.clf()
    return

        
def generate_moving_hist(model: torch.nn.Module,title:str, file_name:str,split_by_rows: bool = True, HIST_BINS:int = 101, range: tuple = (-0.5,0.5), top: int = 30):
    tile_weights = extract_weights_from_tiles(model, split_by_rows=split_by_rows)
    weight_data = feed_to_plot(tile_weights, split_by_rows=split_by_rows)
    # Animate the weights
    plot_hist_animation(weight_data, HIST_BINS=HIST_BINS, title=title, file_name=file_name, range=range, top=top) 

# -------*-------- HISTOGRAPH FUNCTIONS --------*--------

# -------*-------- GENERAL PLOTTING FUNCTIONS --------*--------

''' The following have been taken from the matplotlib documentation'''

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts




# -------*-------- GENERAL PLOTTING FUNCTIONS --------*--------

# -------*-------- UTILITY FUNCTIONS --------*--------

def get_colors(data, cmap):
    # Get the colors associated to the data points
    min_v = min(data)
    max_v = max(data)
    return [cmap((x-min_v)/(max_v-min_v)) for x in data]

# -------*-------- UTILITY FUNCTIONS --------*--------