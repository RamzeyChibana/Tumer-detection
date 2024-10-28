import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter



def get_regions(image,rows=3,columns=3):

    region_height = image.shape[0] // rows
    region_width = image.shape[1] // columns

    image_height = image.shape[0]
    image_width = image.shape[1]

    regions = []
    for i in range(0,rows):
        for j in range(0,columns):

            row_end = i*region_height+region_height
            col_end = j*region_width+region_width
            
            if i == rows-1 :
                row_end = image_height
            if j == columns-1:
                col_end = image_width
            regions.append(image[i*region_height:row_end,j*region_width:col_end])
    
    return regions




def get_frequency(image:np.array,num_bins):

    image = np.ravel(image)
    bins_range = int(256/num_bins)
    freq = np.bincount(image,minlength=256)

    bin_freq = np.zeros(shape=num_bins)
    for i in range(num_bins-1):
        bin_freq[i] = np.sum(freq[i*bins_range:i*bins_range+bins_range]) # sliding window (window size = bins_range) step = bins_range
    
    i=num_bins-1
    bin_freq[-1]=np.sum(freq[i*bins_range:i*bins_range+bins_range])
    
    return bin_freq






def get_features(regions,num_bins):
    
    features = []
    for region in regions :
        features.append(get_frequency(region,num_bins))
    features = np.concatenate(features)
    return features


if __name__ == "__main__":
    row,col = 4,3
    image = Image.open("Tr-no_0011.jpg").convert("L").resize((214,214))
    image = np.array(image)
    print(image.shape)

    regions = get_regions(image,row,col)
    print(get_features(regions,10).shape)


    fig,axs = plt.subplots(row,col,figsize=(15,15))

    axs = axs.flatten()

    for i,region in enumerate(regions):
        axs[i].imshow(region,cmap="gray")
        axs[i].set_title(region.shape)

    plt.show()



















