import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib import cm


'''
    Change value to rgb
    used as distance field visualization
    return [rr,gg,bb]
'''

def _changeValueToColor(maxValue,minValue,value):
    rr = 0
    gg = 0
    bb = 0
    


    if(value<minValue):
        return [0,0,0]
    
    if((maxValue-minValue)<0.00000001):
        return [0,0,1]
        
    temp = (value-minValue)/(maxValue-minValue)
    
    if(temp>0.75):
        rr =1
        gg = 1-(temp-0.75)/0.25
        if(gg<0):
            gg=0.0
        bb=0
        return [rr,gg,bb]
        
    if(temp>0.5):
        rr = (temp-0.5)/0.25
        gg=1
        bb=0
        return [rr,gg,bb]
    
    if(temp>0.25):
        rr = 0
        gg=1
        bb=1-(temp-0.25)/0.25
        return [rr,gg,bb]
        
    return [0,temp/0.25,1]


def plot_examples(colormaps):
    """
    Helper function to plot data with associated colormap.
    """
    np.random.seed(19680801)
    data = np.random.randn(30, 30)
    n = len(colormaps)
    fig, axs = plt.subplots(1, n, figsize=(n * 2 + 2, 3),
                            constrained_layout=True, squeeze=False)
    for [ax, cmap] in zip(axs.flat, colormaps):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    plt.show()



if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    #colorArray = np.zeros((101,3))

    #indexing 2d-numpy array with[]:
    #And it could modify the array

    #255 0 0    -> red
    #0   0 255  -> blue


    viridis = cm.get_cmap('viridis', 101)
    colorArray = viridis(np.linspace(0, 1, 101))

    
    maxVV = 30 #10
    minVV = 0
    #replace colobar with our values
    for i in range(101):
        ratio = i/100.0
        if(i == 100):
            ratio = 1.0
            
        # rr = 255.0*(1.0- ratio)
        # gg = 0.0
        # bb = 255.0 * ratio
        
        # colorArray[i,0] = rr/255.0
        # colorArray[i,1] = gg/255.0
        # colorArray[i,2] = bb/255.0
        
        [rr,gg,bb] = _changeValueToColor(maxVV,minVV,ratio*(maxVV-minVV)+minVV)
        colorArray[i,0] = rr
        colorArray[i,1] = gg
        colorArray[i,2] = bb

    newcmp = mpl.colors.ListedColormap(colorArray)    
    #plot_examples([newcmp])    
    # print("Color Array:\n",colorArray)

    norm2 = mpl.colors.Normalize(vmin=minVV, vmax=maxVV)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm2, cmap=newcmp),
                  cax=ax, orientation='horizontal', label='Distance error colorbar(unit: mm)')
    plt.show()

    # cm = mpl.cm.cool

    #print('viridis.colors', viridis.colors)

    # ccmap = mpl.colors.Colormap(colorArray)

    # # #cmap = mpl.cm.cool
    # norm2 = mpl.colors.Normalize(vmin=5, vmax=10)

    # fig.colorbar(mpl.cm.ScalarMappable(norm=norm2, cmap=ccmap),
                 # cax=ax, orientation='horizontal', label='Some Units')
                 
    # plt.show()