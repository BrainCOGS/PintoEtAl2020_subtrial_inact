"""
Lucas Pinto's utility functions
"""
#!/usr/bin/env python

import numpy as np
import copy
import matplotlib.pyplot as plt
import re

# ==============================================================================
# false discovery rate
def FDR(pvals,alpha=0.05):

    """
    isSig, alpha_correct = FDR(pvals, alpha)

    performs a false discovery rate correction according to method described
    in Benjamini & Hochberg, 1995, J Royal Stat Soci B, 57(1):289-300 and
    used in Guo et al (2014) Neuron; Pinto et al (2019) Neuron
    Briefly, it ranks p-values in ascending order and defines a value i as being
    significant if it satisfies P(i) <= alpha*i/n where n is the number of
    comparisons
    alpha_correct is FDR-corrected statistical threshold
    isSig is npvals x 1 boolean vector where true indicates significance

    after LP aug 2016
    """

    n       = np.size(pvals)
    pranked = np.sort(pvals)
    idx     = np.argsort(pvals)

    fdr     = (np.arange(1,n+1) * alpha) / n
    isSig   = pranked <= fdr

    if n == 1:
        alpha_correct = alpha;
    else:
        if sum(isSig) > 0:
            alpha_correct = pranked[sum(isSig)-1]
        else:
            alpha_correct = alpha

    isSig   = pvals <= alpha_correct

    return (isSig, alpha_correct)

# ==============================================================================
# rows and columns of figure suplots
def subplot_org(npanels, maxcols=5):

    """
    nrows, ncols = subplot_org(npanels, maxcols=5)

    calculates number of rows and columns in multi-panel figure given the number
    of panels and the max number of desired columns
    """

    nrows = np.ceil(npanels/maxcols)
    ncols = np.ceil(npanels/nrows)

    return (nrows, ncols)

# ==============================================================================
# rows and columns of figure suplots
def applyPlotDefaults(axisHandle,params=None):

    """
    axisHandle = applyPlotDefaults(axisHandle,params=None)

    applies defaults like font size etc
    params is a dictionary containing default values (refer to function body)
    default params is analyzeBehavior.params
    """

    if params is None:
        from analyzeBehavior import params

    # remove right and top spines a la matlab box off
    axisHandle.spines['right'].set_visible(False)
    axisHandle.spines['top'].set_visible(False)

    # fonts
    axisHandle.tick_params(axis='both', which='major', labelsize=params['tick_lbl_size'])
    # for tick in axisHandle.get_xticklabels():
    #     tick.set_fontname(params['plot_font'])
    # for tick in axisHandle.get_yticklabels():
    #     tick.set_fontname(params['plot_font'])

    axisHandle.set_ylabel(axisHandle.get_ylabel(), fontsize=params['axis_lbl_size'])#, fontname=params['plot_font'])
    axisHandle.set_xlabel(axisHandle.get_xlabel(), fontsize=params['axis_lbl_size'])#, fontname=params['plot_font'])
    axisHandle.set_title(axisHandle.get_title(), fontsize=params['title_size'], fontweight='bold')#, fontname=params['plot_font'])

    return axisHandle

# ==============================================================================
# y pos range of laser inactivation given epoch label
def yRangeFromEpochLabel(epoch):

    """
    yrange = yRangeFromEpochLabel(epoch)
    transforms epoch label (epoch, string) into a y pos range
    """

    epoch_list  = ['cueHalf1', 'cueHalf2', 'cueQuart1', 'cueQuart2', 'cueQuart3', 'mem', 'cue', 'whole']
    yrange_list = [[0, 100],[100, 200],[0, 50],[50, 100],[100, 150],[200, 300],[0, 200],[0, 300]]

    idx    = epoch_list.index(epoch)
    yrange = yrange_list[idx]

    return yrange

# ==============================================================================
# plot circles above / below statistically significant data
def plotPvalCircles(axisHandle, x, y, p, isSig = None, where='above',rotate=False,color=[.5, .5, .5]):

    """
    axisHandle = plotPvalCircles(axisHandle, x, y, p, isSig=None, where='above', rotate=False)
    plots circles above / below statistically significant data
    axisHandle
    x: x data
    y: y data (typically accounting for errorbars if any)
    p: p values, determines how many circles
    isSig: whether p vals are significant, if None do p < .05
    where: 'above' or 'below' data points
    rotate: False to spread horizontally, True to do it vertically
    """

    if isSig is None:
        isSig = p < 0.05

    xoffset = (x[-1] - x[0])/12 # offset is an order of magnitude smaller
    yoffset = np.nanmean(abs(y)) / 3
    if where == 'below':
        yoffset = yoffset * -1

    for iPoint in range(np.size(x)):
        if isSig[iPoint] == True:
            if p[iPoint] > .01: # one circle
                thisx = x[iPoint]
                thisy = y[iPoint]+yoffset

            elif p[iPoint] > .001 and p[iPoint] <= .01: # two circles
                if rotate:
                    thisx = [x[iPoint], x[iPoint]]
                    thisy = [y[iPoint]+yoffset, y[iPoint]+yoffset*1.5]
                else:
                    thisx = [x[iPoint]-xoffset/2, x[iPoint]+xoffset/2]
                    thisy = [y[iPoint]+yoffset, y[iPoint]+yoffset]

            elif p[iPoint] <= .001: # three circles
                if rotate:
                    thisx = [x[iPoint], x[iPoint], x[iPoint]]
                    thisy = [y[iPoint]+yoffset, y[iPoint]+yoffset*1.5, y[iPoint]+yoffset*2]
                else:
                    thisx = [x[iPoint]-xoffset, x[iPoint], x[iPoint]+xoffset]
                    thisy = [y[iPoint]+yoffset, y[iPoint]+yoffset, y[iPoint]+yoffset]

            axisHandle.plot(thisx,thisy,'o',linestyle='none',color=color,markersize=2)

    return axisHandle

# ==============================================================================
# get default colors for area
def getAreaColors(area_names):

    """
    colors = getAreaColors(area_names)
    returns list of colors corresponding to areas in list area_names
    """

    area_list = ['V1', 'mV2', 'PPC', 'RSC', 'Post', 'mM2', 'aM2', 'M1', 'Front']
    cmap_name = 'nipy_spectral'
    cmap1     = plt.cm.get_cmap(cmap_name,12)
    cmap2     = plt.cm.get_cmap(cmap_name,16)
    cl        = [None] * len(area_list)
    colors    = [None] * len(area_names)

    for iArea in range(len(area_list)):
        if iArea < 5:
            cl[iArea] = cmap1(iArea+1)
        else:
            cl[iArea] = cmap2(iArea+6)

    for iArea in range(len(area_names)):
        colors[iArea] = cl[area_list.index(area_names[iArea])]

    return colors

# ==============================================================================
# format variable names for figure plotting etc
def formatVarNames(varnames):
    """
    names = formatVarNames(varnames)
    takes list of strings, returns list of strings with capitalization, without underscores etc
    """

    nvars = len(varnames)
    names = [None] * nvars

    # handle abbreviations, underscores, capitalization etc
    for iVar, var in enumerate(varnames):
        thisname    = var
        thisname    = thisname.replace('num','num.')
        thisname    = thisname.capitalize()
        thisname    = thisname.replace('lsr','"laser ON"')
        thisname    = thisname.replace('ctrl','"laser OFF"')
        thisname    = thisname.replace('_',' ')

        names[iVar] = thisname

    return names

# ==============================================================================
# Matlab-like x-corr
def xcorr(x,y,mode='full',maxLags=None,normalization='coeff',returnPositiveOnly=False):
    """
    corr_vec, lags = xcorr(x,y,mode='full',maxLags=None,normalization='coeff',returnPositiveOnly=False)
    takes two vectors and computes cross-correlation using numpy.correlate but adding Matlab-like
    functionality to truncate at maxLag and do normalization. returnPositiveOnly will only return
    positive lags if true, useful for autocorr
    """

    corr_vec = np.correlate(x,y,mode=mode)

    if normalization == 'coeff':
        zero_idx = np.int(np.ceil(np.size(corr_vec)/2))
        corr_vec = corr_vec / corr_vec[zero_idx]

    currLags     = (np.size(corr_vec)-1)/2
    if maxLags is None:
        maxLags  = copy.deepcopy(currLags)

    extra_entries = np.int(currLags-maxLags)
    corr_vec      = corr_vec[range(extra_entries,np.int(np.size(corr_vec)-extra_entries))]
    lags          = np.arange(-maxLags,maxLags+1)

    if returnPositiveOnly:
        idx      = range(maxLags+1,np.size(corr_vec))
        corr_vec = corr_vec[idx]
        lags     = lags[idx]

    return corr_vec, lags

# ==============================================================================
# parse mouse names from list of paths
def get_mouse_name(path_list):

    """
    mouse_names = get_mouse_name(path_list)
    takes list of paths (strings) and returns list of corresponding mouse names
    """

    mouse_names = [None] * len(path_list)
    for iMouse, path in enumerate(path_list):
        str = re.findall('[a-z][a-z][0-9][0-9]',path)
        if str == []:
            str = re.findall('[a-z][a-z][0-9]',path)
        mouse_names[iMouse] = str

    return mouse_names

# ==============================================================================
# translate widefield data nomenclature
def wf_ROIlbl_translate(ROIlbl):

    """
    area_names = wf_ROIlbl_translate(ROIlbl)
    takes list of widefield area name strings and returns list with current naming conventions
    """

    area_names = [None] * len(ROIlbl)
    for iArea, lbl in enumerate(ROIlbl):
        if lbl == 'VISp-R' or lbl == 'VISp-L':
            area_names[iArea] = 'V1'
        elif lbl == 'mV2-R' or lbl == 'mV2-L':
            area_names[iArea] = 'mV2'
        elif lbl == 'VISa-R' or lbl == 'VISa-L':
            area_names[iArea] = 'PPC'
        elif lbl == 'RSP-R' or lbl == 'RSP-L':
            area_names[iArea] = 'RSC'
        elif lbl == 'mMOs-R' or lbl == 'mMOs-L':
            area_names[iArea] = 'mM2'
        elif lbl == 'aMOs-R' or lbl == 'aMOs-L':
            area_names[iArea] = 'aM2'
        elif lbl == 'MOp-R' or lbl == 'MOp-L':
            area_names[iArea] = 'M1'

    return area_names

# ==============================================================================
# calculate fit R2 from data and prediction
def fit_rsquare(data,pred):
    residuals = data - pred
    ss_res    = np.sum(residuals**2)
    ss_tot    = np.sum((data-np.mean(data))**2)
    return 1 - (ss_res / ss_tot)
