"""
Module to analyze GLMs of widefield Ca2+ data
Lucas Pinto 2020, lucas.pinto@northwestern.edu
"""
#!/usr/bin/env python

# Libraries
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import lp_utils as utils
import deepdish as dd
import analyzeBehavior as behav
import mat73
from   scipy.io import loadmat
from   statsmodels.stats.anova import AnovaRM
from   statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd

wf_params = {
            'glm_comp_file'         : '/Users/lpinto/Dropbox/PintoEtAl2020_subtrial_inact/data/dffGLM_autoregr_model_comp.mat',
            'glm_summ_file'         : '/Users/lpinto/Dropbox/PintoEtAl2020_subtrial_inact/data/glmSummary_ROI_corr_autoRegr.mat',
            'glm_eg_file'           : '/Users/lpinto/Dropbox/PintoEtAl2020_subtrial_inact/data/dffGLM_space_ridge_ROI_autoRegr_ai3_20170201.mat',
            'mouse_eg_id'           : 0,
            'areaList'              : ['V1','mV2','PPC','RSC','mM2','aM2','M1'],
            'area_eg_idx_autoregr'  : [0, 3, 4, 6],
            'area_eg_idx_glm'       : [0, 6, 10, 12],
            'trial_idx_glm'         : 10
            }

# ==============================================================================
# load widefield model data
def load_glm(wf_params=wf_params):

    """
    glm = load_glm(wf_params=wf_params)
    takes optional wf_params dictionary, returns dictionary with GLM summary, comparison and single examples
    loaded from .mat files
    """

    glm                 = dict()
    glm['comp']         = loadmat(wf_params['glm_comp_file'], struct_as_record=True, squeeze_me=True)
    summ                = loadmat(wf_params['glm_summ_file'], struct_as_record=False, squeeze_me=True)
    glm['summ']         = summ['glmSumm']
    eg                  = mat73.loadmat(wf_params['glm_eg_file'])
    glm['eg']           = dict()
    glm['eg']['y']      = eg['dffFit']['bestpred_y'][:-2]
    glm['eg']['yhat']   = eg['dffFit']['bestpred_yhat'][:-2]
    glm['eg']['ROIlbl'] = utils.wf_ROIlbl_translate(eg['dffFit']['ROIlbl'][:-2])

    return glm

# ==============================================================================
# summarize glm results
def summarize_glm(glm=None):

    """
    glm_summ = summarize_glm(glm=None)
    takes glm dictionary (will load if None), returns dictionary with summary
    analysis of model performance and auto-regressive parameters
    """

    if glm is None:
        glm = load_glm()

    # set up
    glm_summ                  = dict()
    mouse_names               = utils.get_mouse_name(glm['comp']['recls'])
    glm_summ['mice']          = np.unique(mouse_names)
    num_mice                  = len(glm_summ['mice'])
    glm_summ['model_names']   = glm['comp']['model_names']
    num_models                = len(glm['comp']['model_names'])
    glm_summ['areaList']      = wf_params['areaList']
    num_areas                 = len(glm_summ['areaList'])
    accuracy                  = glm['comp']['accuracy'][:,:-2,:].flatten()
    glm_summ['accuracy_mice'] = np.zeros((num_mice,num_areas,num_models))
    glm_areas                 = utils.wf_ROIlbl_translate(glm['summ'].ROIlbl[0:-2])

    # compare accuracy
    area_mat  = np.stack([np.stack([glm_areas]*len(mouse_names),axis=0)]*num_models,axis=2).flatten()
    mouse_mat = np.stack([np.stack([mouse_names]*num_areas*2,axis=1)]*num_models,axis=2)[:,:,:,0].flatten()
    for iMouse, mouse in enumerate(glm_summ['mice']):
        for iArea, area in enumerate(glm_summ['areaList']):
            is_area  = area_mat == area
            is_mouse = mouse_mat == mouse
            this_mat = accuracy[np.logical_and(is_area,is_mouse)]
            this_mat = np.reshape(this_mat,(np.sum(np.array(mouse_names) == mouse),2,num_models))
            glm_summ['accuracy_mice'][iMouse,iArea,:] = np.mean(np.mean(this_mat,axis=0),axis=0)

    # remove just correlations model from analysis, irrelevant here
    is_valid                  = np.zeros((num_mice,num_areas,num_models)) == 0
    is_valid[:,:,1]           = False
    glm_summ['accuracy_mice'] = glm_summ['accuracy_mice'][is_valid].reshape(num_mice,num_areas,num_models-1)

    glm_summ['model_names']   = ['Task','Task + auto-regr.','Task + corr + auto-regr.']
    num_models                = len(glm_summ['model_names'])

    glm_summ['accuracy_mean'] = np.mean(glm_summ['accuracy_mice'],axis=0)
    glm_summ['accuracy_sem']  = np.std(glm_summ['accuracy_mice'],axis=0,ddof=1) / np.sqrt(num_mice)

    # stats
    # 2-way ANOVA with RM
    table = pd.DataFrame({'area' : np.repeat(range(num_areas), num_mice*num_models),
                          'mouse': np.tile(np.repeat(range(num_mice), num_models), num_areas),
                          'model': np.tile(range(num_models), num_mice*num_areas),
                          'acc'  : glm_summ['accuracy_mice'].flatten()})
    anova = AnovaRM(data=table, depvar='acc', subject='mouse', within=['area','model']).fit()
    glm_summ['accuracy_anova_table'] = anova.anova_table

    # posthoc comparison
    glm_summ['accuracy_posthoc']       = pairwise_tukeyhsd(glm_summ['accuracy_mice'].flatten(),np.tile(range(num_models), num_mice*num_areas))
    glm_summ['accuracy_posthoc_pvals'] = glm_summ['accuracy_posthoc'].pvalues

    ## timescales

    # extract relevant predictors\
    ispred_idx = np.array([glm['summ'].predLbls[iPred].find('auto') \
                           for iPred in range(np.size(glm['summ'].predLbls))])==0
    num_lags   = np.sum(ispred_idx)
    weights    = glm['summ'].weights[:-2,ispred_idx,:]
    glm_summ['acorr_weights']  = np.zeros((num_areas,num_lags,num_mice))
    glm_summ['taus']           = np.zeros((num_areas,num_mice))
    glm_summ['acorr_lags']     = np.arange(0,num_lags*5,5)
    glm_summ['acorr_xaxis']    = np.arange(0,num_lags*5-5,1)
    glm_summ['acorr_fit_mice'] = np.zeros((num_areas,np.size(glm_summ['acorr_xaxis']),num_mice))

    # average over areas, normalize and fit exponentials
    for iArea, area in enumerate(glm_summ['areaList']):
        is_area = np.array(glm_areas) == area
        glm_summ['acorr_weights'][iArea,:,:] = np.mean(weights[is_area,:,:],axis=0)
        for iMouse in range(num_mice):
            glm_summ['acorr_weights'][iArea,:,iMouse] = glm_summ['acorr_weights'][iArea,:,iMouse] \
                                                        / glm_summ['acorr_weights'][iArea,0,iMouse]
            fit , _   = sp.optimize.curve_fit(exp_decay, glm_summ['acorr_lags'], \
                                              glm_summ['acorr_weights'][iArea,:,iMouse], maxfev=40000)
            glm_summ['taus'][iArea,iMouse]             = fit[0]
            glm_summ['acorr_fit_mice'][iArea,:,iMouse] = exp_decay(glm_summ['acorr_xaxis'],*fit)

    # one-way ANOVA with RM for taus
    table = pd.DataFrame({'area' : np.repeat(range(num_areas), num_mice),
                          'mouse': np.tile(range(num_mice), num_areas),
                          'tau'  : glm_summ['taus'].flatten()})
    glm_summ['taus_anova_table'] = AnovaRM(data=table, depvar='tau', subject='mouse', within=['area']).fit().anova_table
    glm_summ['taus_anova_pval']  = glm_summ['taus_anova_table']['Pr > F'].to_numpy()

    # avg fit
    glm_summ['acorr_weights_mean'] = np.mean(glm_summ['acorr_weights'],axis=2)
    glm_summ['acorr_weights_sem']  = np.std(glm_summ['acorr_weights'],axis=2,ddof=1) / np.sqrt(num_mice)
    glm_summ['acorr_fit_pred']     = np.zeros((num_areas,np.size(glm_summ['acorr_xaxis'])))
    for iArea in range(num_areas):
        fit , _ = sp.optimize.curve_fit(exp_decay, glm_summ['acorr_lags'], \
                                        glm_summ['acorr_weights_mean'][iArea,:], maxfev=2000)
        glm_summ['acorr_fit_pred'][iArea,:] = exp_decay(glm_summ['acorr_xaxis'],*fit)

    return glm_summ

# ==============================================================================
# plot glm timescales
def plot_glm_timescales(glm_summ,glm):

    """
    fig = plot_glm_timescales(glm_summ,glm)
    takes glm and glm_summ dictionaries, plots data summary and examples with emphasis on timescales
    returns figure handle fig
    """

    colors = utils.getAreaColors(glm_summ['areaList'])
    num_areas = len(glm_summ['areaList'])
    fig = plt.figure(figsize=(5,4))

    # plot model performance
    ax = fig.add_subplot(2,4,5)
    for iArea in range(num_areas):
        plt.plot(glm_summ['accuracy_mean'][iArea,:],color=colors[iArea])
    ax.set_xticks(range(4))
    ax.set_xticklabels(glm_summ['model_names'],rotation=90)
    ax.set_ylabel('Cross-val accuracy (r)')
    ax.legend(glm_summ['areaList'],fontsize='x-small',ncol=1)
    ax.set_xlim([-.25,3.25])
    ax = utils.applyPlotDefaults(ax)

    # plot example model predictions
    area_eg_idx  = wf_params['area_eg_idx_glm']
    data_points  = range(wf_params['trial_idx_glm']*60,(wf_params['trial_idx_glm']+1)*60)
    xaxis        = np.arange(0,300,5)
    # subplot      = [2, 3, 9, 10]
    subplot      = [2, 6, 10, 14]

    for iEg in range(len(area_eg_idx)):
        ax = fig.add_subplot(4,4,subplot[iEg])
        thisy = sp.ndimage.gaussian_filter1d(glm['eg']['y'][area_eg_idx[iEg]][data_points], 1)
        ax.plot(xaxis,thisy,linestyle='-',linewidth=.75,color=[.3, .3, .3],label='Data')
        thisy = sp.ndimage.gaussian_filter1d(glm['eg']['yhat'][area_eg_idx[iEg]][data_points], 1)
        thiscl = utils.getAreaColors([glm['eg']['ROIlbl'][area_eg_idx[iEg]]])
        ax.plot(xaxis,thisy,linestyle='-',linewidth=1,color=thiscl[0],label='Prediction')
        ax.set_xticks(range(0,400,100))
        ax.set_title(glm['eg']['ROIlbl'][area_eg_idx[iEg]])

        if iEg == 2:
            ax.set_xlabel('y pos. (cm)')
        if iEg == 3:
            ax.set_ylabel('Df/f (z-score)')

        utils.applyPlotDefaults(ax)

    # plot example exponential fits
    # subplot      = [4, 5, 11, 12]
    subplot      = [3, 7, 11, 15]
    area_eg_idx  = wf_params['area_eg_idx_autoregr']
    for iEg in range(len(area_eg_idx)):
        ax      = fig.add_subplot(4,4,subplot[iEg])
        areaidx = area_eg_idx[iEg]
        thisy   = glm_summ['acorr_weights'][areaidx,:,wf_params['mouse_eg_id']]
        ax.plot(glm_summ['acorr_lags']+5,thisy,linestyle='-',linewidth=.75,color=[.3, .3, .3],label='Data')

        thisy   = glm_summ['acorr_fit_mice'][areaidx,:,wf_params['mouse_eg_id']]
        thiscl  = utils.getAreaColors([glm_summ['areaList'][areaidx]])
        ax.plot(glm_summ['acorr_xaxis']+5,thisy,linestyle='-',linewidth=1,color=thiscl[0],label='Prediction')
        ax.set_xticks(range(0,125,25))
        ax.set_xlim([0,50])
        ax.set_title(glm_summ['areaList'][areaidx])

        # if iEg == 0:
        # ax.legend()
        if iEg == 2:
            ax.set_xlabel('Lag (cm)')
        if iEg == 3:
            ax.set_ylabel('Norm. auto-regr. coeff.')

        utils.applyPlotDefaults(ax)

    # plot average fits
    ax = fig.add_subplot(2,4,4)
    for iArea in range(num_areas):
        plt.plot(glm_summ['acorr_xaxis']+5,glm_summ['acorr_fit_pred'][iArea,:],color=colors[iArea],linewidth=.5)
    ax.set_xlabel('Lag (cm)')
    ax.set_ylabel('Norm. auto-regr. coeff.')
    ax.set_xlim([2,25])
    ax.set_xticks(range(5,30,10))
    ax = utils.applyPlotDefaults(ax)

    # plot taus
    num_mice = np.size(glm_summ['taus'],axis=1)
    ax = fig.add_subplot(2,4,8)
    for iArea in range(num_areas):
        thismean = np.mean(glm_summ['taus'][iArea,:])
        thissem  = np.std(glm_summ['taus'][iArea,:],ddof=1)/np.sqrt(num_mice)
        plt.bar(iArea,thismean,facecolor=colors[iArea])
        plt.errorbar(iArea,thismean,thissem,color=colors[iArea])
    ax.set_xticks(range(num_areas))
    ax.set_xticklabels(glm_summ['areaList'],rotation=90)
    ax.set_ylabel('$\u03C4$ (cm)')
    ax.set_ylim([0, 4])
    ax.text(0,3,'p = {:1.3f}'.format(glm_summ['taus_anova_pval'][0]))
    ax = utils.applyPlotDefaults(ax)

    fig.subplots_adjust(left=.1, bottom=.25, right=.95, top=.75, wspace=.75, hspace=1)

    return fig

# ==============================================================================
# exponential decay function for fits
def exp_decay(x,tau,a,b):
    return b+a*np.exp(-x/tau)

# ==============================================================================
# plot ROI activity predictors
def plot_corr_pred(glm):

    """
    fig = plot_corr_pred(glm)
    takes glm dictionary, plots coupling coefficients
    returns figure handle fig
    """

    # extract relevant predictors
    ispred_idx  = np.array([glm['summ'].predLbls[iPred].find('ROI') \
                           for iPred in range(np.size(glm['summ'].predLbls))])==0
    num_areas   = np.size(glm['summ'].ROIlbl)
    num_mice    = np.size(glm['summ'].weights,axis=2)
    weights     = np.ones((num_areas,num_areas,num_mice)) * np.nan

    # there are no self predictors, fill in accordingly
    for iArea in range(num_areas):
        area_idx                  = np.zeros(num_areas) == 0
        area_idx[iArea]           = False
        weights[iArea,area_idx,:] = glm['summ'].weights[iArea,ispred_idx,:]

    weights  = weights[:-2,:-2,:] # remove somatosensory ctx from analysis
    wmean    = np.mean(weights,axis=2) # avg across mice

    # plot
    fig = plt.figure(figsize=(3,3))
    ax  = fig.gca()
    plt.imshow(wmean,vmin=-0.2,vmax=0.2,cmap='PuOr')
    cb  = plt.colorbar(ax=ax)
    cb.set_label('ROI activity coefficient')
    ax.set_xticks(range(num_areas-2))
    ax.set_xticklabels(glm['summ'].ROIlbl[:-2],rotation=90)
    ax.set_yticks(range(num_areas-2))
    ax.set_yticklabels(glm['summ'].ROIlbl[:-2])

    return fig
