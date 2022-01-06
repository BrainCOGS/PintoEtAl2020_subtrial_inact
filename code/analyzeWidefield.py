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
import pingouin as pg
import pandas as pd

wf_params = {
            'glm_comp_file'         : '/Users/lpinto/Dropbox/PintoEtAl2020_subtrial_inact/data/dffGLM_autoregr_model_comp_time.mat',
            'glm_summ_file'         : '/Users/lpinto/Dropbox/PintoEtAl2020_subtrial_inact/data/glmSummary_ROI_corr_autoRegr_time.mat',
            'glm_eg_file'           : '/Users/lpinto/Dropbox/PintoEtAl2020_subtrial_inact/data/dffGLM_time_ridge_ROI_autoRegr_ai3_20170201.mat',
            'mouse_eg_id'           : 1,
            'areaList'              : ['V1','mV2','PPC','RSC','mM2','aM2','M1'],
            'area_eg_idx_autoregr'  : [0, 3, 4, 6],
            'area_eg_idx_glm'       : [0, 7, 10, 13],
            'trial_idx_glm'         : range(933,983)
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
    glm_summ['accuracy_anova_table']   = pg.rm_anova(dv='acc', within=['area','model'],subject='mouse',data=table)
    glm_summ['accuracy_posthoc_pvals'] = pg.pairwise_tukey(data=table,dv='acc',between='model')

    # timescales and towers
    glm_summ = analyze_timescales(glm,glm_summ)
    glm_summ = analyze_tower_weights(glm,glm_summ)

    return glm_summ

# ==============================================================================
# analyze autogressive coefficients
def analyze_timescales(glm,glm_summ):

    # extract relevant predictors
    num_mice   = len(glm_summ['mice'])
    num_areas  = len(glm_summ['areaList'])
    glm_areas  = utils.wf_ROIlbl_translate(glm['summ'].ROIlbl[0:-2])
    ispred_idx = np.array([glm['summ'].predLbls[iPred].find('auto') \
                           for iPred in range(np.size(glm['summ'].predLbls))])==0
    num_lags   = np.sum(ispred_idx)
    weights    = glm['summ'].weights[:-2,ispred_idx,:]
    glm_summ['acorr_weights']  = np.zeros((num_areas,num_lags,num_mice))
    glm_summ['taus']           = np.zeros((num_areas,num_mice))
    glm_summ['acorr_lags']     = np.arange(0,num_lags*.1,.1)
    glm_summ['acorr_xaxis']    = np.arange(0,num_lags*.1-.1,.01)
    glm_summ['acorr_fit_mice'] = np.zeros((num_areas,np.size(glm_summ['acorr_xaxis']),num_mice))
    glm_summ['acorr_fit_r2']   = np.zeros((num_areas,num_mice))

    # average over areas, normalize and fit exponentials
    for iArea, area in enumerate(glm_summ['areaList']):
        is_area = np.array(glm_areas) == area
        glm_summ['acorr_weights'][iArea,:,:] = np.mean(weights[is_area,:,:],axis=0)
        for iMouse in range(num_mice):
            glm_summ['acorr_weights'][iArea,:,iMouse] = glm_summ['acorr_weights'][iArea,:,iMouse] \
                                                        / glm_summ['acorr_weights'][iArea,0,iMouse]
            fit , _   = sp.optimize.curve_fit(exp_decay, glm_summ['acorr_lags'], \
                                              glm_summ['acorr_weights'][iArea,:,iMouse], maxfev=4000, bounds=[0, 1])
            glm_summ['taus'][iArea,iMouse]             = fit[0]
            glm_summ['acorr_fit_mice'][iArea,:,iMouse] = exp_decay(glm_summ['acorr_xaxis'],*fit)
            glm_summ['acorr_fit_r2'][iArea,iMouse]     = utils.fit_rsquare(glm_summ['acorr_weights'][iArea,:,iMouse], \
                                                                           exp_decay(glm_summ['acorr_lags'],*fit))

    # one-way ANOVA with RM for taus
    table = pd.DataFrame({'area' : np.repeat(range(num_areas), num_mice),
                          'mouse': np.tile(range(num_mice), num_areas),
                          'tau'  : glm_summ['taus'].flatten()})
    glm_summ['taus_anova_table'] = pg.rm_anova(dv='tau', within='area',subject='mouse',data=table)
    glm_summ['taus_anova_pval']  = glm_summ['taus_anova_table']['p-unc'].to_numpy()

    # avg fit
    glm_summ['acorr_weights_mean'] = np.mean(glm_summ['acorr_weights'],axis=2)
    glm_summ['acorr_weights_sem']  = np.std(glm_summ['acorr_weights'],axis=2,ddof=1) / np.sqrt(num_mice)
    glm_summ['acorr_fit_pred']     = np.zeros((num_areas,np.size(glm_summ['acorr_xaxis'])))
    for iArea in range(num_areas):
        fit , _ = sp.optimize.curve_fit(exp_decay, glm_summ['acorr_lags'], \
                                        glm_summ['acorr_weights_mean'][iArea,:], maxfev=4000, bounds=[0, 1])
        glm_summ['acorr_fit_pred'][iArea,:] = exp_decay(glm_summ['acorr_xaxis'],*fit)

    return glm_summ

# ==============================================================================
# analyze model coefficients related to towers
def analyze_tower_weights(glm,glm_summ,smooth_w=3):

    """
    glm_summ = analyze_tower_weights(glm,glm_summ,smooth_w=3)
    analyzes contralateral and \delta tower weights for each area
    called by summarize_glm. smooth_w is number of datapoints in sigma of a gaussian window
    for smoothing coefficients over lags
    """

    # extract relevant predictors
    towL_idx       = np.array([glm['summ'].predLbls[iPred].find('tow_L') \
                           for iPred in range(np.size(glm['summ'].predLbls))])==0
    towR_idx       = np.array([glm['summ'].predLbls[iPred].find('tow_R') \
                           for iPred in range(np.size(glm['summ'].predLbls))])==0
    delta_idx      = np.array([glm['summ'].predLbls[iPred].find('Delta') \
                           for iPred in range(np.size(glm['summ'].predLbls))])>0
    towL_weights   = glm['summ'].weights[:-2,towL_idx,:]
    towR_weights   = glm['summ'].weights[:-2,towR_idx,:]
    delta_weights  = glm['summ'].weights[:-2,delta_idx,:]

    # combine in ipsi and contra
    ROIlbl     = glm['summ'].ROIlbl[0:-2]
    glm_areas  = np.unique(utils.wf_ROIlbl_translate(ROIlbl))
    isR        = np.array([ROIlbl[iROI].find('-R') for iROI in range(len(ROIlbl))])>0
    isL        = ~isR

    hemR_towL_weights   = towL_weights[isR,:,:]
    hemR_towR_weights   = towR_weights[isR,:,:]
    hemL_towL_weights   = towL_weights[isL,:,:]
    hemL_towR_weights   = towR_weights[isL,:,:]
    contraDelta_weights = delta_weights[isL,:,:] # since \Delta = R - L, contra is always left hemisphere
    ipsiDelta_weights   = delta_weights[isR,:,:] # since \Delta = R - L, ipsi is always right hemisphere

    # initialize vars
    num_lags_tow   = np.sum(towL_idx)
    num_lags_delta = np.sum(delta_idx)
    dt             = glm['summ'].glmCfg.timeBins[1] - glm['summ'].glmCfg.timeBins[0]
    num_mice       = len(glm_summ['mice'])
    num_areas  = len(glm_areas)
    glm_summ['tower_taxis']          = np.arange(0,num_lags_tow*dt,dt)
    glm_summ['delta_taxis']          = np.arange(0,num_lags_delta*dt,dt)
    glm_summ['contraTow_weights']    = np.zeros((num_areas,num_lags_tow,num_mice))
    glm_summ['ipsiTow_weights']      = np.zeros((num_areas,num_lags_tow,num_mice))
    glm_summ['contraTow_peakTime']   = np.zeros((num_areas,num_mice))
    glm_summ['ipsiTow_peakTime']     = np.zeros((num_areas,num_mice))
    glm_summ['contraDelta_weights']  = np.zeros((num_areas,num_lags_delta,num_mice))
    glm_summ['ipsiDelta_weights']    = np.zeros((num_areas,num_lags_delta,num_mice))
    glm_summ['contraDelta_peakTime'] = np.zeros((num_areas,num_mice))
    glm_summ['ipsiDelta_peakTime']   = np.zeros((num_areas,num_mice))

    # baseline subtract, normalize, smooth, measure time of peak coefficient value
    for iArea, area in enumerate(glm_summ['areaList']):
        is_area = np.array(glm_areas) == area
        for iMouse in range(num_mice):
            thisw_hRtR = hemR_towR_weights[is_area,:,iMouse][0]
            thisw_hRtL = hemR_towL_weights[is_area,:,iMouse][0]
            thisw_hLtR = hemL_towR_weights[is_area,:,iMouse][0]
            thisw_hLtL = hemL_towL_weights[is_area,:,iMouse][0]

            this_contra = sp.ndimage.gaussian_filter1d((thisw_hLtR + thisw_hRtL)/2,smooth_w)
            this_ipsi   = sp.ndimage.gaussian_filter1d((thisw_hLtL + thisw_hRtR)/2,smooth_w)
            this_contra = this_contra - np.min(this_contra)
            this_contra = this_contra / np.max(this_contra)
            this_ipsi   = this_ipsi - np.min(this_ipsi)
            this_ipsi   = this_ipsi / np.max(this_ipsi)

            glm_summ['contraTow_weights'][iArea,:,iMouse] = this_contra
            glm_summ['contraTow_peakTime'][iArea,iMouse]  = glm_summ['tower_taxis'][this_contra==np.max(this_contra)]
            glm_summ['ipsiTow_weights'][iArea,:,iMouse] = this_ipsi
            glm_summ['ipsiTow_peakTime'][iArea,iMouse]  = glm_summ['tower_taxis'][this_ipsi==np.max(this_ipsi)]

            thisw = contraDelta_weights[is_area,:,iMouse][0]
            thisw = sp.ndimage.gaussian_filter1d(thisw,smooth_w)
            thisw = thisw - np.min(thisw)
            thisw = thisw / np.max(thisw)
            glm_summ['contraDelta_weights'][iArea,:,iMouse] = thisw
            glm_summ['contraDelta_peakTime'][iArea,iMouse]  = glm_summ['delta_taxis'][thisw==np.max(thisw)]

            thisw = ipsiDelta_weights[is_area,:,iMouse][0]
            thisw = sp.ndimage.gaussian_filter1d(thisw,smooth_w)
            thisw = thisw - np.min(thisw)
            thisw = thisw / np.max(thisw)

            glm_summ['ipsiDelta_weights'][iArea,:,iMouse] = thisw
            glm_summ['ipsiDelta_peakTime'][iArea,iMouse]  = glm_summ['delta_taxis'][thisw==np.max(thisw)]

    # one-way ANOVA with RM for peaks
    table = pd.DataFrame({'area' : np.repeat(range(num_areas), num_mice),
                          'mouse': np.tile(range(num_mice), num_areas),
                          'contraTow_peakTime' : glm_summ['contraTow_peakTime'].flatten()})
    glm_summ['contraTow_peakTime_anova_table'] = pg.rm_anova(dv='contraTow_peakTime', within='area',subject='mouse',data=table)
    glm_summ['contraTow_peakTime_anova_pval']  = glm_summ['contraTow_peakTime_anova_table']['p-unc'].to_numpy()

    table = pd.DataFrame({'area' : np.repeat(range(num_areas), num_mice),
                          'mouse': np.tile(range(num_mice), num_areas),
                          'ipsiTow_peakTime' : glm_summ['ipsiTow_peakTime'].flatten()})
    glm_summ['ipsiTow_peakTime_anova_table'] = pg.rm_anova(dv='ipsiTow_peakTime', within='area',subject='mouse',data=table)
    glm_summ['ipsiTow_peakTime_anova_pval']  = glm_summ['ipsiTow_peakTime_anova_table']['p-unc'].to_numpy()

    table = pd.DataFrame({'area' : np.repeat(range(num_areas), num_mice),
                          'mouse': np.tile(range(num_mice), num_areas),
                          'contraDelta_peakTime' : glm_summ['contraDelta_peakTime'].flatten()})
    glm_summ['contraDelta_peakTime_anova_table'] = pg.rm_anova(dv='contraDelta_peakTime', within='area',subject='mouse',data=table)
    glm_summ['contraDelta_peakTime_anova_pval']  = glm_summ['contraDelta_peakTime_anova_table']['p-unc'].to_numpy()

    table = pd.DataFrame({'area' : np.repeat(range(num_areas), num_mice),
                          'mouse': np.tile(range(num_mice), num_areas),
                          'ipsiDelta_peakTime' : glm_summ['ipsiDelta_peakTime'].flatten()})
    glm_summ['ipsiDelta_peakTime_anova_table'] = pg.rm_anova(dv='ipsiDelta_peakTime', within='area',subject='mouse',data=table)
    glm_summ['ipsiDelta_peakTime_anova_pval']  = glm_summ['ipsiDelta_peakTime_anova_table']['p-unc'].to_numpy()

    return glm_summ

# ==============================================================================
# plot model coefficients related to towers
def plot_tower_weights(glm_summ,plot_median=True):

    """
    fig = plot_tower_weights(glm_summ,plot_median=True)
    plots contralateral and \delta tower weights and analysis thereof for each area
    glm_summ is output of summarize_glm, fig is fig handle
    plot_median is boolean to show median rather than mean across mice
    """

    fig    = plt.figure(figsize=(5,6.5))
    cl     = utils.getAreaColors(glm_summ['areaList'])
    nareas = len(glm_summ['areaList'])
    nmice  = np.size(glm_summ['contraTow_peakTime'],axis=1)

    # plot contralateral tower weights by time
    plot_what = 'contraTow_weights'
    taxis     = glm_summ['tower_taxis']
    for iArea in range(nareas):
       ax    = fig.add_subplot(4,4,iArea+1)
       thism = np.mean(glm_summ[plot_what][iArea,:,:],axis=1)
       thiss = np.std(glm_summ[plot_what][iArea,:,:],axis=1,ddof=1) / np.sqrt(nmice-1)
       ax.fill_between(taxis,thism-thiss,thism+thiss,color=[.7,.7,.7,.5])
       ax.plot(taxis,thism,color=cl[iArea],lw=1.5)

       ax.set_xticks(np.arange(0,2.5,.5))
       ax.set_yticks(np.arange(0,1.25,.25))
       ax.set_xlim([0,2])
       ax.set_ylim([0,1])

       if iArea == 4:
           ax.set_xlabel('Lag (s)')
           ax.set_ylabel('Norm. contra tower coeff.')
       ax = utils.applyPlotDefaults(ax)

    # plot peaks per area / mouse
    ax = fig.add_subplot(4,4,8)
    plot_what = 'contraTow_peakTime'
    jitter    = np.random.uniform(size=(nmice))*0.05-0.025
    for iArea in range(nareas):
       this_peak = glm_summ[plot_what][iArea,:]
       ax.bar(iArea,np.mean(this_peak),color=cl[iArea])
       ax.errorbar(iArea,np.mean(this_peak),np.std(this_peak,ddof=1)/np.sqrt(nmice-1),color=cl[iArea])
       ax.plot(iArea+jitter,this_peak,marker='.',c=[.6,.6,.6,.5], ls='None')

    ax.set_xticks(np.arange(nareas))
    ax.set_xticklabels(glm_summ['areaList'],rotation=90)
    ax.set_yticks(np.arange(0,2.5,.5))
    ax.set_xlim([-.5,iArea+.5])
    ax.set_ylim([0,2.05])
    ax.set_ylabel('Time of peak (s)')
    # print ANOVA p (bins)
    plt.text(0,2.2,'p = {:.2f}'.format(glm_summ['contraTow_peakTime_anova_pval'][0]),fontsize=8,color='k')

    ax = utils.applyPlotDefaults(ax)


    # plot contralateral \Delta tower weights by time
    plot_what = 'contraDelta_weights'
    taxis     = glm_summ['delta_taxis']
    for iArea in range(nareas):
       ax    = fig.add_subplot(4,4,iArea+2+nareas)
       thism = np.mean(glm_summ[plot_what][iArea,:,:],axis=1)
       thiss = np.std(glm_summ[plot_what][iArea,:,:],axis=1,ddof=1) / np.sqrt(nmice-1)
       ax.fill_between(taxis,thism-thiss,thism+thiss,color=[.7,.7,.7,.5])
       ax.plot(taxis,thism,color=cl[iArea],lw=1.5)

       ax.set_xticks(np.arange(0,2.5,.5))
       ax.set_yticks(np.arange(0,1.25,.25))
       ax.set_xlim([0,2])
       ax.set_ylim([0,1])

       if iArea == 4:
           ax.set_xlabel('Lag (s)')
           ax.set_ylabel('Norm. \u0394 towers coeff.')
       ax = utils.applyPlotDefaults(ax)

    # plot peaks per area / mouse
    ax = fig.add_subplot(4,4,16)
    plot_what = 'contraDelta_peakTime'
    jitter    = np.random.uniform(size=(nmice))*0.05-0.025
    for iArea in range(nareas):
       this_peak = glm_summ[plot_what][iArea,:]
       if plot_median:
           thism = np.median(this_peak)
           thiss = sp.stats.iqr(this_peak)
       else:
           thism = np.mean(this_peak)
           thiss = np.std(this_peak,ddof=1)/np.sqrt(nmice-1)
       ax.bar(iArea,thism,color=cl[iArea])
       ax.errorbar(iArea,thism,thiss,color=cl[iArea])
       ax.plot(iArea+jitter,this_peak,marker='.',c=[.6,.6,.6,.5], ls='None')

    ax.set_xticks(np.arange(nareas))
    ax.set_xticklabels(glm_summ['areaList'],rotation=90)
    ax.set_yticks(np.arange(0,2.5,.5))
    ax.set_xlim([-.5,iArea+.5])
    ax.set_ylim([0,2.05])
    ax.set_ylabel('Time of peak (s)')
    # print ANOVA p (bins)
    plt.text(0,2.2,'p = {:.2f}'.format(glm_summ['contraDelta_peakTime_anova_pval'][0]),fontsize=8,color='k')
    ax = utils.applyPlotDefaults(ax)

    fig.subplots_adjust(left=.1, bottom=.25, right=.95, top=.75, wspace=.75, hspace=.75)

    return fig

# ==============================================================================
# plot glm timescales
def plot_glm_timescales(glm_summ,glm,plot_median=True):

    """
    fig, figdata = plot_glm_timescales(glm_summ,glm,plot_median=True)
    takes glm and glm_summ dictionaries, plots data summary and examples with emphasis on timescales
    plot_median is boolean to show median rather than mean across mice
    returns figure handle fig and source data dictionary figdata
    """

    colors    = utils.getAreaColors(glm_summ['areaList'])
    num_areas = len(glm_summ['areaList'])
    fig       = plt.figure(figsize=(5,4.5))
    figdata   = dict()

    # plot model performance
    ax  = fig.add_subplot(2,4,5)
    acc = glm_summ['accuracy_mice'][:,:,-1].flatten()
    ax.hist(acc,bins=np.arange(0,1.05,.05),color=[.5,.5,.5])
    ax.set_xticks(np.arange(0,1.25,.25))
    ax.set_xlabel('Cross-val. accuracy (avg. r)')
    ax.set_ylabel('Num. observations\n(Mice * ROI)')
    ax = utils.applyPlotDefaults(ax)
    figdata['glm_xval_accuracy_corrcoeff'] = acc

    # plot example model predictions
    area_eg_idx  = wf_params['area_eg_idx_glm']
    data_points  = wf_params['trial_idx_glm']
    xaxis        = np.arange(0,len(data_points)*.1,.1)
    subplot      = [2, 6, 10, 14]

    figdata['glm_examples_xaxis']       = xaxis
    figdata['glm_examples_xaxis_label'] = 'Time from trial start (s)'
    figdata['glm_examples_yaxis_label'] = '\DeltaF/F (z-score)'
    figdata['glm_examples_areaLabels']  = list()
    figdata['glm_examples_data']        = list()
    figdata['glm_examples_prediction']  = list()
    for iEg in range(len(area_eg_idx)):
        ax = fig.add_subplot(4,4,subplot[iEg])
        if iEg == len(area_eg_idx):
            data_points = data_points + 38 # hack: M1 is misaligned due to some NaN in this file
        thisy = sp.ndimage.gaussian_filter1d(glm['eg']['y'][area_eg_idx[iEg]][data_points], 1)
        figdata['glm_examples_data'].append(thisy)
        ax.plot(xaxis,thisy,linestyle='-',linewidth=.75,color=[.3, .3, .3],label='Data')
        thisy = sp.ndimage.gaussian_filter1d(glm['eg']['yhat'][area_eg_idx[iEg]][data_points], 1)
        figdata['glm_examples_prediction'].append(thisy)
        thiscl = utils.getAreaColors([glm['eg']['ROIlbl'][area_eg_idx[iEg]]])
        ax.plot(xaxis,thisy,linestyle='-',linewidth=1,color=thiscl[0],label='Prediction')
        ax.set_xticks(range(0,6,1))
        ax.set_title(glm['eg']['ROIlbl'][area_eg_idx[iEg]])

        if iEg == 2:
            ax.set_xlabel('Time from trial start (s)')
        if iEg == 3:
            ax.set_ylabel('Df/f (z-score)')

        utils.applyPlotDefaults(ax)
        figdata['glm_examples_areaLabels'].append(glm['eg']['ROIlbl'][area_eg_idx[iEg]])

    # plot example exponential fits
    # subplot      = [4, 5, 11, 12]
    subplot      = [3, 7, 11, 15]
    area_eg_idx  = wf_params['area_eg_idx_autoregr']
    figdata['expfit_examples_xaxis']       = -(np.flip(glm_summ['acorr_lags']+.1))
    figdata['expfit_examples_xaxis_label'] = 'Lag (s)'
    figdata['expfit_examples_yaxis_label'] = 'Norm. auto-regr. coeff.'
    figdata['expfit_examples_areaLabels']  = list()
    figdata['expfit_examples_data']        = list()
    figdata['expfit_examples_prediction']  = list()
    for iEg in range(len(area_eg_idx)):
        ax      = fig.add_subplot(4,4,subplot[iEg])
        areaidx = area_eg_idx[iEg]
        thisy   = glm_summ['acorr_weights'][areaidx,:,wf_params['mouse_eg_id']]
        ax.plot(-(np.flip(glm_summ['acorr_lags']+.1)),np.flip(thisy),linestyle='-',linewidth=.75,color=[.3, .3, .3],label='Data')
        figdata['expfit_examples_data'].append(np.flip(thisy))

        thisy   = glm_summ['acorr_fit_mice'][areaidx,:,wf_params['mouse_eg_id']]
        thiscl  = utils.getAreaColors([glm_summ['areaList'][areaidx]])
        ax.plot(-(np.flip(glm_summ['acorr_xaxis']+.1)),np.flip(thisy),linestyle='-',linewidth=1,color=thiscl[0],label='Prediction')
        ax.set_xticks(np.arange(-1,.25,.25))
        ax.set_xlim([-1,0])
        ax.set_title(glm_summ['areaList'][areaidx])

        figdata['expfit_examples_prediction'].append(np.flip(thisy))
        figdata['expfit_examples_areaLabels'].append(glm_summ['areaList'][areaidx])

        # if iEg == 0:
        # ax.legend()
        if iEg == 2:
            ax.set_xlabel('Lag (s)')
        if iEg == 3:
            ax.set_ylabel('Norm. auto-regr. coeff.')

        utils.applyPlotDefaults(ax)

    # plot exp. model R2
    ax  = fig.add_subplot(3,4,4)
    acc = glm_summ['acorr_fit_r2'].flatten()
    ax.hist(acc,bins=np.arange(0,1.05,.05),color=[.5,.5,.5])
    ax.set_xticks(np.arange(0,1.25,.25))
    ax.set_xlabel('Exponential fit R^2')
    ax.set_ylabel('Num. observations\n(Mice * ROI)')
    ax = utils.applyPlotDefaults(ax)
    figdata['exponentialFit_r2'] = acc

    # plot average fits
    ax = fig.add_subplot(3,4,8)
    figdata['expfit_areaList'] = glm_summ['areaList']
    figdata['expFit_area_avg'] = list()
    for iArea in range(num_areas):
        plt.plot(-(np.flip(glm_summ['acorr_xaxis']+.1)),np.flip(glm_summ['acorr_fit_pred'][iArea,:]),color=colors[iArea],linewidth=.5)
        figdata['expFit_area_avg'].append(np.flip(glm_summ['acorr_fit_pred'][iArea,:]))
    ax.set_xlabel('Lag (s)')
    ax.set_ylabel('Norm. auto-regr. coeff.')
    ax.set_xticks(np.arange(-.5,.1,.1))
    ax.set_yticks(np.arange(0,1.25,.25))
    ax.set_xlim([-.5,-.05])
    ax.set_ylim([0,1])
    ax = utils.applyPlotDefaults(ax)

    # plot taus
    figdata['expFit_tau_bymouse'] = list()
    if plot_median:
        figdata['expFit_tau_median']      = list()
        figdata['expFit_tau_interQrange'] = list()
    else:
        figdata['expFit_tau_mean']        = list()
        figdata['expFit_tau_sem']         = list()

    num_mice = np.size(glm_summ['taus'],axis=1)
    jitter   = np.random.uniform(size=(num_mice))*0.1-0.05
    ax       = fig.add_subplot(3,4,12)
    for iArea in range(num_areas):
        taus     = glm_summ['taus'][iArea,:]
        figdata['expFit_tau_bymouse'].append(taus)
        if plot_median:
            thismean = np.median(taus)
            thissem = sp.stats.iqr(taus)
            figdata['expFit_tau_median'].append(thismean)
            figdata['expFit_tau_interQrange'].append(thissem)
        else:
            thismean = np.mean(taus)
            thissem  = np.std(taus,ddof=1)/np.sqrt(num_mice)
            figdata['expFit_tau_mean'].append(thismean)
            figdata['expFit_tau_sem'].append(thissem)

        plt.bar(iArea,thismean,facecolor=colors[iArea])
        plt.errorbar(iArea,thismean,thissem,color=colors[iArea])
        ax.plot(iArea+jitter,taus,marker='.',c=[.6,.6,.6,.5], ls='None')

    ax.set_xticks(range(num_areas))
    ax.set_xticklabels(glm_summ['areaList'],rotation=90)
    ax.set_ylabel('$\u03C4$ (s)')
    ax.set_ylim([0, .4])
    ax.set_yticks(np.arange(0,.45,.1))
    ax.text(0,.15,'p = {:1.3f}'.format(glm_summ['taus_anova_pval'][0]))
    ax = utils.applyPlotDefaults(ax)
    figdata['expFit_tau_pval']  = glm_summ['taus_anova_pval'][0]

    fig.subplots_adjust(left=.1, bottom=.25, right=.95, top=.75, wspace=.75, hspace=1)

    return fig, figdata

# ==============================================================================
# plot glm timescales
def plot_glm_timescales_space(glm_summ,glm):

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
    plt.imshow(wmean,vmin=-0.3,vmax=0.3,cmap='PuOr')
    cb  = plt.colorbar(ax=ax)
    cb.set_label('ROI activity coefficient')
    ax.set_xticks(range(num_areas-2))
    ax.set_xticklabels(glm['summ'].ROIlbl[:-2],rotation=90)
    ax.set_yticks(range(num_areas-2))
    ax.set_yticklabels(glm['summ'].ROIlbl[:-2])

    return fig
