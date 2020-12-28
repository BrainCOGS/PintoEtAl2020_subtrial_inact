# ==============================================================================
# Running this script will generate all manuscript figures
# Lucas Pinto, October 2020, lucas.pinto@northwestern.edu
#
# ==============================================================================
# DEPENDENCIES AND INSTRUCTIONS:
#
# -------------
# Python 3.7
# Matlab >= 2016b
#
# Python libraries / modules:
#    Numpy 1.18.1,       https://www.numpy.org
#    Scipy 1.4.1,        https://www.scipy.org
#    Deepdish 0.3.4,     https://github.com/uchicago-cs/deepdish
#    Statsmodels 0.11.0, https://www.statsmodels.org
#    Matplotlib 3.1.3,   https://www.matplotlib.org
#    Pandas 1.0.1,       https://www.pandas.pydata.org
#    Pingouin 0.3.8,     https://pingouin-stats.org
#    Mat7.3,             https://github.com/skjerns/mat7.3
#
# -------------
# Widefield GLMs: https://github.com/BrainCOGS/widefieldImaging.git
#   1. run dffGLM() with default parameters for each recording (full list in widefield_recLs.m)
#   2. run summarizeGLM() with autoRegrFlag set to true. This will save glmSummary_ROI_corr_autoRegr.mat
#   3. run compareGLMs_autoregr() to compare accuracy of different model versions
#   example recording is originally in '/jukebox/braininit/RigData/VRwidefield/widefield/ai3/20170201/dffGLM_space_ridge_ROI_autoRegr.mat',
#       was renamed manually and added to this repo for simplicity
#
# -------------
# Flattened matlab behavioral log: https://github.com/BrainCOGS/behavioralAnalysis.git
#   run concatLogs_subtrial()
#   this will save '/jukebox/braininit/Analysis/laserGalvo/concatLog_subtrial_inactivation.mat'
#   refer to behavLogDataFormatDescription.txt in this repository for documentation of data format
#
# ==============================================================================
# %% initialize
# declare directories (change as appropriate)
root_dir = '/Users/lpinto/Dropbox/PintoEtAl2020_subtrial_inact/' # repository path
code_dir = '{}code/'.format(root_dir) # where code is
data_dir = '{}data/'.format(root_dir) # where data is saved
fig_dir  = '{}figs/'.format(root_dir) # where figs are saved

import os
os.chdir(code_dir) # cd to code directory assuming it isn't in the path already

import analyzeBehavior as behav # this module contains all functions to analyze behavior
from   analyzeBehavior import params # importing parameters separately
import lp_utils as utils # Lucas Pinto's utility functions
import analyzeWidefield as wf # widefield GLM analysis
from   analyzeWidefield import wf_params # importing parameters separately
import matplotlib.pyplot as plt
import numpy as np

# %% ===========================================================================
# load behavior log and dictionary with behavioral analysis
# this command is identical to running the analysis from scratch
# params can be modified and passed to this function if desired
params['savePath']               = data_dir
inact_effects, lg, summary_table = behav.inact_batch_analyze(params=params)

# %% ===========================================================================
# Fig 1 (data panels): overlaid psychometrics & logistic regression by mouse
mouse_data = behav.behav_by_mouse(lg, doPlot=False) # per-mouse behavior
logRegr    = behav.evidence_logRegr(lg.choice[~lg.laserON], lg.cuePos_R[~lg.laserON], lg.cuePos_L[~lg.laserON]) # aggregate logistic regression
psych      = behav.psychometrics(lg.choice[~lg.laserON], lg.nCues_RminusL[~lg.laserON]) # aggregate psychometrics
fig1       = plotCtrlBehav(mouse_data,psych,logRegr)
fig1.savefig('{}fig1_ctrlBehav.pdf'.format(fig_dir))

# %% ===========================================================================
# Figs S1-3, Table S1: plot all inactivation conditions, plus table
fig_s1_percCorrect = behav.plot_multiArea(inact_effects,'percCorrect', 'overall_diff')
fig_s2_psych       = behav.plot_multiArea(inact_effects,'psych', None)
fig_s3_logRegr     = behav.plot_multiArea(inact_effects,'logRegr','coeffs_diff')
fig_s1_percCorrect.savefig('{}fig_s1_percCorrect_all.pdf'.format(fig_dir))
fig_s2_psych.savefig('{}fig_s2_psych_all.pdf'.format(fig_dir))
fig_s3_logRegr.savefig('{}fig_s3_logRegr_diff_all.pdf'.format(fig_dir))
saveTable(summary_table,fig_dir)

# %% ===========================================================================
# Fig 2: analyze and plot laser-triggered averages (each area, clustering)
trig_logRegr = behav.laser_trig_logRegr(inact_effects)
fig2         = plot_trig_logRegr(trig_logRegr)
fig2.savefig('{}fig2_trig_logRegr.pdf'.format(fig_dir))

# %% ===========================================================================
# Fig 2: analyze and plot laser-triggered averages (each area, clustering)
fig_s4_offset         = plot_trig_offset(trig_logRegr)
fig_s4_offset.savefig('{}fig_s4_trig_offset.pdf'.format(fig_dir))

# %% ===========================================================================
# Fig S5: laser-triggered averages, simultaneous vs avg post/front
fig_s5_simultaneous = plot_simultaneous_vs_avg(trig_logRegr)
fig_s5_simultaneous.savefig('{}fig_s5_simultaneous.pdf'.format(fig_dir))

# %% ===========================================================================
# Fig 3: GLM -- widefield timescales
wf_params['glm_comp_file'] = '{}dffGLM_autoregr_model_comp.mat'.format(data_dir)
wf_params['glm_summ_file'] = '{}glmSummary_ROI_corr_autoRegr.mat'.format(data_dir)
wf_params['glm_eg_file']   = '{}dffGLM_space_ridge_ROI_autoRegr_ai3_20170201.mat'.format(data_dir)
glm      = wf.load_glm(wf_params=wf_params) # load mat files
glm_summ = wf.summarize_glm(glm) # summarize and do stats for figure
fig3     = wf.plot_glm_timescales(glm_summ,glm) # plot
fig3.savefig('{}fig3_glm.pdf'.format(fig_dir)) # save

# %% ===========================================================================
# Fig S4: other ROI activity predictors
fig_s5_ROIcorrGLM  = wf.plot_corr_pred(glm)
fig_s5_ROIcorrGLM.savefig('{}fig_s5_ROIcorrGLM.pdf'.format(fig_dir))

# %%
# FUNCTIONS
# ==============================================================================
# Fig 1
def plotCtrlBehav(mouse_data,psych,logRegr):

    fig      = plt.figure(figsize=[1.1,3])
    num_mice = len(mouse_data['psych'])
    ax       = plt.subplot(2,1,1)
    for iMouse in range(num_mice):
        ax.plot(mouse_data['psych'][iMouse]['fit_x'],mouse_data['psych'][iMouse]['fit_y'], \
                 color=[.7, .7, .7],linestyle='-',linewidth=.35)
    ax.errorbar(psych['delta_towers'],psych['P_wentRight'],\
             np.transpose(psych['P_wentRight_CI']),linewidth=.75,color='k',linestyle='none',marker='.',markersize=2)
    ax.plot(psych['fit_x'],psych['fit_y'],color='k',linewidth=.75)
    ax.set_xticks(np.arange(-15,20,5))
    ax.set_yticks(np.arange(0,1.25,.25))
    ax.set_yticklabels(np.arange(0,125,25))
    ax.set_ylabel('Went right (%)')
    utils.applyPlotDefaults(ax)

    ax = plt.subplot(2,1,2)
    for iMouse in range(num_mice):
        ax.plot(mouse_data['logRegr'][iMouse]['evidence_vals'],mouse_data['logRegr'][iMouse]['coeff'], \
                 color=[.7, .7, .7],linestyle='-',linewidth=.35)
    ax.errorbar(logRegr['evidence_vals'],logRegr['coeff'],logRegr['coeff_err'],linewidth=.75,color='k',marker='.',markersize=2)
    ax.set_ylabel('Weight on decision (a.u.)')
    ax.set_yticks(np.arange(0,.3,.1))
    ax.set_xlim([0, 200])
    ax.set_xlabel('Cue y (cm)')
    utils.applyPlotDefaults(ax)

    fig.subplots_adjust(left=.1, bottom=.1, right=.95, top=.9, wspace=.5, hspace=1)

    return fig

# ==============================================================================
# Fig 2
def plot_trig_logRegr(trig_logRegr):

    # plot  per-area trig log regr
    fig          = plt.figure(figsize=(4.5,4.5))
    area_list    = ['V1','RSC','mV2','PPC','mM2','aM2','M1']
    subplots     = [1, 2, 5, 6, 9, 10, 13]
    yrange       = [-47.5/2, 47.5/2]
    fillx        = [yrange[0], yrange[1], yrange[1], yrange[0]]
    filly        = [-.4, -.4, .2, .2]
    fillcl       = params['lsr_color']

    for ct, area in enumerate(area_list):
        ax = fig.add_subplot(4,4,subplots[ct])
        ax.plot([-100-47.5/2, 47.5/2],[0, 0],linestyle='--',linewidth=.5,color=[.7,.7,.7])
        ax.fill(fillx,filly,fillcl,alpha=.25)
        condidx = np.argwhere(np.array(trig_logRegr['area_lbls']) == area)
        for iCond in range(np.size(condidx)):
            ax.plot(trig_logRegr['onset_trig_xaxis'],trig_logRegr['onset_trig_coeffs'][condidx[iCond],:][0], \
                    color=[.7, .7, .7],linewidth=.5,marker='x',markerfacecolor='none',markersize=4)

        iArea = params['inact_locations'].index(area)
        ax.errorbar(trig_logRegr['onset_trig_xaxis'],trig_logRegr['onset_trig_area_mean'][iArea,:],\
                    trig_logRegr['onset_trig_area_sem'][iArea,:],color='k',linewidth=1) #,marker='o',markersize=3)
        ax.set_title(area,pad=0)
        ax.set_xticks(range(-100,50,50))
        ax.set_xlim([-100-47.5/2,47.5/2])
        ax.set_ylim([-.4,.2])

        # plot pvals
        x     = trig_logRegr['onset_trig_xaxis']
        y     = trig_logRegr['onset_trig_area_mean'][iArea,:] - trig_logRegr['onset_trig_area_sem'][iArea,:]
        p     = trig_logRegr['onset_trig_area_pvals'][iArea,:]
        isSig = trig_logRegr['onset_trig_area_isSig'][iArea,:]
        where = 'below'
        ax    = utils.plotPvalCircles(ax, x, y-.07, p, isSig, where,color='gray')

        if ct == 6:
            plt.ylabel('$\Delta$ weight on choice')
            plt.xlabel('Evidence y dist. from laser ON (cm)')

        # print ANOVA p (bins)
        plt.text(-100,.18,'p(y) = {:1.3f}'.format(trig_logRegr['onset_trig_area_anova_bins_pval'][iArea]),fontsize=6,color='k')

        utils.applyPlotDefaults(ax)

    # plot clustering
    ax      = fig.add_subplot(2,2,2)
    x       = trig_logRegr['onset_trig_xaxis']

    ax.fill(fillx,filly,fillcl,alpha=.25)
    ax.plot([-100-47.5/2, 47.5/2],[0, 0],linestyle='--',linewidth=.5,color='gray')

    num_clust = np.size(trig_logRegr['cluster_pvals'],axis=0)
    for iClust in range(num_clust):
        idx = np.argwhere(trig_logRegr['cluster_ids'] == iClust)
        if 0 in idx:
            thiscl = 'seagreen'
        elif 6 in idx:
            thiscl = 'coral'
        else:
            thiscl = 'mediumpurple'

        y   = trig_logRegr['cluster_mean'][iClust,:]
        s   = trig_logRegr['cluster_sem'][iClust,:]
        ax.errorbar(x,y,s,label='clust{}'.format(iClust),color=thiscl,linewidth=1)

        # plot pvals
        p     = trig_logRegr['cluster_pvals'][iClust,:]
        isSig = trig_logRegr['cluster_isSig'][iClust,:]
        where = 'below'
        ax    = utils.plotPvalCircles(ax, x, y-s-.02, p, isSig, where, color=thiscl)

    # ax.legend()
    plt.text(-100,.12,'clust1',color='coral',fontsize=7)
    plt.text(-100,.09,'clust2',color='mediumpurple',fontsize=7)
    plt.text(-100,.06,'clust3',color='seagreen',fontsize=7)
    ax.set_xticks(range(-100,50,50))
    ax.set_xlim([-100-47.5/2,47.5/2])
    ax.set_ylim([-.3,.15])
    plt.ylabel('$\Delta$ weight on choice (laser ON - laser OFF, a.u.)')
    plt.xlabel('Evidence y dist. from laser ON (cm)')
    utils.applyPlotDefaults(ax)

    fig.subplots_adjust(left=.1, bottom=.1, right=.95, top=.9, wspace=.5, hspace=.5)

    return fig

# ===========================================================================
# Suppl Fig 4 - offset-triggered coefficients
def plot_trig_offset(trig_logRegr):

    fig          = plt.figure(figsize=(7,1.5))
    area_list    = ['V1','RSC','mV2','PPC','mM2','aM2','M1']

    for ct, area in enumerate(area_list):
        ax = fig.add_subplot(1,7,ct+1)
        ax.plot([47.5/2, 100+47.5/2],[0, 0],linestyle='--',linewidth=.5,color=[.7,.7,.7])
        condidx = np.argwhere(np.array(trig_logRegr['area_lbls']) == area)
        for iCond in range(np.size(condidx)):
            ax.plot(trig_logRegr['offset_trig_xaxis'],trig_logRegr['offset_trig_coeffs'][condidx[iCond],:][0], \
                    color=[.7, .7, .7],linewidth=.5,marker='x',markerfacecolor='none',markersize=4)

        iArea = params['inact_locations'].index(area)
        ax.errorbar(trig_logRegr['offset_trig_xaxis'],trig_logRegr['offset_trig_area_mean'][iArea,:],\
                    trig_logRegr['offset_trig_area_sem'][iArea,:],color='k',linewidth=1) #,marker='o',markersize=3)
        ax.set_title(area,pad=0)
        ax.set_xticks(range(0,150,50))
        ax.set_yticks(np.arange(-.5,.75,.25))
        ax.set_xlim([47.5/2,100+47.5/2])
        ax.set_ylim([-.5,.5])

        # plot pvals
        x     = trig_logRegr['offset_trig_xaxis']
        y     = trig_logRegr['offset_trig_area_mean'][iArea,:] - trig_logRegr['offset_trig_area_sem'][iArea,:]
        p     = trig_logRegr['offset_trig_area_pvals'][iArea,:]
        isSig = trig_logRegr['offset_trig_area_isSig'][iArea,:]
        where = 'below'
        ax    = utils.plotPvalCircles(ax, x, y-.07, p, isSig, where,color='gray')

        if ct == 0:
            plt.ylabel('$\Delta$ weight on choice')
        if ct == 3:
            plt.xlabel('Evidence y dist. from laser ON (cm)')

        utils.applyPlotDefaults(ax)

    fig.subplots_adjust(left=.1, bottom=.1, right=.95, top=.9, wspace=.5, hspace=.5)

    return fig

# ===========================================================================
# Suppl Fig 5 - simulatenous vs avg
def plot_simultaneous_vs_avg(trig_logRegr):

    area_list    = params['inact_locations']
    yrange       = [-47.5/2, 47.5/2]
    fillx        = [yrange[0], yrange[1], yrange[1], yrange[0]]
    filly        = [-.4, -.4, .2, .2]
    fillcl       = params['lsr_color']

    fig = plt.figure(figsize=(2.7,1.2))

    # simulatenous inact
    ax  = fig.add_subplot(1,2,1)
    post_idx = area_list.index('Post')
    ax.fill(fillx,filly,fillcl,alpha=.25)
    ax.plot([-100-47.5/2, 47.5/2],[0, 0],linestyle='--',linewidth=.5,color='gray')
    ax.errorbar(trig_logRegr['onset_trig_xaxis'],trig_logRegr['onset_trig_post_mean'],\
                trig_logRegr['onset_trig_post_sem'],label='Single-area mean',color='gray',linewidth=.75)
    ax.errorbar(trig_logRegr['onset_trig_xaxis'],trig_logRegr['onset_trig_area_mean'][4,:],\
                trig_logRegr['onset_trig_area_sem'][post_idx,:],label='Simultaneous',color='k',linewidth=.75)

    plt.text(-100,.1,'Simultaneous',color='k',fontsize=7)
    plt.text(-100,.05,'Single-area avg',color='gray',fontsize=7)
    plt.text(-100,-.19,'p(y) = {:1.1g}'.format(trig_logRegr['onset_trig_post_singleVSsimult_anova_pvals'][0]),color='gray',fontsize=5.5)
    plt.text(-100,-.25,'p(inact.type) = {:1.2f}'.format(trig_logRegr['onset_trig_post_singleVSsimult_anova_pvals'][1]),color='gray',fontsize=5.5)
    ax.set_xticks(range(-100,50,50))
    ax.set_xlim([-100-47.5/2,47.5/2])
    ax.set_ylim([-.3,.1])
    plt.ylabel('$\Delta$ weight on choice (laser ON - laser OFF, a.u.)')
    plt.xlabel('Evidence y dist. from laser ON (cm)')

    ax.set_title('Posterior')
    utils.applyPlotDefaults(ax)

    ax = fig.add_subplot(1,2,2)
    front_idx = area_list.index('Front')
    ax.fill(fillx,filly,fillcl,alpha=.25)
    ax.plot([-100-47.5/2, 47.5/2],[0, 0],linestyle='--',linewidth=.5,color='gray')
    ax.errorbar(trig_logRegr['onset_trig_xaxis'],trig_logRegr['onset_trig_front_mean'],\
                trig_logRegr['onset_trig_front_sem'],label='Single-area mean',color='gray',linewidth=.75)
    ax.errorbar(trig_logRegr['onset_trig_xaxis'],trig_logRegr['onset_trig_area_mean'][8,:],\
                trig_logRegr['onset_trig_area_sem'][front_idx,:],label='Simultaneous',color='k',linewidth=.75)

    plt.text(-100,-.19,'p(y) = {:1.1g}'.format(trig_logRegr['onset_trig_front_singleVSsimult_anova_pvals'][0]),color='gray',fontsize=5.5)
    plt.text(-100,-.25,'p(inact.type) = {:1.2f}'.format(trig_logRegr['onset_trig_front_singleVSsimult_anova_pvals'][1]),color='gray',fontsize=5.5)
    ax.set_xticks(range(-100,50,50))
    ax.set_xlim([-100-47.5/2,47.5/2])
    ax.set_ylim([-.3,.1])

    ax.set_title('Frontal')
    utils.applyPlotDefaults(ax)

    fig.subplots_adjust(left=.1, bottom=.1, right=.95, top=.9, wspace=.5, hspace=.5)

    return fig

# ===========================================================================
# save summary table pdf & xls
def saveTable(table,save_dir):
    summary_table.to_excel('{}table_s1_summary.xlsx'.format(save_dir))
    table_s1, ax = plt.subplots(figsize=(12,4))
    ax.axis('tight')
    ax.axis('off')
    col_names    = utils.formatVarNames(list(summary_table.columns))
    table_handle = ax.table(cellText=summary_table.values,colLabels=col_names,loc='center')
    table_s1.tight_layout()
    table_s1.show()
    table_s1.savefig('{}table_s1_summary.pdf'.format(fig_dir))
