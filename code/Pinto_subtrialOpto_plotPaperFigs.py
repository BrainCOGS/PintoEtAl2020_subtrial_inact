# ==============================================================================
# Running this script will generate all manuscript figures
# Lucas Pinto, October 2020, lucas.pinto@northwestern.edu
# updated December 2021
#
# ==============================================================================
# DEPENDENCIES AND INSTRUCTIONS:
#
# -------------
# Python 3.8
# Matlab >= 2016b
# R 4.1
#
# Python libraries / modules:
#    Numpy 1.18.1,       https://www.numpy.org
#    Scipy 1.4.1,        https://www.scipy.org
#    Statsmodels 0.11.0, https://www.statsmodels.org
#    Matplotlib 3.1.3,   https://www.matplotlib.org
#    Pandas 1.0.1,       https://www.pandas.pydata.org
#    Pingouin 0.5.0,     https://pingouin-stats.org
#    Mat7.3,             https://github.com/skjerns/mat7.3
#    Scikit-learn        https://scikit-learn.org/stable/
#    flammkuchen         https://github.com/portugueslab/flammkuchen
#    pymer4 0.7          https://eshinjolly.com/pymer4/
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
root_dir =  '/Users/lucas/OneDrive - Northwestern University/PintoEtAl2020_subtrial_inact/' # repository path
code_dir = '{}code/'.format(root_dir) # where code is
data_dir = '{}data/'.format(root_dir) # where data is saved
fig_dir  = '{}figs_revision/'.format(root_dir) # where figs are saved

import os
os.chdir(code_dir) # cd to code directory assuming it isn't in the path already

import analyzeBehavior as behav # this module contains all functions to analyze behavior
from   analyzeBehavior import params # importing parameters separately
import lp_utils as utils # Lucas Pinto's utility functions
import analyzeWidefield as wf # widefield GLM analysis
from   analyzeWidefield import wf_params # importing parameters separately
import matplotlib.pyplot as plt
import numpy as np
import flammkuchen as fl

# %% ===========================================================================
# load behavior log and dictionary with behavioral analysis
# this command is identical to running the analysis from scratch
# params can be modified and passed to this function if desired
params['savePath']               = data_dir
inact_effects, lg, summary_table = behav.inact_batch_analyze(params=params)

# %% ===========================================================================
# load dictionaries with results of mixed logistic regressions
# this command is identical to running the analysis from scratch (but takes hours)
# params['regr_params'] can be modified and passed to this function if desired
lr_time_combined, _  = behav.run_mixed_time_regression(lg,regr_type='combinedEpochs',savePath=params['savePath'],overWrite=False)
lr_time_epochSets, _ = behav.run_mixed_time_regression(lg,regr_type='epochSets',savePath=params['savePath'],overWrite=False)
lr_time_shuff, _     = behav.run_mixed_time_regression_shuff(lg,regr_type='combinedEpochs',savePath=params['savePath'],overWrite=False,nshuff=30)

# %% ===========================================================================
# get per-mouse and overall data
if 'mouse_data' not in list(inact_effects.keys()):
    mouse_data                  = behav.behav_by_mouse(lg, doPlot=False) # per-mouse behavior
    inact_effects['speed']      = mouse_data['speed']
    inact_effects['mouse_data'] = mouse_data

    summary_table = behav.diagnose_dataset(lg,convertToDf=False) # save as dictionary, deepdish crashes when saving this as pandas dataframe
    newdata       = {'inact_effects':inact_effects, 'params': params, 'summary_table':summary_table}
    if params['excludeBadMice']:
        filename  = 'multiArea_inact_goodMice.hdf5'
    else:
        filename  = 'multiArea_inact_allMice.hdf5'
    filename      = '{}{}'.format(params['savePath'],filename)
    fl.save(filename,newdata)
else:
    mouse_data    = inact_effects['mouse_data']

logRegr           = behav.evidence_logRegr(lg['choice'][~lg['laserON']], lg['cuePos_R'][~lg['laserON']], lg['cuePos_L'][~lg['laserON']]) # aggregate logistic regression
psych             = behav.psychometrics(lg['choice'][~lg['laserON']], lg['nCues_RminusL'][~lg['laserON']]) # aggregate psychometrics

# %% ===========================================================================
# Fig 1 (data panels): overlaid psychometrics & logistic regression by mouse
fig1a , fig1a_data  = behav.plotCtrlBehav(mouse_data,psych,logRegr)
fig1b , fig1b_data  = behav.plot_multiArea(inact_effects,'percCorrect', 'overall_diff',mouse_data=mouse_data)
fig1a.savefig('{}fig1_ctrlBehav.pdf'.format(fig_dir))
fig1b.savefig('{}fig1_percCorrect_all.pdf'.format(fig_dir))

fig1_data = {'control_behavior': fig1a_data, 'inactivation_effects': fig1b_data}
fl.save('{}fig1_sourceData.hdf5'.format(fig_dir),fig1_data)

# %% ===========================================================================
# Figs S1-3, Table S1: plot all inactivation conditions, speed, plus table
fig_s1_psych, _    = behav.plot_multiArea(inact_effects,'psych', None)
fig_s2_logRegr, _  = behav.plot_multiArea(inact_effects,'logRegr','coeffs_diff')
fig_s3_speed , _   = behav.plot_multiArea(inact_effects,'speed', 'ratio',mouse_data=mouse_data)
fig_s1_psych.savefig('{}fig_s1_psych_all.pdf'.format(fig_dir))
fig_s2_logRegr.savefig('{}fig_s2_logRegr_diff_all.pdf'.format(fig_dir))
fig_s3_speed.savefig('{}fig_s3_speed.pdf'.format(fig_dir))
behav.saveSummaryTable(summary_table,fig_dir)

# %% ===========================================================================
# Fig 2: plot logistic regression in time
fig2, fig2a_data = behav.plot_logRegr_time(lr_time_combined,regr_type='combinedEpochs', \
                                            plot_what='evid_diff_norm',doAreaCl=False,shuff=lr_time_shuff,plot_sim=False)
fig2.add_subplot(3,3,8)
_ , fig2b_data   = behav.plot_model_predictions(lr_time_combined,ax=plt.gca())
fig2.add_subplot(3,3,9)
_ , fig2c_data   = behav.plot_lrtime_postVSfront(lr_time_combined,ax=plt.gca())
fig2.savefig('{}fig2_logRegr_mixed_time.pdf'.format(fig_dir))

fig2_data = {
             'logisticRegression_coefficients'      : fig2a_data,
             'logisticRegression_model_gof'         : fig2b_data,
             'logisticRegression_anova_frontVSpost' : fig2c_data,
             }
fl.save('{}fig2_sourceData.hdf5'.format(fig_dir),fig2_data)

# %% ===========================================================================
# Figs S4-5: further analyses in time
fig_s4_lsrVsCtrl, _ = behav.plot_logRegr_time(lr_time_combined,regr_type='combinedEpochs',\
                                              plot_what=['evid_ctrl','evid_lsr'],plot_sim=True)
fig_s4_lsrVsCtrl.savefig('{}fig_s4_lsrVsCtrl.pdf'.format(fig_dir))

fig_s5_epochs, _    = behav.plot_logRegr_time(lr_time_epochSets,regr_type='epochSets',\
                                              plot_what='evid_diff_norm',plot_sim=True)
fig_s5_epochs.savefig('{}fig_s5_epochs.pdf'.format(fig_dir))

# %% ===========================================================================
# Fig 3: simultaneous vs avg post/front
fig3, fig3_data = behav.plot_simultaneous_vs_avg(lr_time_combined)
fig3.savefig('{}fig3_simultaneous.pdf'.format(fig_dir)) # save
fl.save('{}fig3_sourceData.hdf5'.format(fig_dir),fig3_data)

# %% ===========================================================================
# Fig 3: GLM -- widefield timescales

# ADD TOWER AND DELTA weights AND PEAK TIMES
# PLOT INDIVIDUAL ANIMALS
wf_params['glm_comp_file'] = '{}dffGLM_autoregr_model_comp_time.mat'.format(data_dir)
wf_params['glm_summ_file'] = '{}glmSummary_ROI_corr_autoRegr_time.mat'.format(data_dir)
wf_params['glm_eg_file']   = '{}dffGLM_time_ridge_ROI_corr_autoRegr_ai3_20170201.mat'.format(data_dir)
glm      = wf.load_glm(wf_params=wf_params) # load mat files
glm_summ = wf.summarize_glm(glm) # summarize and do stats for figure
fig4 , fig4_data = wf.plot_glm_timescales(glm_summ,glm) # plot
fig4.savefig('{}fig4_glm.pdf'.format(fig_dir)) # save
fl.save('{}fig4_sourceData.hdf5'.format(fig_dir),fig3_data)

# %% ===========================================================================
# Fig S6: other ROI activity predictors
fig_s6a_towerWeights = wf.plot_tower_weights(glm_summ)
fig_s6a_towerWeights.savefig('{}fig_s6a_towerWeights.pdf'.format(fig_dir))
fig_s6b_ROIcorrGLM   = wf.plot_corr_pred(glm)
fig_s6b_ROIcorrGLM.savefig('{}fig_s6b_ROIcorrGLM.pdf'.format(fig_dir))
