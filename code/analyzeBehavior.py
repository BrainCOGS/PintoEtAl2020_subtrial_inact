"""
Module to analyze behavior (of the mice or RNNs)
Includes pyschometrics, logistic regression, plotting, trial selection
Lucas Pinto 2020-2021, lucas.pinto@northwestern.edu
"""
#!/usr/bin/env python

# Libraries
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import lp_utils as utils
import multiprocessing as mp
import flammkuchen as fl
import time
import copy
import os.path
import pingouin as pg
import numpy.matlib
import mat73
from   os import path
from   statsmodels.discrete.discrete_model import Logit
from   statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from   statsmodels.stats.proportion import proportion_confint
from   pymer4.models import Lmer
from   sklearn.model_selection import train_test_split

# default parameters
params = {
        # analysis
        'rightCode'              : 1,                      # right choices in behavior logs
        'leftCode'               : 0,                      # left choices
        'psych_bins'             : np.arange(-15,20,5),    # bins for \delta evidence in psychometrics
        'logRegr_nbins'          : 4,                      # number of even bins between 10 and 200 cm for logistic regression
        'multiArea_boot_numiter' : 10000,                  # number of bootstrapping iterations for multi-area analysis
        'logRegr_boot_numiter'   : 200,                    # number of bootstrapping iterations for logistic regression
        'alpha'                  : 0.05,                   # significance level before FDR
        'max_clust'              : 5,                      # max num clusts to test in clustering algorithm
        'clust_singleAreas_only' : True,                   # cluster only single areas, or also Front and Post?
        'clust_algo'             : 'Spectral',             # clustering algorithm, 'Hierarchical' or 'Spectral'
        'clust_what'             : 'onset',                # what laser-triggered weights to cluster, 'onset', 'offset' or 'onset+offset'

        'behavLogPath'           : '/Volumes/braininit/Analysis/laserGalvo/concatLog_subtrial_inactivation.mat', # location of .mat behavioral data
        'savePath'               : '/Volumes/braininit/Analysis/laserGalvo/', # save path
        'removeThetaOutlierMode' : 'none',                 # remove outlier trials from log, 'soft', 'stringent', or 'none' (there's some pre-selection already in .mat file)
        'removeZeroEvidTrials'   : True,                   # remove control trials where #r : #l
        'excludeBadMice'         : False,                  # exclude mice with bad logistic regression functions (in space)
        'inact_locations'        : ['V1', 'mV2', 'PPC', 'RSC', 'Post', 'mM2', 'aM2', 'M1', 'Front'],       # area labels for inactivation experiments
        'inact_epochs'           : ['cueQuart1', 'cueQuart2', 'cueHalf1', 'cueQuart3', 'cueHalf2', 'mem'], # trial epoch labels for inactivation experiments

        # plotting
        'ctrl_color'             : 'k',                           # default color for plotting ctrl curves
        'ctrl_shade'             : [.6, .6, .6],                  # default color for plotting ctrl shades
        'lsr_color'              : np.array([57, 133, 255])/255,  # default color for plotting inactivation curves
        'lsr_shade'              : np.array([151, 193, 252])/255, # default color for plotting inactivation shades
        'tick_lbl_size'          : 8,                             # fontsize ticklabels
        'axis_lbl_size'          : 10,                            # fontsize ticklabels
        'title_size'             : 12,                            # fontsize ticklabels
         }

params['regr_params'] = {
                        'bins_per_sec' : [2, 2, 2],       # time bins per sec
                        'max_t'        : [1.5, .5, 1],    # max time in sec to calculate bins
                        'addLsrOffset' : True,            # add explicit laser bias predictor and related mixed effects
                        'zscore'       : True,            # zscore evidence matrix
                        'nfold_xval'   : 10,              # folds of cross-validation runs
                        'method'       : 'Lmer_explicit', #  'Lmer_explicit' or 'Bayes'.
                        }

# ==============================================================================
# ==============================================================================
# batch analyze the inactivation data (wrapper function)
def inact_batch_analyze(params = params, doPlot = False, doSave = True):
    """
    inact_effects, lg = inact_batch_analyze(local_params = params, doPlot == True, doSave == True)
    wrapper function to analyze inactivation effects
    loads log, selects trials and mice, runs bootstrapping analysis of
    logistic regression of choice vs evidence bins, psychometrics, % correct
    returns dictionary with analysis summary and stats, and cleaned up behavioral
    log object. Optional inputs are booleans to plot and save results
    If file is already in disk and parameters match, it will just load the data instead
    """

    if params['excludeBadMice']:
        filename    = 'multiArea_inact_goodMice.hdf5'
        logfilename = 'concatLog_subtrial_inactivation_goodMice.hdf5'
        suffix      = 'goodMice'
    else:
        filename    = 'multiArea_inact_allMice.hdf5'
        logfilename = 'concatLog_subtrial_inactivation_allMice.hdf5'
        suffix      = 'allMice'

    filename    = '{}{}'.format(params['savePath'],filename)

    # first check if file exists and key params match, in which case just load
    if path.exists(filename):
        data            = fl.load(filename)
        stored_params   = data['params']
        key_param_names = ['psych_bins', 'logRegr_nbins', 'multiArea_boot_numiter', 'logRegr_boot_numiter', \
                           'alpha', 'removeThetaOutlierMode', 'removeZeroEvidTrials', 'excludeBadMice'
                          ]
        ct = 0
        for par in key_param_names:
            if isinstance(stored_params[par],np.ndarray):
                if sum(stored_params[par] == params[par]) == np.size(stored_params[par]):
                    ct = ct + 1
            else:
                if stored_params[par] == params[par]:
                    ct = ct + 1

        if ct == len(key_param_names):
            print('behavioral analysis file exists and params match: loading data from disk...')

            lg            = loadBehavLog(params)
            inact_effects = data['inact_effects']
            summary_table = pd.DataFrame(data['summary_table']) # return as data frame

            return inact_effects , lg, summary_table
        else:
            print("behavioral analysis file exists but params don't match: rerunning analysis...")
            del data

    else:
        print("behavioral analysis file doesn't exist: running analysis...")

    # load data and exclude bad trials and bad mice
    print('LOADING AND CLEANING UP BEHAVIORAL DATA...\n')
    lg = loadBehavLog(params)

    # summary stats
    summary_table = diagnose_dataset(lg,convertToDf=False) # save as dictionary, deepdish crashes when saving this as pandas dataframe

    # analyze inactivations
    print('ANALYZING INACTIVATION EFFECTS...\n')
    inact_effects = multiArea_boot_perf(lg, True, params['logRegr_boot_numiter'], params['logRegr_nbins'], params['psych_bins'])

    if doSave == True:
        print('saving results...')
        data = {'params': params, 'inact_effects': inact_effects, 'summary_table': summary_table}
        fl.save(filename, data)

    # plot
    if doPlot == True:
        fig1 = plot_multiArea(inact_effects,'logRegr','coeffs_diff')
        fig2 = plot_multiArea(inact_effects,'psych', None)
        fig3 = plot_multiArea(inact_effects,'percCorrect', 'overall_diff')

        fig1.savefig('{}logRegr_diff_{}.pdf'.format(params['savePath'],suffix))
        fig2.savefig('{}psych_{}.pdf'.format(params['savePath'],suffix))
        fig3.savefig('{}percCorrect_overall_{}.pdf'.format(params['savePath'],suffix))

    summary_table = pd.DataFrame(summary_table) # return as data frame
    return inact_effects , lg, summary_table

# ==============================================================================
# Load .mat7.3 behavior log
def loadBehavLogMatlab(filepath = params['behavLogPath']):
    """
    lg = loadBehavLogMatlab(filepath = params['behavLogPath']):
    loads .mat file with behavioral log, returns log dictionary
    """

    print('loading .mat behavior log...')

    # load behavior data as a dictionary. Variables are attributes
    lg = mat73.loadmat(filepath)
    lg = lg['lg']
    lg['laserON'] = lg['laserON'] == 1
    for iF in list(lg.keys()):
        if isinstance(lg[iF],list):
            lg[iF] = np.array(lg[iF])

    return lg

# ==============================================================================
# Load .hdf5 behavior log
def loadBehavLog(params=params):
    """
    lg = loadBehavLog(params=params):
    loads .mat file with behavioral log, returns log object
    """

    print('loading .hdf5 behavior log...')

    # load behavior data as a dictionary. Variables are attributes
    if params['excludeBadMice']:
        logfilename = 'concatLog_subtrial_inactivation_goodMice.hdf5'
    else:
        logfilename = 'concatLog_subtrial_inactivation_allMice.hdf5'

    logfilename     = '{}{}'.format(params['savePath'],logfilename)

    if path.exists(logfilename):
        lg = fl.load(logfilename)
    else:
        lg = loadBehavLogMatlab(params['behavLogPath'])
        lg = cleanupBehavLog(lg, params['removeThetaOutlierMode'], params['removeZeroEvidTrials'], params['excludeBadMice'])
        fl.save(logfilename,lg)

    return lg

# ===========================================================================
# save summary table pdf & xls
def saveSummaryTable(summary_table,save_dir):
    summary_table.to_excel('{}table_s1_summary.xlsx'.format(save_dir))
    table_s1, ax = plt.subplots(figsize=(12,4))
    ax.axis('tight')
    ax.axis('off')
    col_names    = utils.formatVarNames(list(summary_table.columns))
    table_handle = ax.table(cellText=summary_table.values,colLabels=col_names,loc='center')
    table_s1.tight_layout()
    table_s1.show()
    table_s1.savefig('{}table_s1_summary.pdf'.format(save_dir))

# ==============================================================================
# Psychometric function
def psychometrics(choice, nCues_RminusL, bins=params['psych_bins']):
    """
    psych = psychometrics(choice, nCues_RminusL, bins=params['psych_bins']):
    computes psychometrics as proportion of right choices by amount of right evidence
    and fits a sigmoid
    choice: numpy array with ones and zeros, elements are trials
    nCues_RminusL: numpy array with #R-#L evidence values per trial
    bins: numpy array with bin edges for evidence values (default in params dictionary)

    returns a dictionary with psychometrics data
    """

    psych                   = dict()
    psych['bins']           = bins
    psych['nTrials']        = np.zeros((len(bins)-1,1))
    psych['P_wentRight']    = np.zeros((len(bins)-1,1))
    psych['delta_towers']   = np.zeros((len(bins)-1,1))
    psych['P_wentRight_CI'] = np.zeros((len(bins)-1,2))

    alpha         = 1 - sp.stats.norm.cdf(1, 0, 1) # for confidence interval

    # calculate prop. went right for different evidence levels
    for iBin in range(np.size(bins)-1):
        idx = np.logical_and(nCues_RminusL >= bins[iBin], nCues_RminusL < bins[iBin+1])
        psych['nTrials'][iBin]     = np.sum(idx)
        psych['P_wentRight'][iBin] = np.sum(choice[idx] == params['rightCode']) / psych['nTrials'][iBin]

        # calculate x axis value as weighted average of actual trials rather than center of bin
        deltas    = np.unique(nCues_RminusL[idx])
        numtrials = np.zeros((1,len(deltas)))
        for iDelta in range(len(deltas)):
            numtrials[:,iDelta] = sum(nCues_RminusL[idx] == deltas[iDelta])

        psych['delta_towers'][iBin]   = np.sum(deltas * numtrials) / np.sum(numtrials)

        # binomial confidence intervals
        psych['P_wentRight_CI'][iBin,0] , psych['P_wentRight_CI'][iBin,1] \
                          = proportion_confint(psych['P_wentRight'][iBin], sum(idx), alpha=alpha, method='jeffreys')

    # fit sigmoid
    try:
        psych['fit'] , _  = sp.optimize.curve_fit(psych_fit_fn, np.transpose(psych['delta_towers'][:,0]), \
                                                  np.transpose(psych['P_wentRight'][:,0]),maxfev=5000)
    except (RuntimeError, ValueError, TypeError):
        psych['fit']      = None

    if psych['fit'] is None:
        psych['fit_x']    = np.nan
        psych['fit_y']    = np.nan
        psych['slope']    = np.nan
    else:
        psych['fit_x']    = np.arange(bins[0],bins[-1]+.1,.1)
        psych['fit_y']    = psych_fit_fn(psych['fit_x'],*psych['fit'])
        psych['slope']    = psych['fit'][1] / (4 * psych['fit'][3])

    return psych

# ==============================================================================
# Psychometric sigmoid for fitting
def psych_fit_fn(x, offset, A, x0, l):
    return offset + A / (1 + np.exp(-(x-x0) / l))

# ==============================================================================
# Psychometric plot
def plot_psych(psych, cl='k', sh=[.5, .5, .5], axisHandle=None):
    """
    plot_psych(psych, cl='k', sh=[.5, .5, .5], axisHandle=None):
    plots psychometrics (values and fit)
    psych is dictionary output by analyzeBehavior.psychometrics
    cl: color for datapoints
    sh: color for fit line
    axisHandle: matplotlib axis handle
    """

    if axisHandle is None:
        plt.figure()
        ax = plt.gca()
    else:
        ax = axisHandle

    ax.plot([0, 0],[0, 1],'--',color=[.7, .7, .7],linewidth=.25)
    ax.plot([-15, 15],[.5, .5],'--',color=[.7, .7, .7],linewidth=.25)
    ax.errorbar(psych['delta_towers'],psych['P_wentRight'],\
             np.transpose(psych['P_wentRight_CI']),marker='.',linewidth=.75,color=cl,linestyle='none',markersize=4)

    if psych['fit_x'] is not None:
        ax.plot(psych['fit_x'],psych['fit_y'],color=sh,linewidth=.75)

    ax.set_xlabel('$\Delta$ towers (#R - #L)')
    ax.set_ylabel('P went right')
    ax.set_xlim([-15, 15])
    ax.set_ylim([0, 1])
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return ax

# ==============================================================================
# Logistic regression of choice as a function of evidence bins
def evidence_logRegr(choice, cuePos_R, cuePos_L, numbins=params['logRegr_nbins'], mouseID=None, verbose=False):
    """
    logRegr = evidence_logRegr(choice, cuePos_R, cuePos_L, numbins=params['logRegr_nbins'], mouseID=None):
    computes a logistic regression model of choice as a function of net evidence from different maze bins
    choice:   numpy array with ones and zeros, elements are trials
    cuePos_R: numpy array with y position values of right towers per trial
    cuePos_L: numpy array with y position values of left towers per trial
    numbins:  number of equal bins between 10 cm and 200 cm (default in params dictionary)
    mouseID:  numpy array with mouseID numbers. If provided this input will be interpreted
               as a request for a mixed effects logistic regression where mouse ID has random
               effects on the intercept

    returns a dictionary with model fit data
    """

    lCue    = 200 # cue period length
    bins    = np.linspace(10,lCue,numbins+1) # first possible tower at 10 cm, last at lCue
    ntrials = np.size(choice);

    # predictor matrix for logistic regression, trials x evidence bins
    start_time = time.time()

    if verbose == True:
        print('Building predictor matrix...')

    RminusLmat = np.zeros((ntrials,numbins))
    for iTrial in range(ntrials):
        rhist, bincenters    = np.histogram(cuePos_R[iTrial],bins);
        lhist, _             = np.histogram(cuePos_L[iTrial],bins);
        RminusLmat[iTrial,:] = rhist - lhist

    # fit model
    if mouseID is None: # regular logistic regression
        if verbose == True:
            print('fitting binned evidence model with Logit...')
        model_name = 'logit'
        try:
            model      = Logit(choice,RminusLmat).fit(disp=False)
            coeffs     = model.params
            err        = model.bse
        except:
            coeffs     = np.nan * np.ones(numbins)
            err        = np.nan * np.ones(numbins)

    else: # mixed effects logistic regression
        if verbose == True:
            print('fitting binned evidence model with BinomialBayesMixedGLM...')

        # convert to dataFrame first
        data    = {'choice': choice, 'mouse': mouseID}
        formula = 'choice ~'
        # form_me = '0 '
        for iBin in range(numbins):
            varname       = "b{}".format(iBin)
            data[varname] = RminusLmat[:,iBin]
            formula       += " {} +".format(varname)
            # form_me       += " + C(mouse)*{}".format(varname)

        formula        = formula[:-1]
        data           = pd.DataFrame(data)
        random_effects = {"a": '0 + C(mouse)'} # "b": form_me} slope does not converge
        model          = BinomialBayesMixedGLM.from_formula(formula, random_effects, data).fit_vb()
        model_name     = 'me-logit'
        coeffs         = model.fe_mean[1:] # first element is intercept
        err            = model.fe_sd[1:]

    end_time = time.time()
    if verbose == True:
        print("done after {: 1.2g} min".format((end_time-start_time)/60))

    # organize output
    logRegr = {
              'coeff'        : coeffs,  # fitted coefficients excluding intercept
              'coeff_err'    : err,    # error estimates on coefficients
              'evidence_vals': bins[:-1]+np.diff(bins)[0]/2,   # center of evidence bins
              'model_name'   : model_name,   # center of evidence bins
              }

    return logRegr

# ==============================================================================
# Bootstrap Logistic regression of choice as a function of evidence bins for single dataset
# return average, stds and stats per bin
def boot_logRegr(choice, cuePos_R, cuePos_L, numbins=params['logRegr_nbins'], numiter=params['logRegr_boot_numiter'],alpha=params['alpha']):

    """
    logRegr = boot_logRegr(choice, cuePos_R, cuePos_L, numbins=params['logRegr_nbins'], numiter=params['logRegr_boot_numiter'],alpha=params['alpha'])
    samples trials with replacement and computes a logistic regression model of
    choice as a function of net evidence from different maze bins by calling evidence_logRegr()

    choice:   numpy array with ones and zeros, elements are trials
    cuePos_R: numpy array with y position values of right towers per trial
    cuePos_L: numpy array with y position values of left towers per trial
    numbins:  number of equal bins between 10 cm and 200 cm (default in params dictionary)
    numiter:  number of bootstrapping iterations (default in params dictionary)
    alpha:    significance value, will be FDR corrected across numbins (default in params dictionary)

    returns a dictionary with model fit data, plus significance of coefficients
    """

    # initialize
    start_time = time.time()
    print('bootstrapping logistic regression...')

    coeffs    = np.zeros((numiter,numbins))
    numtrials = np.size(choice)
    trialidx  = np.arange(numtrials)

    # sample with replacement and fit logistic model
    for iBoot in range(numiter):
        idx = np.random.choice(trialidx, size=numtrials, replace=True)
        lr  = evidence_logRegr(choice[idx], cuePos_R[idx], cuePos_L[idx], params['logRegr_nbins'], None, False)
        coeffs[iBoot,:] = lr['coeff']

    # p-vals and false discovery rate correction
    pvals = np.zeros(numbins)
    for iBin in range(numbins):
        pvals[iBin] = sum(coeffs[:,iBin] < 0) / numiter

    isSig, alpha_correct = utils.FDR(pvals,alpha)

    # organize output
    logRegr = {
              'coeff'        : np.mean(coeffs,axis=0),  # fitted coefficients excluding intercept
              'coeff_err'    : np.std(coeffs,axis=0,ddof=1),   # error estimates on coefficients
              'coeff_pval'   : pvals,                   # p values for coefficients
              'coeff_isSig'  : isSig,                   # booolean for significance after FDR correction
              'alpha_correct': alpha_correct,           # significance level after FDR
              'evidence_vals': lr['evidence_vals'],     # center of evidence bins
              'model_name'   : lr['model_name'],        # center of evidence bins
              }

    end_time = time.time()
    print("done after {: 1.2g} min".format((end_time-start_time)/60))

    return logRegr

# ==============================================================================
# Logistic regression plot
def plot_logRegr(logRegr, cl='k', axisHandle=None):

    """
    plot_logRegr(logRegr, cl='k', axisHandle=None):
    plots logistic regression coefficients
    logRegr is dictionary output by analyzeBehavior.evidence_logRegr
    cl: color for datapoints
    axisHandle: matplotlib axis handle
    """

    if axisHandle is None:
        plt.figure()
        ax = plt.gca()
    else:
        ax = axisHandle

    ax.plot([0, 200],[0, 0],'--',color=[.7, .7, .7],linewidth=.5)
    ax.errorbar(logRegr['evidence_vals'],logRegr['coeff'],  \
                logRegr['coeff_err'],marker='.',linewidth=.75,color=cl,markersize=4)
    ax.set_xlabel('y pos (cm)')
    ax.set_ylabel('Evidence weight on decision')
    ax.set_xlim([0, 200])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # if axisHandle is None:
        # plt.show()

    return ax

# ==============================================================================
# Analyze control and psychometrics and logistic regression by mouse + per-condition overall performance (laser and control)
def behav_by_mouse(lg, doPlot=True, logRegr_nbins=params['logRegr_nbins'], psych_bins=params['psych_bins']):

    """
    mouse_data = behav_by_mouse(lg, doPlot=True, logRegr_nbins=params['logRegr_nbins'], psych_bins=params['psych_bins'])
    computes logistic regression (evidence_logRegr) and psychometrics (psychometrics) by mouse
    returns dictionary with lists of corresponding analysis outputs
    """

    mouse_ids = np.unique(lg['mouseID'])
    num_mice  = np.size(mouse_ids)

    # do psychometrics and logistic regression (space) by mouse, control trials
    mouse_data = dict(psych=list(), logRegr=list(), mouse_ids= mouse_ids)
    for iMouse in range(num_mice):
        mouseidx = np.logical_and(~lg['laserON'],lg['mouseID'] == mouse_ids[iMouse])
        mouse_data['psych'].append(psychometrics(lg['choice'][mouseidx],lg['nCues_RminusL'][mouseidx], psych_bins))
        mouse_data['logRegr'].append(evidence_logRegr(lg['choice'][mouseidx], lg['cuePos_R'][mouseidx], lg['cuePos_L'][mouseidx], logRegr_nbins))

    # Analyze % correct and speed by mouse, with or without inactivation
    mouse_data['percCorrect'] = multiArea_percCorrect_bymouse(lg)
    mouse_data['speed']       = multiArea_speed_bymouse(lg)

    # plot
    if doPlot:
        nr, nc = utils.subplot_org(num_mice,6)

        for iMouse in range(num_mice):
            plt.figure(1,figsize=[16,8])
            ax = plt.subplot(nr,nc,iMouse+1)
            ax = plot_psych(mouse_data['psych'][iMouse],'k', [.5, .5, .5], ax)
            ax.set_title("Mouse %i" %mouse_ids[iMouse])

            plt.figure(2,figsize=[16,8])
            ax = plt.subplot(nr,nc,iMouse+1)
            ax = plot_logRegr(mouse_data['logRegr'][iMouse],'k',ax)
            ax.set_title("Mouse %i" %mouse_ids[iMouse])

    return mouse_data

# ==============================================================================
# Analyze % correct by mouse, with or without inactivation
def multiArea_percCorrect_bymouse(lg):

    """
    percCorrect = multiArea_percCorrect_bymouse(lg)
    takes flattened laser log dictionary and returns dictionary with control and laserON
    performance by mouse for each area / epoch combination
    """

    # initialize variables
    num_areas   = len(params['inact_locations'])
    num_epochs  = len(params['inact_epochs'])
    percCorrect = {
                   'data_lbl'          : list(),
                   'overall_ctrl'      : list(),
                   'overall_lsr'       : list(),
                   'overall_diff'      : list(),
                   'overall_diff_pval' : list(),
                   'mouse_ids'         : list(),
                   }

    # loop over areas / epochs
    for area in params['inact_locations']:
        for epoch in params['inact_epochs']:

            # get condition-specific log and loop over mice
            sublg = lg_singleCondition(lg,area,epoch)
            mice  = np.unique(sublg['mouseID'])
            nmice = np.size(mice)

            percCorrect['data_lbl'].append([area, epoch])
            percCorrect['mouse_ids'].append(mice)

            thisctrl = np.zeros(nmice)
            thislsr  = np.zeros(nmice)
            for iMouse in range(nmice):
                ctrlidx          = np.logical_and(sublg['mouseID'] == mice[iMouse],~sublg['laserON'])
                lsridx           = np.logical_and(sublg['mouseID'] == mice[iMouse],sublg['laserON'])
                thisctrl[iMouse] = np.sum(sublg['choice'][ctrlidx] == sublg['trialType'][ctrlidx]) / np.sum(ctrlidx)
                thislsr[iMouse]  = np.sum(sublg['choice'][lsridx] == sublg['trialType'][lsridx]) / np.sum(lsridx)

            thisdiff = thislsr - thisctrl
            _ ,thisp = sp.stats.wilcoxon(thisdiff,alternative='less')

            percCorrect['overall_ctrl'].append(thisctrl)
            percCorrect['overall_lsr'].append(thislsr)
            percCorrect['overall_diff'].append(thisdiff)
            percCorrect['overall_diff_pval'].append(thisp)

            del sublg

        # FDR correction
        percCorrect['overall_diff_isSig'], _ = list(utils.FDR(np.array(percCorrect['overall_diff_pval']),alpha=0.05))

    return percCorrect

# ==============================================================================
# Analyze % correct by mouse, with or without inactivation
def multiArea_speed_bymouse(lg,numiter=params['multiArea_boot_numiter']):

    """
    speed = multiArea_speed_bymouse(lg)
    takes flattened laser log dictionary and returns dictionary with control and laserON
    speed by mouse for each area / epoch combination, plus bootstrapping per condition
    """

    # initialize variables
    num_areas   = len(params['inact_locations'])
    num_epochs  = len(params['inact_epochs'])
    speed       = {
                   'data_lbl'     : list(),
                   'ctrl'         : list(),
                   'ctrl_mean'    : list(),
                   'ctrl_std'     : list(),
                   'lsr'          : list(),
                   'lsr_mean'     : list(),
                   'lsr_std'      : list(),
                   'diff'         : list(),
                   'diff_mean'    : list(),
                   'diff_std'     : list(),
                   'ratio'        : list(),
                   'ratio_mean'   : list(),
                   'ratio_std'    : list(),
                   'pval'         : list(),
                   'pval_mice'    : list(),
                   'mouse_ids'    : list(),
                   }

    # loop over areas / epochs
    for area in params['inact_locations']:
        for epoch in params['inact_epochs']:

            # get condition-specific log and loop over mice
            sublg = lg_singleCondition(lg,area,epoch)
            mice  = np.unique(sublg['mouseID'])
            nmice = np.size(mice)

            speed['data_lbl'].append([area, epoch])
            speed['mouse_ids'].append(mice)

            thisctrl = np.zeros(nmice)
            thislsr  = np.zeros(nmice)
            yrange   = utils.yRangeFromEpochLabel(epoch)

            for iMouse in range(nmice):
                ctrlidx          = np.where(np.logical_and(sublg['mouseID'] == mice[iMouse],~sublg['laserON']))[0]
                lsridx           = np.where(np.logical_and(sublg['mouseID'] == mice[iMouse],sublg['laserON']))[0]
                speed_ctrl       = np.zeros(np.size(ctrlidx))
                speed_lsr        = np.zeros(np.size(lsridx))

                for iCtrl in range(np.size(ctrlidx)):
                    t1    = np.where(sublg['pos'][ctrlidx[iCtrl]][:,1]>=yrange[0])[0][0]
                    t2    = np.where(sublg['pos'][ctrlidx[iCtrl]][:,1]<yrange[1])[0][-1]
                    dur   = sublg['time'][ctrlidx[iCtrl]][t2] - sublg['time'][ctrlidx[iCtrl]][t1]
                    xd    = sublg['pos'][ctrlidx[iCtrl]][t1:t2,0]
                    yd    = sublg['pos'][ctrlidx[iCtrl]][t1:t2,1]
                    displ = np.sum(np.sqrt(np.diff(xd)**2+np.diff(yd)**2))
                    speed_ctrl[iCtrl] = displ / dur

                for iLsr in range(np.size(lsridx)):
                    t1    = np.where(sublg['pos'][lsridx[iLsr]][:,1]>=yrange[0])[0][0]
                    t2    = np.where(sublg['pos'][lsridx[iLsr]][:,1]<yrange[1])[0][-1]
                    dur   = sublg['time'][lsridx[iLsr]][t2] - sublg['time'][lsridx[iLsr]][t1]
                    xd    = sublg['pos'][lsridx[iLsr]][t1:t2,0]
                    yd    = sublg['pos'][lsridx[iLsr]][t1:t2,1]
                    displ = np.sum(np.sqrt(np.diff(xd)**2+np.diff(yd)**2))
                    speed_lsr[iLsr] = displ / dur

                thisctrl[iMouse] = np.mean(speed_ctrl)
                thislsr[iMouse]  = np.mean(speed_lsr)

            thisdiff  = thislsr - thisctrl
            thisratio = (thislsr - thisctrl) / thisctrl
            _ ,thisp  = sp.stats.wilcoxon(thisdiff)

            speed['ctrl'].append(thisctrl)
            speed['lsr'].append(thislsr)
            speed['diff'].append(thisdiff)
            speed['ratio'].append(thisratio*100)
            speed['pval_mice'].append(thisp)

            # now bootstrap overall
            thisctrl  = np.zeros(numiter)
            thislsr   = np.zeros(numiter)
            numtrials = np.size(sublg['choice'])
            trialidx  = np.arange(numtrials)

            # sample with replacement and fit logistic model
            for iBoot in range(numiter):
                idx        = np.random.choice(trialidx, size=numtrials, replace=True)
                laserON    = sublg['laserON'][idx]
                pos        = sublg['pos'][idx]
                ttime      = sublg['time'][idx]
                ctrlidx    = np.where(~laserON)[0]
                lsridx     = np.where(laserON)[0]
                speed_ctrl = np.zeros(np.size(ctrlidx))
                speed_lsr  = np.zeros(np.size(lsridx))

                for iCtrl in range(np.size(ctrlidx)):
                    t1    = np.where(pos[ctrlidx[iCtrl]][:,1]>=yrange[0])[0][0]
                    t2    = np.where(pos[ctrlidx[iCtrl]][:,1]<yrange[1])[0][-1]
                    dur   = ttime[ctrlidx[iCtrl]][t2] - ttime[ctrlidx[iCtrl]][t1]
                    xd    = pos[ctrlidx[iCtrl]][t1:t2,0]
                    yd    = pos[ctrlidx[iCtrl]][t1:t2,1]
                    displ = np.sum(np.sqrt(np.diff(xd)**2+np.diff(yd)**2))
                    speed_ctrl[iCtrl] = displ / dur

                for iLsr in range(np.size(lsridx)):
                    t1    = np.where(pos[lsridx[iLsr]][:,1]>=yrange[0])[0][0]
                    t2    = np.where(pos[lsridx[iLsr]][:,1]<yrange[1])[0][-1]
                    dur   = ttime[lsridx[iLsr]][t2] - ttime[lsridx[iLsr]][t1]
                    xd    = pos[lsridx[iLsr]][t1:t2,0]
                    yd    = pos[lsridx[iLsr]][t1:t2,1]
                    displ = np.sum(np.sqrt(np.diff(xd)**2+np.diff(yd)**2))
                    speed_lsr[iLsr] = displ / dur

                thisctrl[iBoot] = np.mean(speed_ctrl)
                thislsr[iBoot]  = np.mean(speed_lsr)

            # bootstrap stats
            thisdiff  = thislsr - thisctrl
            thisratio = (thislsr - thisctrl) / thisctrl

            if np.mean(thisdiff) <= 0:
                thisp = np.sum(thisdiff>0) / numiter
            else:
                thisp = np.sum(thisdiff<0) / numiter

            speed['ctrl_mean'].append(np.mean(thisctrl))
            speed['ctrl_std'].append(np.std(thisctrl,ddof=1))
            speed['lsr_mean'].append(np.mean(thislsr))
            speed['lsr_std'].append(np.std(thislsr,ddof=1))
            speed['diff_mean'].append(np.mean(thisdiff))
            speed['diff_std'].append(np.std(thisdiff,ddof=1))
            speed['ratio_mean'].append(np.mean(thisratio))
            speed['ratio_std'].append(np.std(thisratio,ddof=1))
            speed['pval'].append(thisp)

            del sublg

        # FDR correction
        speed['isSig'], _      = list(utils.FDR(np.array(speed['pval']),alpha=0.025))
        speed['isSig_mice'], _ = list(utils.FDR(np.array(speed['pval_mice']),alpha=0.025))

    return speed

# ==============================================================================
# Flag bad trials based on view angle criteria etc
def findBadTrials(lg,removeThetaOutlierMode=params['removeThetaOutlierMode'],removeZeroEvidTrials=params['removeZeroEvidTrials']):

    """
    isBadTrial = findBadTrials(lg,removeThetaOutlierMode=params['removeThetaOutlierMode'],removeZeroEvidTrials=params['removeZeroEvidTrials'])
    takes flattened behavioral log structure and returns a 1 x num_trials boolean array
    where True indicates bad trials according to view angle criteria (refer to function
    body for details) and whether tower deletion control trials where #r = #l should be removed
    """

    # view angle outliers
    total_trials    = len(lg['choice'])
    if removeThetaOutlierMode == 'soft': # exclude top 10 percentile of bad theta trials
        theta_std   = np.array([ np.std(lg['viewAngle_byYpos'][iTrial]) for iTrial in np.arange(0,total_trials) ])
        large_theta = theta_std > np.percentile(theta_std,95)

    elif removeThetaOutlierMode == 'stringent': # soft + some absolute thresholds on theta values
        theta_std   = np.array([ np.std(lg['viewAngle_byYpos'][iTrial]) for iTrial in np.arange(0,total_trials) ])
        large_theta = theta_std > np.percentile(theta_std,90)
        norm_theta  = np.array([ lg['viewAngle_byYpos'][iTrial]/lg['viewAngle_byYpos'][iTrial][-1] for iTrial in np.arange(0,total_trials) ])
        bad_norm    = np.array([ np.any(norm_theta[iTrial][0:200]>.4) or np.any(norm_theta[iTrial][200:250]>.5) or np.any(norm_theta[iTrial][250:290]>.9)
                               for iTrial in np.arange(0,total_trials) ])
        large_theta = large_theta | bad_norm

    elif removeThetaOutlierMode == 'none': # no exclusion
        large_theta = np.zeros(total_trials) > 1

    # remove #R = #L trials (from some control conditions)
    if removeZeroEvidTrials == True:
      badRL       = np.array([ np.size(lg['cuePos_R'][iTrial]) == np.size(lg['cuePos_L'][iTrial]) for iTrial in np.arange(0,total_trials) ])
    else:
      badRL       = np.zeros(total_trials) > 1

    isBadTrial    = large_theta | badRL
    isBadTrial    = np.logical_or(isBadTrial,lg['loclbl'] == 'unknown')

    return isBadTrial

# ==============================================================================
# Flag bad mice based on logistic regression
def findBadMice(lg, numbins=params['logRegr_nbins'], numiter=params['logRegr_boot_numiter']):

    """
    isBadMouse, mouse_ids = findBadMice(lg, numbins=params['logRegr_nbins'], numiter=params['logRegr_boot_numiter'])
    takes flattened behavioral log structure and returns a 1 x num_mice boolean array
    where True indicates bad mice according to logistic regression (having any coefficient not statistically != 0),
    and a list of mouse IDs corresponding to the boolean array
    """

    mouse_ids  = np.unique(lg['mouseID'])
    num_mice   = np.size(mouse_ids)
    isBadMouse = np.zeros(num_mice) > 1

    # do logistic regression by mouse, bad mice are ones that have logistic regression
    # coefficients not significantly different from zero
    for iMouse in range(num_mice):
        mouseidx = np.logical_and(~lg['laserON'], lg['mouseID'] == mouse_ids[iMouse])
        lr       = boot_logRegr(lg['choice'][mouseidx], lg['cuePos_R'][mouseidx], lg['cuePos_L'][mouseidx], numbins, numiter)
        isBadMouse[iMouse] = sum(lr['coeff_isSig']) < numbins

    return (isBadMouse, mouse_ids)

# ==============================================================================
# cleanup log by removing bad mice and bad trials
def cleanupBehavLog(lg, removeThetaOutlierMode = params['removeThetaOutlierMode'], removeZeroEvidTrials = params['removeZeroEvidTrials'], excludeBadMice = params['excludeBadMice'], badmice=None):

    """
    lg_clean = cleanupBehavLog(lg, removeThetaOutlierMode = params['removeThetaOutlierMode'], removeZeroEvidTrials = params['removeZeroEvidTrials'], excludeBadMice = params['excludeBadMice'], badmice=None)
    deletes bad trials from flattened behavioral log structure based on trial and mouse
    selection criteria. See findBadTrials and findBadMice for details
    """

    # get bad trials
    isBadTrial = findBadTrials(lg,removeThetaOutlierMode,removeZeroEvidTrials)

    # get bad mice
    if excludeBadMice:
        if badmice is None:
            isBadMouse, mouse_ids = findBadMice(lg)
            badmice               = mouse_ids[isBadMouse]

        for iMouse in range(np.size(badmice)):
            trialidx = lg['mouseID'] == badmice[iMouse]
            isBadTrial[trialidx] = True

    # remove trials
    lg_clean  = dict() #copy.deepcopy(lg)
    fields    = list(lg.keys())
    for iF in range(len(fields)):
        thisvar = copy.deepcopy(lg[fields[iF]])
        if np.size(thisvar) == np.size(isBadTrial):
            lg_clean[fields[iF]] = thisvar[~isBadTrial]
        else:
            lg_clean[fields[iF]] = thisvar

    return lg_clean

# ==============================================================================
# Bootstrap Logistic regression, psychometrics and overall % correct over trials
# compare laser and control trials for each region being inactivated
def multiArea_boot_perf(lg, doParallel=True, numiter=params['multiArea_boot_numiter'], numbins=params['logRegr_nbins'], psych_bins=params['psych_bins']):

    """
    inact_effects = multiArea_boot_perf(lg, doParallel=True, numiter=params['multiArea_boot_numiter'], numbins=params['logRegr_nbins'], psych_bins=params['psych_bins'])
    analyzes all area / epoch pairs by bootstrapping over trials and calculating inactivation
    effects on logistic regression of choice vs evidence, psychometrics, and overall percent correct
    INPUT
        lg: flattened behavioral log object
        doParallel: use multiprocessing (parallelizes over inactivation conditions)
        numiter: number of bootstrapping iterations
        numbins: number of bins for logistic regression
        psych_bins: bins of evidence for psychometrics
    OUTPUT
        inact_effects: dictionary with averages, stds, pvals etc for
            logistic regression, psychometrics and percent correct per inactivation condition
            (as arrays with rows corresponding to data_lbl), for control and laser trials,
            as well as their difference (or ratio where applicable)
    """

    # initialize output dictionary
    inact_effects = {
                    'logRegr'      : dict(),                     # analysis of logistic regression
                    'trig_logRegr' : dict(),                     # laser-triggered logistic regression
                    'percCorrect'  : dict(),                     # analysis of overall performance
                    'psych'        : dict(),                     # analysis of psychometrics
                    'epoch_list'   : params['inact_epochs'],     # inactivation epoch labels
                    'area_list'    : params['inact_locations'],  # inactivation location labels
                    'data_lbl'     : list(),                     # labels for each epoch - area pair
                    'num_iter'     : numiter,                    # number of bootstrapping iterations
                    }

    # organize input for parallel processing, will do over conditions
    num_areas  = len(inact_effects['area_list'])
    num_epochs = len(inact_effects['epoch_list'])
    num_cond   = num_areas * num_epochs
    data_list  = list()

    print('organizing data...')
    for iArea in range(num_areas):
        for iEpoch in range(num_epochs):
            # trim to single area lg
            thisarea  = inact_effects['area_list'][iArea]
            thisepoch = inact_effects['epoch_list'][iEpoch]
            thislg    = lg_singleCondition(lg, thisarea, thisepoch)

            # compile data list
            thislist  = [thislg, numiter, numbins, psych_bins, [thisarea, thisepoch]]
            data_list.append(thislist)
            inact_effects['data_lbl'].append([thisarea, thisepoch])
            del thislg, thislist

    # iterate over conditions, in parallel or othewrise
    if doParallel == True:
        pool    = mp.Pool(mp.cpu_count()) # start parallel poolobj with all available cores
        results = pool.map(singleArea_boot_perf,data_list) # run core function in parallel
        pool.close() # close parpool
    else:
        results = list()
        for iCond in range(num_cond):
            results.append(singleArea_boot_perf(data_list[iCond]))

    # organize results as condition x data matrices, find first non-empty structure
    for iCond in range(num_cond):
        if bool(results[iCond]['logRegr']):
            break

    inact_effects['logRegr']['evidence_vals'] = results[iCond]['logRegr']['evidence_vals']
    inact_effects['psych']['psych_bins']      = psych_bins
    inact_effects['psych']['fit_x']           = results[iCond]['psych']['fit_x']

    logRegr_vars = ['coeffs_ctrl_mean','coeffs_ctrl_std','coeffs_lsr_mean','coeffs_lsr_std',      \
                    'coeffs_diff_mean','coeffs_diff_std','coeffs_diff_p','coeffs_diff_isSig',     \
                    'coeffs_ratio_mean','coeffs_ratio_std','coeffs_ratio_p','coeffs_ratio_isSig', \
                    ]
    psych_vars   = ['fit_y_ctrl_mean','fit_y_ctrl_std','slope_ctrl_mean','slope_ctrl_std',        \
                    'fit_y_lsr_mean','fit_y_lsr_std','slope_lsr_mean','slope_lsr_std',            \
                    'slope_diff_mean','slope_diff_std','slope_diff_p',                            \
                    'P_wentRight_ctrl_mean','P_wentRight_ctrl_std','P_wentRight_lsr_mean',        \
                    'P_wentRight_lsr_std','P_wentRight_diff_mean','P_wentRight_diff_std']
    pc_vars      = ['overall_ctrl_mean','overall_ctrl_std','overall_lsr_mean','overall_lsr_std',  \
                    'overall_diff_mean','overall_diff_std','overall_diff_p',                      \
                    'easy_ctrl_mean','easy_ctrl_std','easy_lsr_mean','easy_lsr_std',              \
                    'easy_diff_mean','easy_diff_std','easy_diff_p',                               \
                    'hard_ctrl_mean','hard_ctrl_std','hard_lsr_mean','hard_lsr_std',              \
                    'hard_diff_mean','hard_diff_std','hard_diff_p',                               \
                   ]

    # create arrays
    for var in logRegr_vars:
        if var.find('coeffs') < 0:
            inact_effects['logRegr'][var] = np.ones((num_cond,1))*np.nan
        else:
            inact_effects['logRegr'][var] = np.ones((num_cond,numbins))*np.nan
    for var in psych_vars:
        if var.find('P_') >= 0:
            inact_effects['psych'][var]   = np.ones((num_cond,np.size(psych_bins)-1))*np.nan
        elif var.find('fit_') >= 0:
            inact_effects['psych'][var]   = np.ones((num_cond,301))*np.nan
        else:
            inact_effects['psych'][var]   = np.ones((num_cond,1))*np.nan
    for var in pc_vars:
        inact_effects['percCorrect'][var] = np.ones((num_cond,1))*np.nan

    # fill in
    for iCond in range(num_cond):
        if bool(results[iCond]['logRegr']):
            for var in logRegr_vars:
                inact_effects['logRegr'][var][iCond,:]     = results[iCond]['logRegr'][var]
            for var in psych_vars:
                inact_effects['psych'][var][iCond,:]       = results[iCond]['psych'][var]
            for var in pc_vars:
                inact_effects['percCorrect'][var][iCond,:] = results[iCond]['percCorrect'][var]

    # do FDR correction
    for var in psych_vars:
        # do tests only for differences
        if var.find('diff_p') != -1:
            inact_effects['psych']["{}_isSig".format(var)], inact_effects['psych']["{}_alpha_correct".format(var)] \
              = utils.FDR(inact_effects['psych'][var][:,0],params['alpha'])

    for var in pc_vars:
        # do tests only for differences
        if var.find('diff_p') != -1:
            inact_effects['percCorrect']["{}_isSig".format(var)], inact_effects['percCorrect']["{}_alpha_correct".format(var)] \
              = utils.FDR(inact_effects['percCorrect'][var][:,0],params['alpha'])


    return inact_effects

# ==============================================================================
# core computations of multiArea_boot_perf, takes a list of data from multiArea_boot_perf()
# see above for details
def singleArea_boot_perf(data_in):

    """
    inact_effects = singleArea_boot_perf(data_in)
    called by multiArea_boot_perf()
    core bootstrapping computations
    """

    # parse inputs
    thislg    = data_in[0]
    numiter   = data_in[1]
    numbins   = data_in[2]
    psychbins = data_in[3]
    lbl       = data_in[4]

    start_time = time.time()
    print("bootstrapping {}, {}".format(lbl[0],lbl[1]))

    # initialize matrices
    logRegr_vars = ['coeffs_ctrl','coeffs_lsr','coeffs_diff','coeffs_ratio']
    psych_vars   = ['fit_y_ctrl','slope_ctrl','P_wentRight_ctrl',  \
                    'fit_y_lsr','slope_lsr','P_wentRight_lsr',     \
                    'slope_diff','P_wentRight_diff']
    pc_vars      = ['overall_ctrl','overall_lsr','overall_diff',   \
                    'easy_ctrl','easy_lsr','easy_diff',            \
                    'hard_ctrl','hard_lsr','hard_diff']

    logRegr     = dict()
    psych       = dict()
    percCorrect = dict()
    for var in logRegr_vars:
        if var.find('coeffs') < 0:
            logRegr[var] = np.ones((numiter,1))*np.nan
        else:
            logRegr[var] = np.ones((numiter,numbins))*np.nan
    for var in psych_vars:
        if var.find('P_') >= 0:
            psych[var]   = np.ones((numiter,np.size(psychbins)-1))*np.nan
        elif var.find('fit_') >= 0:
            psych[var]   = np.ones((numiter,301))*np.nan
        else:
            psych[var]   = np.ones((numiter,1))*np.nan
    for var in pc_vars:
        percCorrect[var] = np.ones((numiter,1))*np.nan

    # bootstrap over mice, sessions, trials
    numtrials = np.size(thislg['choice'])
    trialidx  = np.arange(numtrials)

    # trial difficulty is \delta / total
    nr           = np.array([np.size(thislg['cuePos_R'][iTrial]) for iTrial in range(numtrials)])
    nl           = np.array([np.size(thislg['cuePos_L'][iTrial]) for iTrial in range(numtrials)])
    total_towers = nr + nl
    trial_diff   = thislg['nCues_RminusL'] / total_towers
    isHard       = trial_diff > np.median(trial_diff)

    # bootstrap over mice, sessions, trials
    for iBoot in range(numiter):
        idx           = np.random.choice(trialidx, size=numtrials, replace=True)
        choice        = thislg['choice'][idx]
        laserON       = thislg['laserON'][idx]
        cuePos_L      = thislg['cuePos_L'][idx]
        cuePos_R      = thislg['cuePos_R'][idx]
        nCues_RminusL = thislg['nCues_RminusL'][idx]
        trialType     = thislg['trialType'][idx]
        hard          = isHard[idx]

        if sum(laserON) == 0:
            continue

        # percent correct
        percCorrect['overall_ctrl'][iBoot,:] = sum(choice[~laserON] == trialType[~laserON]) / sum(~laserON)
        percCorrect['overall_lsr'][iBoot,:]  = sum(choice[laserON] == trialType[laserON]) / sum(laserON)
        percCorrect['overall_diff'][iBoot,:] = percCorrect['overall_lsr'][iBoot,:] - percCorrect['overall_ctrl'][iBoot,:]
        percCorrect['easy_ctrl'][iBoot,:]    = sum(choice[~laserON & ~isHard] == trialType[~laserON & ~isHard]) / sum(~laserON & ~isHard)
        percCorrect['easy_lsr'][iBoot,:]     = sum(choice[laserON & ~isHard] == trialType[laserON & ~isHard]) / sum(laserON & ~isHard)
        percCorrect['easy_diff'][iBoot,:]    = percCorrect['easy_lsr'][iBoot,:] - percCorrect['easy_ctrl'][iBoot,:]
        percCorrect['hard_ctrl'][iBoot,:]    = sum(choice[~laserON & isHard] == trialType[~laserON & isHard]) / sum(~laserON & isHard)
        percCorrect['hard_lsr'][iBoot,:]     = sum(choice[laserON & isHard] == trialType[laserON & isHard]) / sum(laserON & isHard)
        percCorrect['hard_diff'][iBoot,:]    = percCorrect['hard_lsr'][iBoot,:] - percCorrect['hard_ctrl'][iBoot,:]

        # logistic regression
        lr_ctrl = evidence_logRegr(choice[~laserON], cuePos_R[~laserON], cuePos_L[~laserON], numbins)
        lr_lsr  = evidence_logRegr(choice[laserON], cuePos_R[laserON], cuePos_L[laserON], numbins)

        logRegr['coeffs_ctrl'][iBoot,:]  = lr_ctrl['coeff']
        logRegr['coeffs_lsr'][iBoot,:]   = lr_lsr['coeff']
        logRegr['coeffs_diff'][iBoot,:]  = lr_lsr['coeff'] - lr_ctrl['coeff']
        logRegr['coeffs_ratio'][iBoot,:] = lr_lsr['coeff'] / lr_ctrl['coeff']

        # psychometrics
        psych_ctrl = psychometrics(choice[~laserON],nCues_RminusL[~laserON], psychbins)
        psych_lsr  = psychometrics(choice[laserON],nCues_RminusL[laserON], psychbins)

        psych['fit_y_ctrl'][iBoot,:]       = psych_ctrl['fit_y']
        psych['slope_ctrl'][iBoot,:]       = psych_ctrl['slope']
        psych['P_wentRight_ctrl'][iBoot,:] = np.transpose(psych_ctrl['P_wentRight'][:,0])
        psych['fit_y_lsr'][iBoot,:]        = psych_lsr['fit_y']
        psych['slope_lsr'][iBoot,:]        = psych_lsr['slope']
        psych['P_wentRight_lsr'][iBoot,:]  = np.transpose(psych_lsr['P_wentRight'][:,0])
        psych['P_wentRight_diff'][iBoot,:] = psych['P_wentRight_lsr'][iBoot,:] - psych['P_wentRight_ctrl'][iBoot,:]
        psych['slope_diff'][iBoot,:]       = psych_lsr['slope'] - psych_ctrl['slope']


    # do stats (mean, std, pvals)
    # pvals are proportion of bootstrapping iterations where value increases, i.e.
    # one-sided test of the hypothesis that inactivation decreases values
    inact_effects = dict(logRegr = dict(), psych = dict(), percCorrect = dict())

    if 'lr_ctrl' in locals():
        inact_effects['logRegr']['evidence_vals'] = lr_ctrl['evidence_vals']
        inact_effects['psych']['fit_x']           = psych_ctrl['fit_x']

        # logistic regression
        for var in logRegr_vars:
            inact_effects['logRegr']["{}_mean".format(var)] = np.nanmean(logRegr[var],axis=0)
            inact_effects['logRegr']["{}_std".format(var)]  = np.nanstd(logRegr[var],axis=0,ddof=1)

            # do tests only for differences and ratios
            if var.find('diff') != -1:
                inact_effects['logRegr'][var] = logRegr[var]

                pvals = np.zeros(numbins)
                for iBin in range(numbins):
                    pvals[iBin] = sum(logRegr[var][:,iBin] >= 0) / numiter
                isSig, alpha_correct = utils.FDR(pvals,params['alpha'])
                inact_effects['logRegr']["{}_p".format(var)]     = pvals
                inact_effects['logRegr']["{}_isSig".format(var)] = isSig
                inact_effects['logRegr']["{}_alpha_correct".format(var)] = alpha_correct

            elif var.find('ratio') != -1:
                pvals = np.zeros(numbins)
                for iBin in range(numbins):
                    pvals[iBin] = sum(logRegr[var][:,iBin] >= 1) / numiter
                isSig, alpha_correct = utils.FDR(pvals,params['alpha'])
                inact_effects['logRegr']["{}_p".format(var)]     = pvals
                inact_effects['logRegr']["{}_isSig".format(var)] = isSig
                inact_effects['logRegr']["{}_alpha_correct".format(var)] = alpha_correct

        # psychometrics
        for var in psych_vars:
            inact_effects['psych']["{}_mean".format(var)] = np.nanmean(psych[var],axis=0)
            inact_effects['psych']["{}_std".format(var)]  = np.nanstd(psych[var],axis=0,ddof=1)

            # do tests only for differences
            if var.find('slope_diff') != -1:
                inact_effects['psych']["{}_p".format(var)] = sum(psych[var] >= 0) / numiter

        # percent correct
        for var in pc_vars:
            inact_effects['percCorrect']["{}_mean".format(var)] = np.nanmean(percCorrect[var],axis=0)
            inact_effects['percCorrect']["{}_std".format(var)]  = np.nanstd(percCorrect[var],axis=0,ddof=1)

            # do tests only for differences
            if var.find('diff') != -1:
                inact_effects['percCorrect']["{}_p".format(var)] = sum(percCorrect[var] >= 0) / numiter

        # wrap up
        end_time = time.time()
        print("done after {: 1.2g} min".format((end_time-start_time)/60))

    else:
        print('no laser trials found, returning empty dictionary')

    return inact_effects

# ==============================================================================
# organize and call appropriate mixed regression function
def run_mixed_time_regression(lg,regr_type,regr_params=params['regr_params'],savePath=params['savePath'],overWrite=False):

    """
    lr_time, lr_filebasename = run_mixed_time_regression(lg,regr_type,regr_params=params['regr_params'],savePath=params['savePath'],overWrite=False)
    takes flattened behavioral log structure and returns a dictionary with a
    list of mixed-effects regression results per area (lr_time).
    lr_filebasename is filename without path or extension, e.g. for saving figs

    INPUT
    regr_type can be 'combinedEpochs' (all epochs go into single model as a random effect),
                     'singleEpochs' (each epoch in a separate model)
                     'epochSets' (hybrid: early cue, late cue, delay epochs get combined separately)
    regr_params is a dictionary of analysis parameters, optional input
    savePath is directory where data should be saved, Optional
    overWrite is boolean flag to overwrite existing data (default False). If False, existing data matching model parameters will be loaded
    """

    # generate filename and load if it exists (unless overwrite is set to True)
    if params['excludeBadMice']:
        suffix      = 'goodMice'
    else:
        suffix      = 'allMice'

    lr_filebasename = 'logRegr_mixedEffects_time_bin{:d}_{}_{}_{}'.format( \
                       regr_params['bins_per_sec'][0],regr_params['method'],regr_type,suffix)
    lr_fn           = '{}{}.hdf5'.format(savePath,lr_filebasename)

    if ~overWrite:
        if path.exists(lr_fn):
            print('found existing file, loading from disk')
            lr_time = fl.load(lr_fn)
            return lr_time, lr_filebasename

    # otherwise run desired regression
    if regr_type == 'combinedEpochs':
        lr_time = batch_mixed_logRegr_time_combined(lg,regr_params)

    elif regr_type == 'singleEpochs':
        lr_time = batch_mixed_logRegr_time_split(lg,regr_params,combine_epochs=False)

    elif regr_type == 'epochSets':
        lr_time = batch_mixed_logRegr_time_split(lg,regr_params,combine_epochs=True)

    else:
        print('warning: unknown regr_type, returning nothing')
        return None, lr_filebasename

    # save
    fl.save(lr_fn,lr_time)

    return lr_time, lr_filebasename

# ==============================================================================
# batch mixed-effects logistic regression in time, combining across epochs
def batch_mixed_logRegr_time_combined(lg,regr_params=params['regr_params']):

    """
    inact_effects = batch_mixed_logRegr_time_combined(lg,regr_params=params['regr_params'])
    takes flattened behavioral log structure and returns a dictionary with a
    list of mixed-effects regression results per area. In this case epochs are
    combined into a single regression model
    regr_params is a dictionary of analysis parameters, optional input
    """

    # initialize output dictionary
    inact_effects = {
                    'results'    : list(),                     # analysis of logistic regression
                    'area_list'  : params['inact_locations'],  # inactivation location labels
                    'regr_params': regr_params,                # see above
                    }

    # organize input for parallel processing, will do over conditions
    epoch_list = params['inact_epochs']
    num_areas  = len(inact_effects['area_list'])
    cm_range   = [utils.yRangeFromEpochLabel(epoch_list[iE]) for iE in range(len(epoch_list))]

    for iArea in range(num_areas):
        # trim to single area lg
        start_time = time.time()
        print('area {}/{}...'.format(iArea+1, num_areas))
        print('\torganizing data...')
        thisarea  = inact_effects['area_list'][iArea]
        thislg    = lg_singleCondition(lg, thisarea, epoch_list)

        print('\tfitting model',end='')
        inact_effects['results'].append(logRegr_time_combined_mixed(thislg,cm_range,regr_params=regr_params))

        end_time = time.time()
        print('\tdone after {: 1.2g} min'.format((end_time-start_time)/60))

        del thislg

    return inact_effects

# ==============================================================================
# mixed-effects logistic regression in time for single area, combining across epochs
def logRegr_time_combined_mixed(sublg,lsr_epoch_cm,regr_params):

    """
    logRegr = logRegr_time_combined_mixed(sublg,lsr_epoch_cm,regr_params)
    mixed-effects logistic regression in time for single area, combining across epochs
    sublg is log containing data from single area
    lsr_epoch_cm is list with cm ranges of all inactivation epochs in the log
    regre_params is dictionary with analysis parameters
    retruns logRegr, dictionary with model object, fitted cofficients, predictor labels,
    goodness of fit, random effects
    """

    # epoch lengths in cm
    lCue = 200
    lMem = 100

    # just convenience
    choice        = sublg['choice']
    cueOnset_R    = sublg['cueOnset_R']
    cueOnset_L    = sublg['cueOnset_L']
    trial_time    = sublg['time']
    pos           = sublg['pos']
    lsr_epoch_vec = sublg['laserEpoch']
    mouseID       = sublg['mouseID']
    laserON       = sublg['laserON']
    maxt          = regr_params['max_t']
    bins_per_sec  = regr_params['bins_per_sec']

    # calculate evidence binning and initialize variables
    ntrials    = np.size(choice)
    numbins    = []
    bins       = []
    bincenters = []
    for iBin in range(len(maxt)):
        numbins.append(int(maxt[iBin]*bins_per_sec[iBin]))
        bins.append(np.linspace(0, maxt[iBin], int(numbins[iBin]+1)))
        bincenters.append(None)

    # build predictor matrix
    RminusLmat = np.zeros((ntrials, np.sum(numbins)))  # delta_towers
    epochID    = np.zeros(ntrials)  # epoch
    for iTrial in range(ntrials):
        # define inactivation epoch
        if ~laserON[iTrial]:
            # figure out closest laser epoch withn +/- 500 trials, use that as dummy epoch
            idx      = np.arange(np.max([iTrial-500,0]),np.min([iTrial+500,ntrials]))
            idxc     = np.abs(idx-iTrial+1)
            islsr    = laserON[idx]
            closeidx = np.where(np.logical_and(idxc==np.min(idxc[islsr]),islsr))[0][0]
            this_ep  = utils.yRangeFromEpochLabel(lsr_epoch_vec[idx[closeidx]])
        else:
            this_ep     = utils.yRangeFromEpochLabel(lsr_epoch_vec[iTrial])
        epochID[iTrial] = lsr_epoch_cm.index(this_ep)

        # decide whether this trial has towers before, during or after laser
        run_pre    = this_ep[0] > 0
        run_during = this_ep[0] < lCue
        run_post   = this_ep[1] < lCue

        # evidence preceding laser onset
        if run_pre:
            epoch_onset = trial_time[iTrial][np.where(pos[iTrial][:, 1] < this_ep[0])[0][-1]]
            this_R      = cueOnset_R[iTrial]
            this_R      = epoch_onset - this_R[this_R < epoch_onset]
            this_L      = cueOnset_L[iTrial]
            this_L      = epoch_onset - this_L[this_L < epoch_onset]
            rhist, binedges = np.histogram(this_R, bins[0])
            lhist, _        = np.histogram(this_L, bins[0])
            RminusLmat[iTrial, range(numbins[0])] = rhist - lhist
            if bincenters[0] is None:
                bincenters[0] = binedges[:-1] + (binedges[1]-binedges[0])/2

        # evidence within laser
        if run_during:
            epoch_offset = trial_time[iTrial][np.where(pos[iTrial][:, 1] < this_ep[1])[0][-1]]
            this_R       = cueOnset_R[iTrial]
            this_R       = epoch_offset - this_R[this_R < epoch_offset]
            this_L       = cueOnset_L[iTrial]
            this_L       = epoch_offset - this_L[this_L < epoch_offset]
            rhist, binedges = np.histogram(this_R, bins[1])
            lhist, _        = np.histogram(this_L, bins[1])
            RminusLmat[iTrial, range(numbins[0],np.sum(numbins[0:1])+1)] = rhist - lhist
            if bincenters[1] is None:
                bincenters[1] = binedges[:-1] + (binedges[1]-binedges[0])/2

        # evidence after laser offset
        if run_post:
            epoch_offset = trial_time[iTrial][np.where(pos[iTrial][:, 1] > this_ep[1])[0][0]]
            this_R       = cueOnset_R[iTrial]
            this_R       = epoch_offset - this_R[this_R > epoch_offset]
            this_L       = cueOnset_L[iTrial]
            this_L       = epoch_offset - this_L[this_L > epoch_offset]
            rhist, binedges = np.histogram(-this_R, bins[2])
            lhist, _        = np.histogram(-this_L, bins[2])
            RminusLmat[iTrial, range(np.sum(numbins[0:1])+1,np.sum(numbins))] = rhist - lhist
            if bincenters[2] is None:
                bincenters[2] = binedges[:-1] + (binedges[1]-binedges[0])/2

    # remove trials with no towers within evidence bins
    bad_trials = np.where(np.sum(RminusLmat==0,axis=1)==np.size(RminusLmat,axis=1))
    RminusLmat = np.delete(RminusLmat,bad_trials,axis=0)
    choice     = np.delete(choice,bad_trials)
    laserON    = np.delete(laserON,bad_trials)
    epochID    = np.delete(epochID,bad_trials)
    mouseID    = np.delete(mouseID,bad_trials)

    if regr_params['method'] == 'Bayes':
        if regr_params['nfold_xval']>1:
            print('Warning: Cross validation not yet implemented for Binomial Bayes GLM')
        model, random_effects, formula, choice_pred, coeffs =               \
            fitBinomialBayes(choice, laserON, RminusLmat, mouseID, epochID, \
                             addLsrOffset=regr_params['addLsrOffset'], zscore=regr_params['zscore'])

    elif regr_params['method'] == 'Lmer_explicit':
        model, random_effects, formula, choice_pred, coeffs =          \
            fitLmer(choice, laserON, RminusLmat, mouseID, epochID,     \
                             addLsrOffset=regr_params['addLsrOffset'], \
                             zscore=regr_params['zscore'],nfold_xval=regr_params['nfold_xval'])

    else:
        print('Warning: fitting method not implemented')
        return

    # which evidence coefficient belongs to pre, during, post
    coeffs_by_epoch = []
    ct              = 0
    for iEpoch in range(len(bincenters)):
        coeffs_by_epoch.append([])
        for iBin in range(np.size(bincenters[iEpoch])):
            coeffs_by_epoch[iEpoch].append('b{}'.format(ct))
            ct = ct + 1

    # organize output
    logRegr = {
              'model_obj'          : model,           # full model object
              'formula'            : formula,         # model formula
              'coeff'              : coeffs,          # fitted coefficients, overall and separated by condition
              'evidence_vals'      : bincenters,      # center of evidence bins
              'evidence_vals_lbls' : coeffs_by_epoch, # cofficient labels corresponding to diferent evidence bins
              'random_effects'     : random_effects,  # dataframe with random effects info
              'choice_pred'        : choice_pred,     # proportion / correlation of choices correctly predicted by model (thresholded)
              }

    return logRegr

# ==============================================================================
# mixed-effects logistic regression in time for single area, single epochs or epoch sets (but not all epochs combined)
def batch_mixed_logRegr_time_split(lg,regr_params=params['regr_params'],combine_epochs=False):

    """
    inact_effects = batch_mixed_logRegr_time_split(lg,regr_params=params['regr_params'],combine_epochs=False)
    takes flattened behavioral log structure and returns a dictionary with a
    list of mixed-effects regression results per area and epoch.
    regr_params is a dictionary of analysis parameters, optional input
    combineEpochs is logical flag to group into early and late cue region, and delay
    """

    # initialize output dictionary
    inact_effects = {
                    'results'    : list(),                     # analysis of logistic regression
                    'area_list'  : params['inact_locations'],  # inactivation location labels
                    'regr_params': regr_params,                # see above
                    'data_lbl'   : list(),                     # labels for each epoch - area pair
                    }

    # organize input
    if combine_epochs:
        epoch_list = [['cueQuart1', 'cueQuart2', 'cueHalf1'],
                      ['cueQuart3', 'cueHalf2'], ['mem']]
    else:
        epoch_list = params['inact_epochs']

    num_areas  = len(inact_effects['area_list'])
    num_epochs = len(epoch_list)
    num_cond   = num_areas * num_epochs
    inact_effects['epoch_list'] = epoch_list

    # loop over areas and epochs
    for iArea in range(num_areas):
        print('area {}/{}...'.format(iArea+1, num_areas))
        for iEpoch in range(num_epochs):

            start_time = time.time()
            print('\tepoch {}/{}...'.format(iEpoch+1, num_epochs))
            print('\t\torganizing data...')

            # trim to single area lg
            thisarea  = inact_effects['area_list'][iArea]
            thisepoch = epoch_list[iEpoch]
            thislg    = lg_singleCondition(lg, thisarea, thisepoch)

            # compile data list
            if isinstance(thisepoch, list):
                cm_range = [utils.yRangeFromEpochLabel(thisepoch[iE]) for iE in range(len(thisepoch))]
                if thisepoch.count('cueHalf1'):
                    epoch_lbl = 'cueStart'
                elif thisepoch.count('cueHalf2'):
                    epoch_lbl = 'cueEnd'
                else:
                    epoch_lbl = 'delay'
            else:
                cm_range  = utils.yRangeFromEpochLabel(thisepoch)
                epoch_lbl = thisepoch

            print('\t\tfitting model',end='')
            inact_effects['results'].append(logRegr_time_split_mixed(thislg,cm_range,regr_params=regr_params))
            inact_effects['data_lbl'].append([thisarea, epoch_lbl])

            end_time = time.time()
            print('\t\tdone after {: 1.2g} min'.format((end_time-start_time)/60))

            del thislg

    return inact_effects

# ==============================================================================
# use fitLmer method to perform mixed-effects logistic regression with cross-validation with separate epochs
def logRegr_time_split_mixed(sublg, lsr_epoch_cm, regr_params):

    """
    logRegr = logRegr_time_spli_mixed(sublg,lsr_epoch_cm,regr_params)
    mixed-effects logistic regression in time for single area, separately for epochs (or sets thereof)
    sublg is log containing data from single area
    lsr_epoch_cm is list with cm ranges of all inactivation epochs in the log
    regre_params is dictionary with analysis parameters
    retruns logRegr, dictionary with model object, fitted cofficients, predictor labels,
    goodness of fit, random effects
    """

    # epoch lengths in cm
    lCue = 200
    lMem = 100

    # just convenience
    choice        = sublg['choice']
    cueOnset_R    = sublg['cueOnset_R']
    cueOnset_L    = sublg['cueOnset_L']
    trial_time    = sublg['time']
    pos           = sublg['pos']
    lsr_epoch_vec = sublg['laserEpoch']
    mouseID       = sublg['mouseID']
    laserON       = sublg['laserON']
    maxt          = regr_params['max_t']
    bins_per_sec  = regr_params['bins_per_sec']

    # calculate evidence binning and initialize variables
    ntrials    = np.size(choice)
    numbins    = []
    bins       = []
    bincenters = []
    for iBin in range(len(maxt)):
        numbins.append(int(maxt[iBin]*bins_per_sec[iBin]))
        bins.append(np.linspace(0, maxt[iBin], int(numbins[iBin]+1)))
        bincenters.append(None)

    # predictor matrix for logistic regression, trials x evidence bins

    # decide which regressions to run
    if isinstance(lsr_epoch_cm[0], list):
        onset_list  = [lsr_epoch_cm[iE][0] for iE in range(len(lsr_epoch_cm))]
        offset_list = [lsr_epoch_cm[iE][1] for iE in range(len(lsr_epoch_cm))]
        run_pre     = np.max(onset_list) >= lCue/2
        run_during  = np.max(onset_list) < lCue
        run_post    = np.max(offset_list) <= lCue/2
    else:
        run_pre     = lsr_epoch_cm[0] > 0
        run_during  = lsr_epoch_cm[0] < lCue
        run_post    = lsr_epoch_cm[1] < lCue

    # run regression for cues before start of laser onset
    epochID    = np.zeros(ntrials)  # epoch
    if run_pre:
        RminusLmat_pre = np.zeros((ntrials, numbins[0]))  # delta_towers
        for iTrial in range(ntrials):
            # define inactivation epoch
            if ~laserON[iTrial]:
                # figure out closest laser epoch withn +/- 100 trials, use that as dummy epoch
                idx      = np.arange(np.max([iTrial-500,0]),np.min([iTrial+500,ntrials]))
                idxc     = np.abs(idx-iTrial+1)
                islsr    = laserON[idx]
                closeidx = np.where(np.logical_and(idxc==np.min(idxc[islsr]),islsr))[0][0]
                this_ep  = utils.yRangeFromEpochLabel(lsr_epoch_vec[idx[closeidx]])
            else:
                this_ep  = utils.yRangeFromEpochLabel(lsr_epoch_vec[iTrial])

            if isinstance(lsr_epoch_cm[0], list):
                epochID[iTrial] = lsr_epoch_cm.index(this_ep)

            epoch_onset     = trial_time[iTrial][np.where(pos[iTrial][:, 1] < this_ep[0])[0][-1]]
            this_R          = cueOnset_R[iTrial]
            this_R          = epoch_onset - this_R[this_R < epoch_onset]
            this_L          = cueOnset_L[iTrial]
            this_L          = epoch_onset - this_L[this_L < epoch_onset]
            rhist, binedges = np.histogram(this_R, bins[0])
            lhist, _        = np.histogram(this_L, bins[0])
            RminusLmat_pre[iTrial, :] = rhist - lhist
            if bincenters[0] is None:
                bincenters[0] = binedges[:-1] + (binedges[1]-binedges[0])/2

    else:
        RminusLmat_pre = None
        bincenters[0]  = np.nan * np.ones(numbins[0])

    # run regression for cues during laser
    if run_during:
        RminusLmat_during = np.zeros((ntrials, numbins[1]))
        for iTrial in range(ntrials):
            # define inactivation epoch
            if ~laserON[iTrial]:
                # figure out closest laser epoch withn +/- 100 trials, use that as dummy epoch
                idx      = np.arange(np.max([iTrial-500,0]),np.min([iTrial+500,ntrials]))
                idxc     = np.abs(idx-iTrial+1)
                islsr    = laserON[idx]
                closeidx = np.where(np.logical_and(idxc==np.min(idxc[islsr]),islsr))[0][0]
                this_ep  = utils.yRangeFromEpochLabel(lsr_epoch_vec[idx[closeidx]])
            else:
                this_ep  = utils.yRangeFromEpochLabel(lsr_epoch_vec[iTrial])

            if epochID[iTrial] == 0:
                if isinstance(lsr_epoch_cm[0], list):
                    epochID[iTrial] = lsr_epoch_cm.index(this_ep)

            epoch_offset    = trial_time[iTrial][np.where(pos[iTrial][:, 1] < this_ep[1])[0][-1]]
            this_R          = cueOnset_R[iTrial]
            this_R          = epoch_offset - this_R[this_R < epoch_offset]
            this_L          = cueOnset_L[iTrial]
            this_L          = epoch_offset - this_L[this_L < epoch_offset]
            rhist, binedges = np.histogram(this_R, bins[1])
            lhist, _        = np.histogram(this_L, bins[1])
            RminusLmat_during[iTrial, :] = rhist - lhist
            if bincenters[1] is None:
                bincenters[1] = binedges[:-1] + (binedges[1]-binedges[0])/2

    else:
        RminusLmat_during = None
        bincenters[1]     = np.nan * np.ones(numbins[1])

    # run regression for cues post laser
    if run_post:
        RminusLmat_post = np.zeros((ntrials, numbins[2]))
        for iTrial in range(ntrials):
            # define inactivation epoch
            if ~laserON[iTrial]:
                # figure out closest laser epoch withn +/- 100 trials, use that as dummy epoch
                idx      = np.arange(np.max([iTrial-500,0]),np.min([iTrial+500,ntrials]))
                idxc     = np.abs(idx-iTrial+1)
                islsr    = laserON[idx]
                closeidx = np.where(np.logical_and(idxc==np.min(idxc[islsr]),islsr))[0][0]
                this_ep  = utils.yRangeFromEpochLabel(lsr_epoch_vec[idx[closeidx]])
            else:
                this_ep  = utils.yRangeFromEpochLabel(lsr_epoch_vec[iTrial])

            if epochID[iTrial] == 0:
                if isinstance(lsr_epoch_cm[0], list):
                    epochID[iTrial] = lsr_epoch_cm.index(this_ep)

            epoch_offset = trial_time[iTrial][np.where(pos[iTrial][:, 1] > this_ep[1])[0][0]]
            this_R = cueOnset_R[iTrial]
            this_R = epoch_offset - this_R[this_R > epoch_offset]
            this_L = cueOnset_L[iTrial]
            this_L = epoch_offset - this_L[this_L > epoch_offset]
            rhist, binedges = np.histogram(-this_R, bins[2])
            lhist, _        = np.histogram(-this_L, bins[2])
            RminusLmat_post[iTrial, :] = rhist - lhist
            if bincenters[2] is None:
                bincenters[2] = binedges[:-1] + (binedges[1]-binedges[0])/2

    else:
        RminusLmat_post = None
        bincenters[2]   = np.nan * np.ones(numbins[2])


    # combine predictor matrix and remove trials with no towers within evidence bins
    if run_pre & run_during:
        RminusLmat = np.hstack((RminusLmat_pre,RminusLmat_during))
    elif run_pre & ~run_during:
        RminusLmat = RminusLmat_pre
    else:
        RminusLmat = RminusLmat_during

    if run_post:
        RminusLmat = np.hstack((RminusLmat,RminusLmat_post))

    bad_trials = np.where(np.sum(RminusLmat==0,axis=1)==np.size(RminusLmat,axis=1))
    RminusLmat = np.delete(RminusLmat,bad_trials,axis=0)
    choice     = np.delete(choice,bad_trials)
    laserON    = np.delete(laserON,bad_trials)
    epochID    = np.delete(epochID,bad_trials)
    mouseID    = np.delete(mouseID,bad_trials)

    if regr_params['method'] == 'Bayes':
        if regr_params['nfold_xval']>1:
            print('Warning: Cross validation not yet implemented for Binomial Bayes GLM')
        model, random_effects, formula, choice_pred, coeffs =               \
            fitBinomialBayes(choice, laserON, RminusLmat, mouseID, epochID, \
                             addLsrOffset=regr_params['addLsrOffset'], zscore=regr_params['zscore'])

    elif regr_params['method'] == 'Lmer_explicit':
        model, random_effects, formula, choice_pred, coeffs =          \
            fitLmer(choice, laserON, RminusLmat, mouseID, epochID,     \
                             addLsrOffset=regr_params['addLsrOffset'], \
                             zscore=regr_params['zscore'],nfold_xval=regr_params['nfold_xval'])

    else:
        print('Warning: fitting method not implemented')
        return

    # which evidence coefficient belongs to pre, during, post
    coeffs_by_epoch = []
    ct              = 0
    for iEpoch in range(len(bincenters)):
        coeffs_by_epoch.append([])
        for iBin in range(np.size(bincenters[iEpoch])):
            coeffs_by_epoch[iEpoch].append('b{}'.format(ct))
            ct = ct + 1

    # organize output
    logRegr = {
              'model_obj'          : model,           # full model object
              'formula'            : formula,         # model formula
              'coeff'              : coeffs,          # fitted coefficients, overall and separated by condition
              'evidence_vals'      : bincenters,      # center of evidence bins
              'evidence_vals_lbls' : coeffs_by_epoch, # cofficient labels corresponding to diferent evidence bins
              'random_effects'     : random_effects,  # dataframe with random effects info
              'choice_pred'        : choice_pred,     # proportion / correlation of choices correctly predicted by model (thresholded)
              }

    return logRegr

# ==============================================================================
# use fitLmer method to perform mixed-effects logistic regression with cross-validation
def fitLmer(choice, laserON, RminusLmat, mouseID, epochID=None, addLsrOffset=True, zscore=True, nfold_xval=1):

    """
    model, random_effects, formula, choice_pred, coeffs =
            fitLmer(choice, laserON, RminusLmat, mouseID, epochID=None, addLsrOffset=True, zscore=True, nfold_xval=1)
    performs mixed-effects logistic regression, without cross-validation for now
    choice, laserON, mouseID, epochID are vectors from the flattened log
    RminusLmat is the delta predictor matrix by time bin
    addLsrOffset boolean flag to add explicit laser bias term
    zscore boolean flag to z-score RminusLmat
    nfold_xval is number of folds for cross-validation. 1 will use full dataset for fitting
    Returns model object, random effects data frame, model formula,
    choice_pred dictionary with accuracy of choice predictions,
    coeffs dictionary with key fitted coefficients
    """

    # zscore evidence matrix
    if zscore:
        RminusLmat = sp.stats.zscore(RminusLmat)

    # if there is only one epoch do not include as random effect
    if epochID is not None:
        if np.size(np.unique(epochID)) == 1:
            epochID = None

    # initialize vars
    model       = [None] * nfold_xval
    choice_pred = {'acc': np.ones(nfold_xval)*np.nan, 'corr': np.ones(nfold_xval)*np.nan}

    # set up x-val if necessary
    train_idx = [np.zeros(1)] * nfold_xval
    test_idx  = [np.zeros(1)] * nfold_xval

    if nfold_xval == 1:
        ntrials      = np.size(choice)
        train_idx[0] = np.arange(ntrials)
        test_idx[0]  = np.arange(ntrials)
    else:
        # xval will be done per mouse to ensure everyone is on every data split
        mice     = list(np.unique(mouseID))
        for mouse in mice:
            mouseidx = np.where(mouseID==mouse)[0]
            for iFold in range(nfold_xval):
                this_shuff       = train_test_split(mouseidx,test_size=1/nfold_xval)
                train_idx[iFold] = np.concatenate((train_idx[iFold],this_shuff[0]))
                test_idx[iFold]  = np.concatenate((test_idx[iFold],this_shuff[1]))
        for iFold in range(nfold_xval):
            train_idx[iFold] = train_idx[iFold][1:].astype(int)
            test_idx[iFold]  = test_idx[iFold][1:].astype(int)

    # cross-val runs
    for iVal in range(nfold_xval):
        # compile data for future dataframne conversion
        data      = {
                     'choice': choice[train_idx[iVal]],
                     'mouse': mouseID[train_idx[iVal]],
                     'lsrIntercept': laserON[train_idx[iVal]].astype(int)
                     }
        data_test = {'choice': choice[test_idx[iVal]],
                     'mouse': mouseID[test_idx[iVal]],
                     'lsrIntercept': laserON[test_idx[iVal]].astype(int)
                     }

        if epochID is not None:
            data['epoch']      = epochID[train_idx[iVal]].astype(int)
            data_test['epoch'] = epochID[test_idx[iVal]].astype(int)

        # write down formula and complete data
        formula = 'choice ~'  # this one is for main effects

        if addLsrOffset:
            form_me = '(1+lsrIntercept|mouse)' # this one is for random effects
        else:
            form_me = '' # this one is for random effects

        # evidence
        num_bins = np.size(RminusLmat, axis=1)
        for iBin in range(num_bins):
            varname       = "b{}".format(iBin)
            data[varname] = RminusLmat[train_idx[iVal], iBin] * (~laserON[train_idx[iVal]]).astype(float)
            formula      += " {} +".format(varname)
            form_me      += " + (0+{}|mouse)".format(varname)
            data_test[varname] = RminusLmat[test_idx[iVal], iBin] * (~laserON[test_idx[iVal]]).astype(float)

        # evidence x laser interaction
        for iBin in range(num_bins):
            varname       = "b{}_lsr".format(iBin)
            data[varname] = RminusLmat[train_idx[iVal], iBin] * laserON[train_idx[iVal]].astype(float)
            formula += " {} +".format(varname)
            form_me += " + (0+{}|mouse)".format(varname)
            data_test[varname] = RminusLmat[test_idx[iVal], iBin] * laserON[test_idx[iVal]].astype(float)

        # laser offset
        if addLsrOffset:
            varname  = 'lsrIntercept'
            formula += " {} +".format(varname)
            if epochID is not None:
                form_me += " + (0+{}|epoch)".format(varname)

        # clean up and fit
        formula      = formula[:-2]
        data         = pd.DataFrame(data)
        data_test    = pd.DataFrame(data_test)
        full_formula = "{} + {}".format(formula, form_me)

        # fit
        print('.',end='')
        model[iVal]  = Lmer(full_formula, data=data, family='binomial')
        model[iVal].fit(verbose=False,summarize=False)

        # predict choices and comparte
        ntrials_test = np.size(test_idx[iVal])
        choice_prob  = model[iVal].predict(data=data_test,skip_data_checks=True,verify_predictions=False)
        choice_hat   = np.zeros(ntrials_test)
        choice_hat[choice_prob > .5] = 1
        choice_pred['acc'][iVal]  = np.sum(choice_hat == choice[test_idx[iVal]])/ntrials_test
        choice_pred['corr'][iVal] = np.corrcoef(choice[test_idx[iVal]], choice_hat)[0, 1]

    # find best model
    best_idx   = np.where(choice_pred['acc'] == np.max(choice_pred['acc']))[0][0]
    best_model = model[best_idx]

    # random effects
    random_effects = dict()
    if epochID is None:
        random_effects['mice']  = best_model.ranef
        random_effects['epoch'] = None
    else:
        random_effects['mice']  = best_model.ranef[0]
        random_effects['epoch'] = best_model.ranef[1]

    # collect coefficients
    coeffs             = dict()
    coeffs['names']    = list(best_model.coefs.index)
    coeffs['mean']     = best_model.coefs['Estimate'].to_numpy()
    coeffs['sem']      = best_model.coefs['SE'].to_numpy()
    coeffs['pvals']    = best_model.coefs['P-val'].to_numpy()

    # compare lsr and control coefficients
    num_mice                  = np.size(np.unique(mouseID))
    coeffs['evid_names']      = [None] * num_bins
    coeffs['evid_ctrl']       = np.zeros(num_bins)
    coeffs['evid_lsr']        = np.zeros(num_bins)
    coeffs['evid_diff']       = np.zeros(num_bins)
    coeffs['evid_ctrl_pvals'] = np.zeros(num_bins)
    coeffs['evid_lsr_pvals']  = np.zeros(num_bins)
    coeffs['evid_ctrl_sem']   = np.zeros(num_bins)
    coeffs['evid_lsr_sem']    = np.zeros(num_bins)
    coeffs['evid_diff_sem']   = np.zeros(num_bins)
    coeffs['evid_diff_pvals'] = np.zeros(num_bins)
    coeffs['evid_ctrl_re']    = np.zeros((num_mice,num_bins))
    coeffs['evid_lsr_re']     = np.zeros((num_mice,num_bins))
    coeffs['evid_diff_re']    = np.zeros((num_mice,num_bins))
    coeffs['evid_diff_norm']       = np.zeros(num_bins)
    coeffs['evid_diff_norm_sem']   = np.zeros(num_bins)
    coeffs['evid_diff_norm_re']    = np.zeros((num_mice,num_bins))
    for iBin in range(num_bins):
        var_idx        = coeffs['names'].index('b{}'.format(iBin))
        var_idx_lsr    = coeffs['names'].index('b{}_lsr'.format(iBin))
        coeff_ctrl     = coeffs['mean'][var_idx]
        coeff_ctrl_sem = coeffs['sem'][var_idx]
        coeff_lsr      = coeffs['mean'][var_idx_lsr]
        coeff_lsr_sem  = coeffs['sem'][var_idx_lsr]
        z = abs((coeff_ctrl-coeff_lsr)/np.sqrt((coeff_ctrl*coeff_ctrl_sem)**2+(coeff_lsr*coeff_lsr_sem)**2))

        coeffs['evid_names'][iBin]      = 'b{}'.format(iBin)
        coeffs['evid_ctrl'][iBin]       = coeff_ctrl
        coeffs['evid_lsr'][iBin]        = coeff_lsr
        coeffs['evid_ctrl_pvals'][iBin] = coeffs['pvals'][var_idx]
        coeffs['evid_lsr_pvals'][iBin]  = coeffs['pvals'][var_idx_lsr]
        coeffs['evid_diff'][iBin]       = coeff_lsr - coeff_ctrl
        coeffs['evid_diff_norm'][iBin]  = (coeff_lsr - coeff_ctrl)/coeff_ctrl
        coeffs['evid_ctrl_sem'][iBin]   = coeff_ctrl_sem
        coeffs['evid_lsr_sem'][iBin]    = coeff_lsr_sem
        coeffs['evid_diff_sem'][iBin]   = np.sqrt((coeff_ctrl_sem**2+coeff_lsr_sem**2)/2)
        coeffs['evid_diff_norm_sem'][iBin] = coeffs['evid_diff_sem'][iBin]/coeff_ctrl
        coeffs['evid_diff_pvals'][iBin] = sp.stats.norm.sf(z)
        coeffs['evid_ctrl_re'][:,iBin]  = coeff_ctrl + random_effects['mice'].loc[:,'b{}'.format(iBin)].to_numpy()
        coeffs['evid_lsr_re'][:,iBin]   = coeff_lsr + random_effects['mice'].loc[:,'b{}_lsr'.format(iBin)].to_numpy()
        coeffs['evid_diff_re'][:,iBin]  = coeffs['evid_lsr_re'][:,iBin] - coeffs['evid_ctrl_re'][:,iBin]
        coeffs['evid_diff_norm_re'][:,iBin]  = (coeffs['evid_lsr_re'][:,iBin] - coeffs['evid_ctrl_re'][:,iBin])/coeffs['evid_ctrl_re'][:,iBin]

    # correct for multiple comps copy to norm for plotting convenience
    coeffs['evid_diff_isSig'], _   = utils.FDR(coeffs['evid_diff_pvals'])
    coeffs['evid_diff_norm_pvals'] = coeffs['evid_diff_pvals']
    coeffs['evid_diff_norm_isSig'] = coeffs['evid_diff_isSig']

    return best_model, random_effects, full_formula, choice_pred, coeffs

# ==============================================================================
# use BinomialBayesGLM method to perform mixed-effects logistic regression, without cross-validation for now
def fitBinomialBayes(choice, laserON, RminusLmat, mouseID, epochID=None, addLsrOffset=False, zscore=True):

    """
    model, random_effects, formula, choice_pred, coeffs =
            fitBinomialBayes(choice, laserON, RminusLmat, mouseID, epochID=None, addLsrOffset=False, zscore=True)
    performs mixed-effects logistic regression, without cross-validation for now
    choice, laserON, mouseID, epochID are vectors from the flattened log
    RminusLmat is the delta predictor matrix by time bin
    addLsrOffset boolean flag to add explicit laser bias term
    zscore boolean flag to z-score RminusLmat
    Returns model object, random effects dictionary, model formula,
    choice_pred dictionary with accuracy of choice predictions,
    coeffs dictionary with key fitted coefficients
    """

    # zscore evidence matrix
    if zscore:
        RminusLmat = sp.stats.zscore(RminusLmat)

    # if there is only one epoch do not include as random effect
    if epochID is not None:
        if np.size(np.unique(epochID)) == 1:
            epochID = None

    # convert to dataFrame and write down formula
    if epochID is None:
        data    = {'choice': choice, 'mouse': mouseID, 'laser': laserON.astype(float)}
    else:
        data    = {'choice': choice, 'mouse': mouseID, 'laser': laserON.astype(float),'epoch': epochID}

    formula = 'choice ~'  # this one is for main effects
    form_me = '0'  # this one is for random effects

    # evidence
    num_bins = np.size(RminusLmat, axis=1)
    for iBin in range(num_bins):
        varname       = "b{}".format(iBin)
        data[varname] = RminusLmat[:, iBin]
        formula      += " {} +".format(varname)
        form_me      += " + C(mouse)*{}".format(varname)

    # evidence x laser interaction
    for iBin in range(num_bins):
        varname  = "b{}*laser".format(iBin)
        formula += " {} +".format(varname)
        form_me += " + C(mouse)*{}".format(varname)

    # laser offset
    if addLsrOffset:
        varname  = 'laser'
        formula += " {} +".format(varname)

    # clean up and include slopes in random effects
    formula    = formula[:-2]
    data       = pd.DataFrame(data)

    if epochID is None:
        re_formula = {"a": '0 + C(mouse)', "b": form_me}
    else:
        re_formula = {"a": '0 + C(mouse) + C(epoch)', "b": form_me}

    # run regression and collect results
    model = BinomialBayesMixedGLM.from_formula(formula, re_formula, data).fit_vb()

    rd                          = model.random_effects()
    random_effects              = dict()
    random_effects['mean']      = rd.iloc[:, 0].to_numpy()
    random_effects['std']       = rd.iloc[:, 1].to_numpy()
    random_effects['var_names'] = list(rd.index)
    random_effects['formula']   = re_formula

    # generate and assess model predictions
    choice_pred = dict()
    ntrials     = np.size(choice)
    choice_prob = model.predict()
    choice_hat  = np.zeros(ntrials)
    choice_hat[choice_prob > .5] = 1
    choice_pred['acc']  = np.sum(choice_hat == choice)/ntrials
    choice_pred['corr'] = np.corrcoef(choice, choice_hat)[0, 1]

    coeffs             = dict()
    coeffs['mean']     = model.fe_mean
    coeffs['sem']      = model.fe_sd
    coeffs['names']    = model.model.exog_names

    return model, random_effects, formula, choice_pred, coeffs

# ==============================================================================
# mixed regression in time for shuffled data
def run_mixed_time_regression_shuff(lg,regr_type,regr_params=params['regr_params'],savePath=params['savePath'],overWrite=False,nshuff=None):

    """
    lr_time, lr_filebasename = run_mixed_time_regression_shuffle(lg,regr_type,regr_params=params['regr_params'],savePath=params['savePath'],overWrite=False,nshuff=None)
    takes flattened behavioral log structure and returns a dictionary with a
    list of mixed-effects regression results per area (lr_time).
    laser trials are random control trials with shuffled laser tags, but preserving
    area, epoch, mouse and session statistics
    lr_filebasename is filename without path or extension, e.g. for saving figs

    INPUT
    regr_type can be 'combinedEpochs' (all epochs go into single model as a random effect),
                     'singleEpochs' (each epoch in a separate model)
                     'epochSets' (hybrid: early cue, late cue, delay epochs get combined separately)
    regr_params is a dictionary of analysis parameters, optional input
    savePath is directory where data should be saved, Optional
    overWrite is boolean flag to overwrite existing data (default False). If False, existing data matching model parameters will be loaded
    nshuff is number of shuffles to perform (default nfold_xval)
    """

    # generate filename and load if it exists (unless overwrite is set to True)
    if params['excludeBadMice']:
        suffix      = 'goodMice'
    else:
        suffix      = 'allMice'

    lr_filebasename = 'logRegr_mixedEffects_time_bin_{:d}_{}_{}_{}_shuff'.format( \
                       regr_params['bins_per_sec'][0],regr_params['method'],regr_type,suffix)
    lr_fn           = '{}{}.hdf5'.format(savePath,lr_filebasename)

    if ~overWrite:
        if path.exists(lr_fn):
            print('found existing file, loading from disk')
            lr_time = fl.load(lr_fn)
            return lr_time, lr_filebasename

    # we will use n-fold as the number of shuffles
    these_params = copy.deepcopy(regr_params)
    if nshuff is None:
        nshuff   = these_params['nfold_xval']
    these_params['nfold_xval'] = 1

    # iterate over shuffles
    lr_time          = dict()
    lr_time['coeff'] = list()
    for iShuff in range(nshuff):
        print('SHUFFLE {}/{}\n\n'.format(iShuff+1,nshuff))
        # generate random laser tags
        lgshuff = shuffle_laser_tags(lg)

        # otherwise run desired regression
        if regr_type == 'combinedEpochs':
            try:
                this_lr = batch_mixed_logRegr_time_combined(lgshuff,these_params)
            except:
                this_lr = None
                print('Warning: error, skipping iteration')

        elif regr_type == 'singleEpochs':
            try:
                this_lr = batch_mixed_logRegr_time_split(lgshuff,these_params,combine_epochs=False)
            except:
                this_lr = None
                print('Warning: error, skipping iteration')

        elif regr_type == 'epochSets':
            try:
                this_lr = batch_mixed_logRegr_time_split(lgshuff,these_params,combine_epochs=True)
            except:
                this_lr = None
                print('Warning: error, skipping iteration')

        if this_lr is not None:
            # copy parameters, labels etc
            if iShuff == 0:
                vars = list(this_lr.keys())
                for iVar in vars:
                    if iVar != 'results':
                        lr_time[iVar] = copy.deepcopy(this_lr[iVar])

            # keep track of only coefficients to not blow up memory
            coeffs = list()
            for iCond in range(len(this_lr['results'])):
                coeffs.append(this_lr['results'][iCond]['coeff'])
            lr_time['coeff'].append(coeffs)

    # save
    fl.save(lr_fn,lr_time)

    return lr_time, lr_filebasename

# ==============================================================================
# randomly assign laser labels to control trials
def shuffle_laser_tags(lg):

    """
    lgshuff = shuffle_laser_tags(lg)
    removes all actual laser trials and assigns the exact area / epoch inactivation
    labels to a random set of control trials
    """

    # copy just control trials
    lgshuff  = dict()
    fields   = list(lg.keys())
    ctrlidx  = ~lg['laserON']
    for iF in range(len(fields)):
        thisvar = copy.deepcopy(lg[fields[iF]])
        if np.size(thisvar) == np.size(ctrlidx):
            lgshuff[fields[iF]] = thisvar[ctrlidx]
        else:
            lgshuff[fields[iF]] = thisvar

    # copy all laser trial info
    locs   = lg['loclbl'][lg['laserON']]
    epochs = lg['laserEpoch'][lg['laserON']]
    mice   = lg['mouseID'][lg['laserON']]
    sess   = lg['sessionID'][lg['laserON']]

    # assign that to random control trials
    numtrials    = np.size(lgshuff['laserON'])
    numlsrtrials = np.sum(lg['laserON'])
    randidx      = np.random.permutation(numtrials)[range(numlsrtrials)].astype(int)
    lgshuff['loclbl'][randidx]     = locs
    lgshuff['laserEpoch'][randidx] = epochs
    lgshuff['mouseID'][randidx]    = mice
    lgshuff['sessionID'][randidx]  = sess
    lgshuff['laserON'][randidx]    = True

    return lgshuff

# ==============================================================================
# get only relevant behavioral trials for area / epoch combination
def lg_singleCondition(lg,area,epoch):

    """
    sublg = lg_singleCondition(lg,area,epoch)
    lg is flattened log
    area is either a string with area name or a list thereof
    epohc is either a string with epoch label or a list thereof
    selects relevant laser and control trials for single condition
    """

    # laser trials
    numtrials = np.size(lg['choice'])
    if isinstance(area,list):
        isarea      = np.ones((numtrials)) * False
        for iarea in area:
            isarea  = np.logical_or(isarea,np.array([lg['loclbl'][iTrial]==iarea for iTrial in range(numtrials)]))
    else:
        isarea      = np.array([lg['loclbl'][iTrial]==area for iTrial in range(numtrials)])
    if isinstance(epoch,list):
        isepoch     = np.ones((numtrials)) * False
        for iepoch in epoch:
            isepoch = np.logical_or(isepoch,np.array([lg['laserEpoch'][iTrial]==iepoch for iTrial in range(numtrials)]))
    else:
        isepoch   = np.array([lg['laserEpoch'][iTrial]==epoch for iTrial in range(numtrials)])
    lsridx    = isarea & isepoch

    # pick only control trials from the same mice and sessions as laser trials
    mice     = np.unique(lg['mouseID'][lsridx])
    ctrlidx  = np.zeros(numtrials) > 1

    for iMouse in range(np.size(mice)):
        midx     = lg['mouseID'] == mice[iMouse]
        sessions = np.unique(lg['sessionID'][np.logical_and(lsridx,midx)])
        for iSession in range(np.size(sessions)):
            sessidx = lg['sessionID'] == sessions[iSession]
            thisidx = midx & sessidx & ~lg['laserON']
            ctrlidx = np.logical_or(ctrlidx , thisidx)

    idx      = lsridx | ctrlidx

    # leave just relevant trials in lg
    sublg  = dict()
    fields = list(lg.keys())
    for iF in range(len(fields)):
        thisvar = copy.deepcopy(lg[fields[iF]])
        if np.size(thisvar) == np.size(idx):
            sublg[fields[iF]] = thisvar[idx]
        else:
            sublg[fields[iF]] = thisvar

    return sublg

# ==============================================================================
# compute laser onset and offset triggered logistic regression coeffs and cluster areas
def laser_trig_logRegr(inact_effects):

    """
    trig_logRegr = laser_trig_logRegr(inact_effects)
    compute laser onset and offset triggered logistic regression coeffs and cluster areas
    inact_effects is dictionary generated by multiArea_boot_perf
    trig_logRegr is dictionary with analysis results
    """

    trig_logRegr               = dict()
    coeffs_diff                = inact_effects['logRegr']['coeffs_diff_mean']
    coeffs_ctrl                = inact_effects['logRegr']['coeffs_ctrl_mean']
    coeffs_lsr                 = inact_effects['logRegr']['coeffs_lsr_mean']
    numbins                    = inact_effects['logRegr']['evidence_vals'].size
    binsize                    = np.diff(inact_effects['logRegr']['evidence_vals'])[0]
    evidence_vals              = np.arange(0,200,200/numbins)
    evidence_vals_off          = np.arange(50,250,200/numbins)
    lbls                       = inact_effects['data_lbl']
    trig_logRegr['area_lbls']  = [lbls[iL][0] for iL in range(len(lbls))]
    trig_logRegr['epoch_lbls'] = [lbls[iL][1] for iL in range(len(lbls))]
    num_cond                   = len(trig_logRegr['area_lbls'])
    trig_nbins                 = evidence_vals.size

    # align by laser onset and offset
    trig_logRegr['onset_trig_coeffs']       = np.ones((num_cond,trig_nbins)) * np.nan
    trig_logRegr['offset_trig_coeffs']      = np.ones((num_cond,trig_nbins)) * np.nan
    for iCond in range(num_cond):
        yrange = utils.yRangeFromEpochLabel(trig_logRegr['epoch_lbls'][iCond])
        idx    = evidence_vals <= yrange[0]
        trig_logRegr['onset_trig_coeffs'][iCond,np.flip(idx)]      = coeffs_diff[iCond,idx]

        idx    = evidence_vals_off > yrange[1]
        trig_logRegr['offset_trig_coeffs'][iCond,range(np.sum(idx))] = coeffs_diff[iCond,idx]

    # remove edge datapoints (little data), set axes
    trig_logRegr['onset_trig_coeffs']      = trig_logRegr['onset_trig_coeffs'][:,1:]
    trig_logRegr['offset_trig_coeffs']     = trig_logRegr['offset_trig_coeffs'][:,:-2]
    trig_logRegr['onset_trig_xaxis']       = np.arange(-binsize*(numbins-2),binsize,binsize)
    trig_logRegr['offset_trig_xaxis']      = np.arange(binsize,binsize*(numbins-1),binsize)

    ## front vs posterior areas
    front_areas = ['mM2','aM2','M1']
    isFront     = np.zeros(num_cond) > 1
    for area in front_areas:
        isFront += np.array(trig_logRegr['area_lbls']) == area
    post_areas = ['V1','mV2','PPC','RSC']
    isPost     = np.zeros(num_cond) > 1
    for area in post_areas:
        isPost += np.array(trig_logRegr['area_lbls']) == area
    trig_logRegr['onset_trig_post_mean']  = np.nanmean(trig_logRegr['onset_trig_coeffs'][isPost,:],axis=0)
    num_not_nan                           = np.sum(~np.isnan(trig_logRegr['onset_trig_coeffs'][isPost,:]),axis=0)
    trig_logRegr['onset_trig_post_sem']   = np.nanstd(trig_logRegr['onset_trig_coeffs'][isPost,:],axis=0,ddof=1) / np.sqrt(num_not_nan-1)
    trig_logRegr['onset_trig_front_mean'] = np.nanmean(trig_logRegr['onset_trig_coeffs'][isFront,:],axis=0)
    num_not_nan                           = np.sum(~np.isnan(trig_logRegr['onset_trig_coeffs'][isFront,:]),axis=0)
    trig_logRegr['onset_trig_front_sem']  = np.nanstd(trig_logRegr['onset_trig_coeffs'][isFront,:],axis=0,ddof=1) / np.sqrt(num_not_nan-1)
    trig_logRegr['offset_trig_post_mean'] = np.nanmean(trig_logRegr['offset_trig_coeffs'][isPost,:],axis=0)
    num_not_nan                           = np.sum(~np.isnan(trig_logRegr['offset_trig_coeffs'][isPost,:]),axis=0)
    trig_logRegr['offset_trig_post_sem']  = np.nanstd(trig_logRegr['offset_trig_coeffs'][isPost,:],axis=0,ddof=1) / np.sqrt(num_not_nan-1)
    trig_logRegr['offset_trig_front_mean']= np.nanmean(trig_logRegr['offset_trig_coeffs'][isFront,:],axis=0)
    num_not_nan                           = np.sum(~np.isnan(trig_logRegr['offset_trig_coeffs'][isFront,:]),axis=0)
    trig_logRegr['offset_trig_front_sem'] = np.nanstd(trig_logRegr['offset_trig_coeffs'][isFront,:],axis=0,ddof=1) / np.sqrt(num_not_nan-1)

    ## compare simulatenous vs average front/post

    # 2-way ANOVA with RM (bins and simulatenous)
    condidx = np.argwhere(np.array(trig_logRegr['area_lbls']) == 'Front')
    y1      = trig_logRegr['onset_trig_coeffs'][isFront,:]
    y2      = np.squeeze(trig_logRegr['onset_trig_coeffs'][condidx,:])
    bins1   = np.zeros((np.sum(isFront),trig_nbins-1)) + np.array(range(trig_nbins-1))
    bins2   = np.zeros((np.size(condidx),trig_nbins-1)) + np.array(range(trig_nbins-1))
    conds2  = np.repeat(np.array(range(np.size(condidx))),trig_nbins-1).reshape((np.size(condidx),trig_nbins-1))
    conds1  = numpy.matlib.repmat(conds2,len(front_areas),1)
    y       = np.concatenate((y1.flatten(),y2.flatten()))
    x       = np.concatenate((bins1.flatten(),bins2.flatten()))
    c       = np.concatenate((conds1.flatten(),conds2.flatten()))
    g       = np.concatenate((np.zeros(np.size(y1)),np.ones(np.size(y2))))
    idx     = np.logical_and(~np.isnan(y),c>0)
    table   = pd.DataFrame({'bin'   : x[idx],
                            'cond'  : c[idx],
                            'group' : g[idx],
                            'coeff' : y[idx]})

    trig_logRegr['onset_trig_front_singleVSsimult_anova'] = pg.rm_anova(dv='coeff', within=['bin','group'],subject='cond',data=table)
    trig_logRegr['onset_trig_front_singleVSsimult_anova_pvals'] = \
        trig_logRegr['onset_trig_front_singleVSsimult_anova']['p-unc'].to_numpy()

    condidx = np.argwhere(np.array(trig_logRegr['area_lbls']) == 'Post')
    y1      = trig_logRegr['onset_trig_coeffs'][isPost,:]
    y2      = np.squeeze(trig_logRegr['onset_trig_coeffs'][condidx,:])
    bins1   = np.zeros((np.sum(isPost),trig_nbins-1)) + np.array(range(trig_nbins-1))
    bins2   = np.zeros((np.size(condidx),trig_nbins-1)) + np.array(range(trig_nbins-1))
    conds2  = np.repeat(np.array(range(np.size(condidx))),trig_nbins-1).reshape((np.size(condidx),trig_nbins-1))
    conds1  = numpy.matlib.repmat(conds2,len(post_areas),1)
    y       = np.concatenate((y1.flatten(),y2.flatten()))
    x       = np.concatenate((bins1.flatten(),bins2.flatten()))
    c       = np.concatenate((conds1.flatten(),conds2.flatten()))
    g       = np.concatenate((np.zeros(np.size(y1)),np.ones(np.size(y2))))
    idx     = np.logical_and(~np.isnan(y),c>0)
    table   = pd.DataFrame({'bin'   : x[idx],
                            'cond'  : c[idx],
                            'group' : g[idx],
                            'coeff' : y[idx]})

    trig_logRegr['onset_trig_post_singleVSsimult_anova'] = pg.rm_anova(dv='coeff', within=['bin','group'],subject='cond',data=table)
    trig_logRegr['onset_trig_post_singleVSsimult_anova_pvals'] = \
        trig_logRegr['onset_trig_post_singleVSsimult_anova']['p-unc'].to_numpy()

    # by bin
    condidx = np.argwhere(np.array(trig_logRegr['area_lbls']) == 'Post')
    ps      = np.zeros(trig_nbins-1)
    for iBin in range(trig_nbins-1):
        _, ps[iBin] = sp.stats.ttest_ind(trig_logRegr['onset_trig_coeffs'][condidx,iBin], \
                                         trig_logRegr['onset_trig_coeffs'][isPost,iBin],nan_policy='omit')
    isSig,_ = utils.FDR(ps)
    trig_logRegr['onset_trig_post_singleVSsimult_pvals'] = ps
    trig_logRegr['onset_trig_post_singleVSsimult_isSig'] = isSig

    condidx = np.argwhere(np.array(trig_logRegr['area_lbls']) == 'Front')
    ps      = np.zeros(trig_nbins-1)
    for iBin in range(trig_nbins-1):
        _, ps[iBin] = sp.stats.ttest_ind(trig_logRegr['onset_trig_coeffs'][condidx,iBin], \
                                         trig_logRegr['onset_trig_coeffs'][isFront,iBin],nan_policy='omit')
    isSig,_ = utils.FDR(ps)
    trig_logRegr['onset_trig_front_singleVSsimult_pvals'] = ps
    trig_logRegr['onset_trig_front_singleVSsimult_isSig'] = isSig

    # trig log by area, onset
    area_list = params['inact_locations']
    nareas    = len(area_list)
    trig_logRegr['onset_trig_area_mean']  = np.zeros((nareas,trig_nbins-1))
    trig_logRegr['onset_trig_area_sem']   = np.zeros((nareas,trig_nbins-1))
    trig_logRegr['onset_trig_area_pvals'] = np.zeros((nareas,trig_nbins-1))
    trig_logRegr['onset_trig_area_isSig'] = np.zeros((nareas,trig_nbins-1))
    trig_logRegr['onset_trig_area_anova_bins_pval'] = np.zeros(nareas)
    trig_logRegr['onset_trig_area_anova_bins']      = [None] * nareas
    for iArea, area in enumerate(area_list):
        condidx = np.argwhere(np.array(trig_logRegr['area_lbls']) == area)

        trig_logRegr['onset_trig_area_mean'][iArea,:]  = np.nanmean(trig_logRegr['onset_trig_coeffs'][condidx,:],axis=0)
        num_not_nan                                    = np.sum(~np.isnan(trig_logRegr['onset_trig_coeffs'][condidx,:],),axis=0)
        trig_logRegr['onset_trig_area_sem'][iArea,:]   = np.nanstd(trig_logRegr['onset_trig_coeffs'][condidx,:],axis=0,ddof=1) / np.sqrt(num_not_nan-1)

        # stats vs. zero
        ps = np.zeros(trig_nbins-1)
        ts = np.zeros(trig_nbins-1)
        for iBin in range(trig_nbins-1):
            ts[iBin], ps[iBin] = sp.stats.ttest_1samp(trig_logRegr['onset_trig_coeffs'][condidx,iBin],0,nan_policy='omit')
        ps    = ps/2 # test is one-sided
        fdr,_ = utils.FDR(ps)
        isSig = np.logical_and(fdr, ts < 0)
        trig_logRegr['onset_trig_area_pvals'][iArea,:] = ps
        trig_logRegr['onset_trig_area_isSig'][iArea,:] = isSig

        # stats across bins (one-way ANOVA with RM)
        vals  = trig_logRegr['onset_trig_coeffs'][condidx,:]
        bins  = np.zeros((np.size(condidx),trig_nbins-1)) + np.array(range(trig_nbins-1))
        conds = np.repeat(np.array(range(np.size(condidx))),trig_nbins-1).reshape((np.size(condidx),trig_nbins-1))
        y     = vals.flatten()
        x     = bins.flatten()
        g     = conds.flatten()
        idx   = np.logical_and(~np.isnan(y),g>0) # remove condition with n =1
        table = pd.DataFrame({'bin'   : x[idx],
                              'cond'  : g[idx],
                              'coeff' : y[idx]})

        trig_logRegr['onset_trig_area_anova_bins'][iArea] = pg.rm_anova(dv='coeff', within='bin',subject='cond',data=table)
        trig_logRegr['onset_trig_area_anova_bins_pval'][iArea] = \
            trig_logRegr['onset_trig_area_anova_bins'][iArea]['p-unc'].to_numpy()

    # trig log by area, offset
    trig_logRegr['offset_trig_area_mean']  = np.zeros((nareas,trig_nbins-2))
    trig_logRegr['offset_trig_area_sem']   = np.zeros((nareas,trig_nbins-2))
    trig_logRegr['offset_trig_area_pvals'] = np.zeros((nareas,trig_nbins-2))
    trig_logRegr['offset_trig_area_isSig'] = np.zeros((nareas,trig_nbins-2))
    trig_logRegr['offset_trig_area_anova_bins_pval'] = np.zeros(nareas)
    trig_logRegr['offset_trig_area_anova_bins']      = [None] * nareas
    for iArea, area in enumerate(area_list):
        condidx = np.argwhere(np.array(trig_logRegr['area_lbls']) == area)

        trig_logRegr['offset_trig_area_mean'][iArea,:]  = np.nanmean(trig_logRegr['offset_trig_coeffs'][condidx,:],axis=0)
        num_not_nan                                    = np.sum(~np.isnan(trig_logRegr['offset_trig_coeffs'][condidx,:],),axis=0)
        trig_logRegr['offset_trig_area_sem'][iArea,:]   = np.nanstd(trig_logRegr['offset_trig_coeffs'][condidx,:],axis=0,ddof=1) / np.sqrt(num_not_nan-1)

        # stats vs. zero
        ps = np.zeros(trig_nbins-2)
        ts = np.zeros(trig_nbins-2)
        for iBin in range(trig_nbins-2):
            ts[iBin], ps[iBin] = sp.stats.ttest_1samp(trig_logRegr['offset_trig_coeffs'][condidx,iBin],0,nan_policy='omit')
        ps    = ps/2 # test is one-sided
        fdr,_ = utils.FDR(ps)
        isSig = np.logical_and(fdr, ts < 0)
        trig_logRegr['offset_trig_area_pvals'][iArea,:] = ps
        trig_logRegr['offset_trig_area_isSig'][iArea,:] = isSig

        # stats across bins (one-way ANOVA with RM)
        vals  = trig_logRegr['offset_trig_coeffs'][condidx,:]
        bins  = np.zeros((np.size(condidx),trig_nbins-2)) + np.array(range(trig_nbins-2))
        conds = np.repeat(np.array(range(np.size(condidx))),trig_nbins-2).reshape((np.size(condidx),trig_nbins-2))
        y     = vals.flatten()
        x     = bins.flatten()
        g     = conds.flatten()
        idx   = np.logical_and(~np.isnan(y),g>0) # remove condition with n =1
        table = pd.DataFrame({'bin'   : x[idx],
                              'cond'  : g[idx],
                              'coeff' : y[idx]})

        trig_logRegr['offset_trig_area_anova_bins'][iArea] = pg.rm_anova(dv='coeff', within='bin',subject='cond',data=table)
        trig_logRegr['offset_trig_area_anova_bins_pval'][iArea] = \
            trig_logRegr['offset_trig_area_anova_bins'][iArea]['p-unc'].to_numpy()

    # cluster area averages of laser-triggered weight changes
    if params['clust_algo'] == 'Hierarchical':
        from sklearn.cluster import AgglomerativeClustering as clust_algo
    elif params['clust_algo'] == 'Spectral':
        from sklearn.cluster import SpectralClustering as clust_algo
    from sklearn import metrics

    # cluster Front and Post too?
    area_idx = np.zeros(nareas) == 0
    if params['clust_singleAreas_only']:
        area_idx[area_list.index('Front')] = False
        area_idx[area_list.index('Post')]  = False

    # choose what to cluster
    if params['clust_what'] == 'onset':
        clustmat = trig_logRegr['onset_trig_area_mean']
        fullmat  = trig_logRegr['onset_trig_coeffs']
    elif params['clust_what'] == 'offset':
        clustmat = trig_logRegr['offset_trig_area_mean']
        fullmat  = trig_logRegr['offset_trig_coeffs']
    elif params['clust_what'] == 'onset+offset':
        clustmat = np.concatenate((trig_logRegr['onset_trig_area_mean'],trig_logRegr['offset_trig_area_mean']),axis=1)

    # test different num clusters and pick best
    clusts   = [None] * (params['max_clust']-1)
    ch_index = [None] * (params['max_clust']-1)
    for iK in range(params['max_clust']-1):
        clusts[iK] = clust_algo(n_clusters=iK+2)
        clusts[iK].fit(clustmat[area_idx,:])
        ch_index[iK] = metrics.calinski_harabasz_score(clustmat[area_idx,:], clusts[iK].labels_)

    trig_logRegr['cluster_ch_index'] = ch_index
    trig_logRegr['cluster_nK']       = np.arange(2,params['max_clust']+1)
    best_clust                       = ch_index.index(max(ch_index))
    trig_logRegr['cluster_best_k']   = trig_logRegr['cluster_nK'][best_clust]
    trig_logRegr['cluster_object']   = clusts[best_clust]
    trig_logRegr['cluster_ids']      = trig_logRegr['cluster_object'].labels_

    # do stats per cluster, but considering conditions instead of areas for statistical power
    trig_logRegr['cluster_areas']  = np.array(area_list)[area_idx]
    trig_logRegr['cluster_mean']   = np.zeros((trig_logRegr['cluster_best_k'],trig_nbins-1))
    trig_logRegr['cluster_sem']    = np.zeros((trig_logRegr['cluster_best_k'],trig_nbins-1))
    trig_logRegr['cluster_pvals']  = np.zeros((trig_logRegr['cluster_best_k'],trig_nbins-1))
    trig_logRegr['cluster_isSig']  = np.zeros((trig_logRegr['cluster_best_k'],trig_nbins-1))
    for iClust in range(trig_logRegr['cluster_best_k']):
        idx = np.argwhere(trig_logRegr['cluster_ids'] == iClust)
        for iArea in range(np.size(idx)):
            area    = trig_logRegr['cluster_areas'][idx[iArea]]
            condidx = np.argwhere(np.array(trig_logRegr['area_lbls']) == area)
            if iArea == 0:
                vals = fullmat[condidx,:]
            else:
                vals = np.concatenate((vals,fullmat[condidx,:]),axis=0)

        vals = np.squeeze(vals)

        trig_logRegr['cluster_mean'][iClust,:] = np.nanmean(vals,axis=0)
        num_not_nan                            = np.sum(~np.isnan(vals),axis=0)
        trig_logRegr['cluster_sem'][iClust,:]  = np.nanstd(vals,axis=0,ddof=1) / np.sqrt(num_not_nan-1)

        # stats
        ps = np.zeros(trig_nbins-1)
        ts = np.zeros(trig_nbins-1)
        for iBin in range(trig_nbins-1):
            ts[iBin], ps[iBin] = sp.stats.ttest_1samp(vals[:,iBin],0,nan_policy='omit')
        ps    = ps/2 # test is one-sided
        fdr,_ = utils.FDR(ps)
        isSig = np.logical_and(fdr, ts < 0)
        trig_logRegr['cluster_pvals'][iClust,:] = ps
        trig_logRegr['cluster_isSig'][iClust,:] = isSig

    return trig_logRegr

# ==============================================================================
# plot results for area / epoch combinations
def plot_multiArea(inact_effects,plot_category='logRegr',plot_var='coeffs_diff',mouse_data=None):

    """
    figHandle = plot_multiArea(inact_effects,plot_category='logRegr',plot_var='coeffs_diff',mouse_data=None)
    plots inactivation effects for all inactivation conditions (one per panel)
    inact_effects is dictionary generated by multiArea_boot_perf
    plot_category is 'logRegr', 'psych' or 'percCorrect'
    plot_var is which variable to plot (diffs, ratios etc), does not apply to psych. Must match variable name exactly
    mouse_data (default None) is dictionary produced by behav_by_mouse() / multiArea_percCorrect_bymouse()
    if present, will overlay per mouse data on % correct plot
    """

    data      = inact_effects[plot_category]
    areas     = inact_effects['area_list']
    numareas  = len(areas)
    epochs    = inact_effects['epoch_list']
    numepochs = len(epochs)

    if plot_category == 'percCorrect' or plot_category == 'speed':
        fig   = plt.figure(figsize=[numepochs, numepochs*.6])
    else:
        fig   = plt.figure(figsize=[numepochs*1.3, numareas*1.1])


    if plot_category == 'logRegr':
        # plot logistic regression
        ct = 1
        for iArea in range(numareas):
            for iEpoch in range(numepochs):
                ax = fig.add_subplot(numareas,numepochs,ct)

                # plot shading indicating y pos range of laser inactivation
                yrange  = utils.yRangeFromEpochLabel(epochs[iEpoch])
                fillx   = [yrange[0], yrange[1], yrange[1], yrange[0]]
                filly   = [-.55, -.55, .55, .55]
                fillcl  = params['lsr_color']
                ax.fill(fillx,filly,fillcl,alpha=.25)

                # plot data
                lr_temp = dict()
                lr_temp['evidence_vals'] = data['evidence_vals']
                lr_temp['coeff']         = data['{}_mean'.format(plot_var)][ct-1,:]
                lr_temp['coeff_err']     = data['{}_std'.format(plot_var)][ct-1,:]
                ax                       = plot_logRegr(lr_temp, axisHandle=ax)
                ax.set_xticks(range(0,300,100))

                if ct != 4*numepochs + 1:
                    ax.set_ylabel(None)
                else:
                    ax.set_ylabel('Weight on decision (a.u.)')
                if ct != numareas*numepochs - 3:
                    ax.set_xlabel(None)
                else:
                    ax.set_xlabel('Cue y (cm)')

                # plot pvals
                x     = data['evidence_vals']
                y     = lr_temp['coeff'] + lr_temp['coeff_err']
                p     = data['{}_p'.format(plot_var)][ct-1,:]
                isSig = data['{}_isSig'.format(plot_var)][ct-1,:]
                where = 'below'
                ax    = utils.plotPvalCircles(ax, x, y-.16, p, isSig, where)

                # title
                ax.set_title('{}, {}'.format(inact_effects['data_lbl'][ct-1][0],inact_effects['data_lbl'][ct-1][1]),pad=0)

                # cosmetics
                ax.set_xlim([0, 225])
                ax.set_ylim([-.55, .35])
                utils.applyPlotDefaults(ax)
                ct = ct + 1


    elif plot_category == 'psych':
        # plot psychometrics
        ct = 1
        for iArea in range(numareas):
            for iEpoch in range(numepochs):
                ax = fig.add_subplot(numareas,numepochs,ct)

                # plot average ctrl and lsr data, with average fits
                psych_temp = dict()
                psych_temp['P_wentRight']    = data['P_wentRight_ctrl_mean'][ct-1,:]
                psych_temp['P_wentRight_CI'] = data['P_wentRight_ctrl_std'][ct-1,:]
                psych_temp['delta_towers']   = data['psych_bins'][:-1] + np.diff(data['psych_bins'])[0]/2
                psych_temp['fit_x']          = data['fit_x']
                # psych_temp['fit_y']          = data['fit_y_ctrl_mean'][ct-1,:]
                thisfit , _  = sp.optimize.curve_fit(psych_fit_fn, np.transpose(psych_temp['delta_towers']), \
                                                     np.transpose(psych_temp['P_wentRight']),maxfev=2000)
                psych_temp['fit_y']    = psych_fit_fn(psych_temp['fit_x'],*thisfit)

                ax = plot_psych(psych_temp, params['ctrl_color'], params['ctrl_shade'], ax)

                psych_temp = dict()
                psych_temp['P_wentRight']    = data['P_wentRight_lsr_mean'][ct-1,:]
                psych_temp['P_wentRight_CI'] = data['P_wentRight_lsr_std'][ct-1,:]
                psych_temp['delta_towers']   = data['psych_bins'][:-1] + np.diff(data['psych_bins'])[0]/2
                psych_temp['fit_x']          = data['fit_x']
                thisfit , _  = sp.optimize.curve_fit(psych_fit_fn, np.transpose(psych_temp['delta_towers']), \
                                                     np.transpose(psych_temp['P_wentRight']),maxfev=10000)
                psych_temp['fit_y']    = psych_fit_fn(psych_temp['fit_x'],*thisfit)

                ax = plot_psych(psych_temp, params['lsr_color'], params['lsr_shade'], ax)

                ax.set_xticks(np.arange(-10,20,10))
                ax.set_yticks(np.arange(0,1.25,.5))
                ax.set_yticklabels(np.arange(0,125,50))

                if ct != 4*numepochs + 1:
                    ax.set_ylabel(None)
                else:
                    ax.set_ylabel('Went right (%)')
                if ct != numareas*numepochs - 3:
                    ax.set_xlabel(None)

                # title
                ax.set_title('{}, {}'.format(inact_effects['data_lbl'][ct-1][0],inact_effects['data_lbl'][ct-1][1]),pad=0)
                # cosmetics
                utils.applyPlotDefaults(ax)
                ct = ct + 1

    elif plot_category == 'percCorrect':
        # plot % correct
        nr, nc = utils.subplot_org(numepochs,3)
        colors = utils.getAreaColors(areas) # default area colors

        for iEpoch in range(numepochs):
            ax = fig.add_subplot(nr,nc,iEpoch+1)

            p_y       = np.zeros(numareas)
            p         = np.zeros(numareas)
            isSig     = np.zeros(numareas)

            # bars
            for iArea in range(numareas):
                idx = iArea*numepochs + iEpoch

                # mean from bootstrapping or across mice if available
                if mouse_data is not None:
                    mdata        = mouse_data['percCorrect'][plot_var][idx]
                    nmice        = np.size(mdata)

                p[iArea]     = data['{}_p'.format(plot_var)][idx,:]
                isSig[iArea] = data['{}_p_isSig'.format(plot_var)][idx]
                thismean     = data['{}_mean'.format(plot_var)][idx,:]
                thisstd      = data['{}_std'.format(plot_var)][idx,:]

                # plot average ctrl and lsr data
                ax.bar(iArea,thismean,facecolor=colors[iArea],edgecolor=colors[iArea])
                ax.errorbar(iArea,thismean,thisstd,color=colors[iArea],linewidth=2)

                p_y[iArea]   = thismean - thisstd

                # plot mouse datapoints if applicable
                if mouse_data is not None:
                    jitter = np.random.uniform(size=(nmice))*0.1-0.05
                    ax.plot(iArea+jitter,mdata,marker='.',c=[.7, .7, .7, .5], ls='None')

            # plot pvals
            x     = np.arange(numareas)
            y     = p_y - .1
            where = 'below'
            ax    = utils.plotPvalCircles(ax, x, y, p, isSig, where, rotate=True, color=[0,0,0])

            # title and axes
            ax.set_title(epochs[iEpoch],pad=0)
            ax.set_ylabel('$\Delta$ P correct')
            ax.set_xticks(x)
            ax.set_xticklabels(areas, rotation='vertical')

            # cosmetics
            ax.set_xlim([-.75, numareas-.25])
            ax.set_ylim([-.5, .25])
            utils.applyPlotDefaults(ax)

    elif plot_category == 'speed':
        # plot % correct
        nr, nc = utils.subplot_org(numepochs,3)
        colors = utils.getAreaColors(areas) # default area colors

        for iEpoch in range(numepochs):
            ax = fig.add_subplot(nr,nc,iEpoch+1)

            p_y       = np.zeros(numareas)
            p         = np.zeros(numareas)
            isSig     = np.zeros(numareas)

            # bars
            for iArea in range(numareas):
                idx = iArea*numepochs + iEpoch

                # mean from bootstrapping or across mice if available
                if mouse_data is not None:
                    mdata        = mouse_data['speed'][plot_var][idx]
                    nmice        = np.size(mdata)
                    p[iArea]     = mouse_data['speed']['pval'][idx]
                    isSig[iArea] = mouse_data['speed']['isSig'][idx]
                    thismean     = mouse_data['speed']['{}_mean'.format(plot_var)][idx]
                    thisstd      = mouse_data['speed']['{}_std'.format(plot_var)][idx]
                    if plot_var == 'ratio':
                        thismean = thismean*100
                        thisstd  = thisstd*100
                else:
                    p[iArea]     = data['pval'][idx,:]
                    isSig[iArea] = data['isSig'][idx]
                    thismean     = data['{}_mean'.format(plot_var)][idx,:]
                    thisstd      = data['{}_std'.format(plot_var)][idx,:]

                # plot average ctrl and lsr data
                ax.bar(iArea,thismean,facecolor=colors[iArea],edgecolor=colors[iArea])
                ax.errorbar(iArea,thismean,thisstd,color=colors[iArea],linewidth=2)

                p_y[iArea]   = thismean + thisstd

                # plot mouse datapoints if applicable
                if mouse_data is not None:
                    jitter = np.random.uniform(size=(nmice))*0.1-0.05
                    ax.plot(iArea+jitter,mdata,marker='.',c=[.7, .7, .7, .5], ls='None')

            # plot pvals
            x     = np.arange(numareas)
            y     = p_y + 10
            where = 'above'
            ax    = utils.plotPvalCircles(ax, x, y, p, isSig, where, rotate=True, color=[0,0,0])

            # title and axes
            ax.set_title(epochs[iEpoch],pad=0)
            if plot_var == 'diff':
                ax.set_ylabel('$\Delta$ speed (cm/s)')
                ax.set_ylim([-20,20])
            elif plot_var == 'ratio':
                ax.set_ylabel('$\Delta$ speed (%)')
                ax.set_ylim([-50,50])
            else:
                ax.set_ylabel('Speed (cm/s)')
                ax.set_ylim([0, 100])
            ax.set_xticks(x)
            ax.set_xticklabels(areas, rotation='vertical')

            # cosmetics
            ax.set_xlim([-.75, numareas-.25])
            utils.applyPlotDefaults(ax)

    if plot_category == 'percCorrect' or plot_category == 'speed':
        fig.subplots_adjust(left=.1, bottom=.1, right=.95, top=.9, wspace=.5, hspace=.7)
    else:
        fig.subplots_adjust(left=.06, bottom=.05, right=.95, top=.95, wspace=.4, hspace=.7)

    # collect source data
    figdata = dict()
    if plot_category == 'percCorrect':
        figdata['inactivation_condition_labels'] = inact_effects['data_lbl']
        figdata['percCorrect_delta_bymouse']     = list()
        figdata['percCorrect_delta_mean']        = list()
        figdata['percCorrect_delta_sem']         = list()
        figdata['percCorrect_delta_pvals']       = list()
        figdata['percCorrect_delta_FDR_isSig']   = list()
        for iEpoch in range(numepochs):
            for iArea in range(numareas):
                idx = iArea*numepochs + iEpoch

                # mean from bootstrapping or across mice if available
                if mouse_data is not None:
                    mdata        = mouse_data['percCorrect'][plot_var][idx]
                else:
                    mdata        = None

                p            = data['{}_p'.format(plot_var)][idx,:]
                isSig        = data['{}_p_isSig'.format(plot_var)][idx]
                thismean     = data['{}_mean'.format(plot_var)][idx,:]
                thisstd      = data['{}_std'.format(plot_var)][idx,:]

                figdata['percCorrect_delta_bymouse'].append(mdata)
                figdata['percCorrect_delta_mean'].append(thismean)
                figdata['percCorrect_delta_sem'].append(thisstd)
                figdata['percCorrect_delta_pvals'].append(p)
                figdata['percCorrect_delta_FDR_isSig'].append(isSig)

    else:
        print('Warning: source data collection only implemented for % correct')

    return fig , figdata

# ==============================================================================
# plot control psychometrics and logistic regression
def plotCtrlBehav(mouse_data,psych,logRegr):

    fig      = plt.figure(figsize=[1.1,3])
    num_mice = len(mouse_data['psych'])
    ax       = plt.subplot(2,1,1)
    ax.plot([-15,15],[.5,.5],ls='--',c=[.8, .8, .8],linewidth=.25)
    ax.plot([0,0],[0,1],ls='--',c=[.8, .8, .8],linewidth=.25)
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
    ax.set_ylim([0, 1])
    utils.applyPlotDefaults(ax)

    ax = plt.subplot(2,1,2)
    ax.plot([0,200],[0,0],ls='--',c=[.8, .8, .8],linewidth=.25)
    for iMouse in range(num_mice):
        ax.plot(mouse_data['logRegr'][iMouse]['evidence_vals'],mouse_data['logRegr'][iMouse]['coeff'], \
                 color=[.7, .7, .7],linestyle='-',linewidth=.35)
    ax.errorbar(logRegr['evidence_vals'],logRegr['coeff'],logRegr['coeff_err'],linewidth=.75,color='k',marker='.',markersize=2)
    ax.set_ylabel('Weight on decision (a.u.)')
    ax.set_yticks(np.arange(-.25,.75,.25))
    ax.set_xlim([0, 200])
    ax.set_ylim([-.25, .5])
    ax.set_xlabel('Cue y (cm)')
    utils.applyPlotDefaults(ax)

    fig.subplots_adjust(left=.1, bottom=.1, right=.95, top=.9, wspace=.5, hspace=1)

    # collect source data
    figdata = dict()
    figdata['psychometric_data_xaxis']          = psych['delta_towers']
    figdata['psychometric_data_yaxis']          = psych['P_wentRight']
    figdata['psychometric_data_yaxis_confInt']  = np.transpose(psych['P_wentRight_CI'])
    figdata['psychometric_fit_xaxis']           = mouse_data['psych'][0]['fit_x']
    figdata['psychometric_fit_yaxis_bymouse']   = list()
    figdata['psychometric_fit_yaxis_aggregate'] = psych['fit_y']
    figdata['psychometric_xaxis_unit']          = '\Delta towers (#R - #L)'
    figdata['psychometric_yaxis_unit']          = 'Proportion went right'

    figdata['logisticRegression_xaxis']         = mouse_data['logRegr'][0]['evidence_vals']
    figdata['logisticRegression_yaxis']         = logRegr['coeff']
    figdata['logisticRegression_yaxis_STD']     = logRegr['coeff_err']
    figdata['logisticRegression_yaxis_bymouse'] = list()
    figdata['logisticRegression_xaxis_unit']    = 'Cue y (cm)'
    figdata['logisticRegression_yaxis_unit']    = 'Weight on decision (a.u.)'

    for iMouse in range(num_mice):
        figdata['psychometric_fit_yaxis_bymouse'].append(mouse_data['psych'][iMouse]['fit_y'])
        figdata['logisticRegression_yaxis_bymouse'].append(mouse_data['logRegr'][iMouse]['coeff'])

    return fig , figdata

# ===========================================================================
# Prediction accuracy of logistic regression in time
def plot_model_predictions(lr_time,plot_what='acc',ax=None):

    if ax is None:
        fig = plt.figure(figsize=(2,2))
        ax  = fig.gca()

    acc = lr_time['results'][0]['choice_pred'][plot_what]
    for iCond in range(1,len(lr_time['results'])):
        acc = np.concatenate((acc,lr_time['results'][iCond]['choice_pred'][plot_what]))

    ax.hist(acc,bins=np.arange(0.5,1.05,.025),color=[.5,.5,.5])
    ax.set_xticks(np.arange(.5,1.25,.25))
    ax.set_xlabel('Cross-val choice prediction accuracy')
    ax.set_ylabel('Num. observations\n(Conditions * Cross-val runs)')
    ax.set_xlim([.5, 1])
    ax = utils.applyPlotDefaults(ax)

    # figdata = {'model_accuracy': acc}

    return ax, acc

# ==============================================================================
# plot results for logistic regression in time
def plot_logRegr_time(lr_time,regr_type,plot_what='evid_diff_norm',doAreaCl=False,shuff=None,plot_sim=True,fig=None):

    """
    fig = plot_logRegr_time(lr_time,regr_type,plot_what='evid_diff_norm',doAreaCl=False)
    plots results of mixed effects logistic regression in time
    lr_time is dictionary output of batch_mixed_logRegr* functions
    regr_type 'combinedEpochs', 'singleEpochs' or 'epochSets' (see run_mixed_time_regression())
    plot_what what to plot on y axis, default 'evid_diff_norm'. can also be list eg ['evid_ctrl','evid_laser']
    doAreaCl True to plot each area with its default color
    returns figure handle fig
    """

    # overall time axis
    nan_array   = np.ones((1)) * np.nan
    axis_pre    = lr_time['results'][1]['evidence_vals'][0]
    nt_pre      = np.size(axis_pre)
    idx_pre     = range(0,nt_pre)
    axis_during = lr_time['results'][1]['evidence_vals'][1]
    nt_during   = np.size(axis_during)
    idx_during  = range(nt_pre,nt_pre+nt_during)
    axis_post   = lr_time['results'][1]['evidence_vals'][2]
    nt_post     = np.size(axis_post)
    idx_post    = range(nt_pre+nt_during,nt_pre+nt_during+nt_post)
    t_axis      = np.concatenate((-1 * np.flip(axis_pre), nan_array,- 1 * np.flip(axis_during), nan_array, axis_post), axis=0)

    t_axis_idx = np.arange(np.size(t_axis))
    t_axis_lbl = list()
    for lbl in list(t_axis):
        if np.isnan(lbl):
            t_axis_lbl.append('')
        else:
            t_axis_lbl.append(np.array2string(lbl))

    # color scheme
    cl      = utils.getAreaColors(lr_time['area_list'])
    if isinstance(plot_what,list):
        doAreaCl  = False
    else:
        plot_what = [plot_what]

    # shuffled model (coefficient distributions)
    if shuff is not None:
        shuff_avg = shuffle_avg(shuff,plot_what[0])
    else:
        shuff_avg = None

    # Figure size, panels (depends on regression types)
    if regr_type == 'combinedEpochs':
        if fig is None:
            fig = plt.figure(figsize=(5, 3.5))
        nr      = 3
        nc      = 3
        datalbl = lr_time['area_list']
        lr_time['data_lbl'] = lr_time['area_list']

    elif regr_type == 'singleEpochs':
        if fig is None:
            fig = plt.figure(figsize=(8, 5))
        nr      = 9
        nc      = 6
        datalbl = lr_time['data_lbl']

    elif regr_type == 'epochSets':
        if fig is None:
            fig = plt.figure(figsize=(5, 5))
        nr      = 9
        nc      = 3
        datalbl = lr_time['data_lbl']

    # set up source data
    figdata = dict()
    figdata['coefficient_type']               = plot_what
    figdata['inactivation_condition_labels']  = datalbl
    figdata['time_axis']                      = t_axis
    figdata['time_axis_unit']                 = 'seconds'
    figdata['coefficient_mean']               = list()
    figdata['coefficient_sem']                = list()
    figdata['coefficient_pval']               = list()
    figdata['coefficient_FDR_isSig']          = list()
    figdata['coefficient_shuffle_mean']       = list()
    figdata['coefficient_shuffle_std']        = list()
    figdata['coefficient_randomEffects_mice'] = list()

    # plot each condition
    plot_ct = 0
    for iCond in range(len(lr_time['results'])):

        # pick color
        if doAreaCl:
            if isinstance(datalbl[0],list):
                thisc     = cl[lr_time['area_list'].index(data_lbl[iCond][0])]
            else:
                thisc     = cl[iCond]
            thisc     = np.array(thisc)
            thiss     = copy.deepcopy(thisc)
            thiss[-1] = .6 # transparency
            thiss     = thiss*.8 # transparency
            thisc     = [thisc]
            thiss     = [thiss]
        else:
            # if list, assume it's ctrl / laser
            thisc = [[0,0,0,1],[.22,.52,1,1]]
            thiss = [[.7,.7,.7,.8],[.6,.75,1,.8]]

        # zero line
        if plot_sim:
            donotPlot = False
        else:
            if regr_type == 'combinedEpochs':
                donotPlot = lr_time['data_lbl'][iCond] =='Front' or lr_time['data_lbl'][iCond]=='Post'
            else:
                donotPlot = lr_time['data_lbl'][iCond][0]=='Front' or lr_time['data_lbl'][iCond][0]=='Post'
        if donotPlot==False:
            plot_ct = plot_ct + 1
            ax      = fig.add_subplot(nr, nc, plot_ct)
            ax.plot([t_axis_idx[0], t_axis_idx[-1]], [0, 0], ':', color=[0, 0, 0])
            fillx  = [t_axis_idx[4]-.5, t_axis_idx[4]+.5, t_axis_idx[4]+.5, t_axis_idx[4]-.5]
            filly  = [-3, -3, 3, 3]
            fillcl = params['lsr_color']
            ax.fill(fillx,filly,fillcl,alpha=.25)

        for iData in range(len(plot_what)):
            # concatenate vectors into a single time axis
            vec      = lr_time['results'][iCond]['coeff']['{}'.format(plot_what[iData])]
            vec_sem  = lr_time['results'][iCond]['coeff']['{}_sem'.format(plot_what[iData])]
            vec_re   = lr_time['results'][iCond]['coeff']['{}_re'.format(plot_what[iData])]
            vec_p    = lr_time['results'][iCond]['coeff']['{}_pvals'.format(plot_what[iData])]
            try:
                vec_sig  = lr_time['results'][iCond]['coeff']['{}_isSig'.format(plot_what[iData])]
            except:
                vec_sig  = np.ones(np.size(vec_p)) * False

            ev_vals  = lr_time['results'][iCond]['evidence_vals']
            nmice    = np.size(vec_re,axis=0)
            nan_vec  = np.ones((nmice,1)) * np.nan

            if np.isnan(ev_vals[0][0]):
                this_nt_pre  = 0
                mean_pre     = np.ones(nt_pre) * np.nan
                sem_pre      = np.ones(nt_pre) * np.nan
                p_pre        = np.ones(nt_pre) * np.nan
                sig_pre      = np.ones(nt_pre) * np.nan
                re_pre       = np.ones((nmice,nt_pre)) * np.nan
                if iData == 0 and shuff_avg is not None:
                    sm_pre   = np.ones(nt_pre) * np.nan
                    sd_pre   = np.ones(nt_pre) * np.nan
            else:
                this_nt_pre  = nt_pre
                mean_pre     = np.flip(vec[idx_pre])
                sem_pre      = np.flip(vec_sem[idx_pre])
                p_pre        = np.flip(vec_p[idx_pre])
                sig_pre      = np.flip(vec_sig[idx_pre])
                re_pre       = np.flip(vec_re[:,idx_pre])
                if iData == 0 and shuff_avg is not None:
                    sm_pre   = np.flip(shuff_avg[iCond]['mean'][idx_pre])
                    sd_pre   = np.flip(shuff_avg[iCond]['std'][idx_pre])

            if np.isnan(ev_vals[1][0]):
                this_nt_dur  = 0
                mean_dur     = np.ones(nt_during) * np.nan
                sem_dur      = np.ones(nt_during) * np.nan
                p_dur        = np.ones(nt_during) * np.nan
                sig_dur      = np.ones(nt_during) * np.nan
                re_dur       = np.ones((nmice,nt_during)) * np.nan
                if iData == 0 and shuff_avg is not None:
                    sm_dur   = np.ones(nt_during) * np.nan
                    sd_dur   = np.ones(nt_during) * np.nan
            else:
                this_nt_dur  = nt_during
                this_idx     = range(this_nt_pre,this_nt_pre+this_nt_dur)
                mean_dur     = np.flip(vec[this_idx])
                sem_dur      = np.flip(vec_sem[this_idx])
                p_dur        = np.flip(vec_p[this_idx])
                sig_dur      = np.flip(vec_sig[this_idx])
                re_dur       = np.flip(vec_re[:,this_idx])
                if iData == 0 and shuff_avg is not None:
                    sm_dur   = np.flip(shuff_avg[iCond]['mean'][this_idx])
                    sd_dur   = np.flip(shuff_avg[iCond]['std'][this_idx])

            if np.isnan(ev_vals[2][0]):
                this_nt_post = 0
                mean_post    = np.ones(nt_post) * np.nan
                sem_post     = np.ones(nt_post) * np.nan
                p_post       = np.ones(nt_post) * np.nan
                sig_post     = np.ones(nt_post) * np.nan
                re_post      = np.ones((nmice,nt_post)) * np.nan
                if iData == 0 and shuff_avg is not None:
                    sm_post   = np.ones(nt_post) * np.nan
                    sd_post   = np.ones(nt_post) * np.nan
            else:
                this_nt_post = nt_post
                this_idx     = range(this_nt_pre+this_nt_dur,this_nt_pre+this_nt_dur+this_nt_post)
                mean_post    = vec[this_idx]
                sem_post     = vec_sem[this_idx]
                p_post       = vec_p[this_idx]
                sig_post     = vec_sig[this_idx]
                re_post      = vec_re[:,this_idx]
                if iData == 0 and shuff_avg is not None:
                    sm_post  = np.flip(shuff_avg[iCond]['mean'][this_idx])
                    sd_post  = np.flip(shuff_avg[iCond]['std'][this_idx])

            thismean = np.concatenate((mean_pre, nan_array, mean_dur, nan_array, mean_post), axis=0)
            thissem  = np.concatenate((sem_pre, nan_array, sem_dur, nan_array, sem_post), axis=0)
            thispval = np.concatenate((p_pre, nan_array, p_dur, nan_array, p_post), axis=0)
            thissig  = np.concatenate((sig_pre, nan_array, sig_dur, nan_array, sig_post), axis=0)
            thisre   = np.hstack((re_pre, nan_vec, re_dur, nan_vec, re_post))

            # significance should also take shuffle into account
            if shuff_avg is not None:
                s_avg = np.concatenate((sm_pre, nan_array, sm_dur, nan_array, sm_post), axis=0)
                s_std = np.concatenate((sd_pre, nan_array, sd_dur, nan_array, sd_post), axis=0)
                figdata['coefficient_shuffle_mean'].append(s_avg)
                figdata['coefficient_shuffle_std'].append(s_std)
                for iP in range(np.size(thissig)):
                    if ~np.isnan(thissig[iP]):
                        if thismean[iP] <= s_avg[iP]:
                            if thismean[iP] + thissem[iP] > s_avg[iP] - s_std[iP]:
                                thissig[iP] = False
                        else:
                            if thismean[iP] - thissem[iP] < s_avg[iP] + s_std[iP]:
                                thissig[iP] = False

            if len(plot_what) == 1:
                figdata['coefficient_mean'].append(thismean)
                figdata['coefficient_sem'].append(thissem)
                figdata['coefficient_pval'].append(thispval)
                figdata['coefficient_FDR_isSig'].append(thissig)
                figdata['coefficient_randomEffects_mice'].append(thisre)
            else:
                if iData == 0:
                    figdata['coefficient_mean'].append(list())
                    figdata['coefficient_sem'].append(list())
                    figdata['coefficient_pval'].append(list())
                    figdata['coefficient_FDR_isSig'].append(list())
                    figdata['coefficient_randomEffects_mice'].append(list())
                figdata['coefficient_mean'][-1].append(thismean)
                figdata['coefficient_sem'][-1].append(thissem)
                figdata['coefficient_pval'][-1].append(thispval)
                figdata['coefficient_FDR_isSig'][-1].append(thissig)

            # plot shuffle shade
            if donotPlot:
                continue

            if iData == 0 and shuff_avg is not None:
                s_avg = np.concatenate((sm_pre, nan_array, sm_dur, sm_dur, sm_dur, nan_array, sm_post), axis=0)
                s_std = np.concatenate((sd_pre, nan_array, sd_dur, sd_dur, sd_dur, nan_array, sd_post), axis=0)
                t_axis_sh = np.hstack((t_axis_idx[0:nt_pre], nan_array, t_axis_idx[nt_pre+1]-.5, \
                                            t_axis_idx[nt_pre+1], t_axis_idx[nt_pre+1]+.5, nan_array, t_axis_idx[nt_pre+nt_during+2:np.size(t_axis_idx)]))
                ax.fill_between(t_axis_sh,s_avg-s_std,s_avg+s_std,color=[.8,.8,.8,.3])

            # random effects (clip outliers for plotting only)
            # range calculated to clip datapoints beyond the [1 99] CI
            thisre[thisre>1.5] = np.nan
            thisre[thisre<-2]  = np.nan
            for iT in range(np.size(thisre,axis=0)):
                ax.plot(t_axis_idx,thisre[iT,:],color=thiss[iData],linewidth=.5, \
                        marker='x',markerfacecolor='none',markersize=4)

            # avg +/- sem
            ax.errorbar(t_axis_idx, thismean, thissem, marker='.', ls='-', c=thisc[iData])

            # significance circles
            if iData == len(plot_what)-1:
                thissig[np.isnan(thissig)] = False

                thisy   = thismean-thissem
                utils.plotPvalCircles(ax, t_axis_idx, thisy, thispval, thissig.astype(bool), where='below',color='k')

        if donotPlot:
            continue

        ax.set_xlim([t_axis_idx[0]-.5, t_axis_idx[-1]+.5])
        ax.set_xticklabels('')
        ax.set_xticks(t_axis_idx)

        # labels, titles, y limits
        if regr_type == 'combinedEpochs':
            if plot_what[0].find('norm') == -1:
                ax.set_ylim([-.5,.75])
            else:
                ax.set_ylim([-2.05, 1.55])
                ax.set_yticks(range(-2,2,1))

            ax.set_title(datalbl[iCond])
            if plot_ct == 6:
                ax.set_xticklabels(t_axis_lbl, rotation=90)

        elif regr_type == 'singleEpochs':
            if plot_what[0].find('norm') == -1:
                ax.set_ylim([-.5,.75])
            else:
                ax.set_ylim([-2.05, 1.55])
                ax.set_yticks(range(-2,2,1))

            if plot_ct < 6:
                ax.set_title(lr_time['data_lbl'][iCond][1])
            elif plot_ct == 48:
                ax.set_xticklabels(t_axis_lbl, rotation=90)

            if np.remainder(iCond, 6) == 0:
                ax.set_ylabel(lr_time['data_lbl'][iCond][0])

        elif regr_type == 'epochSets':
            if plot_what[0].find('norm') == -1:
                ax.set_ylim([-.5,.75])
            else:
                ax.set_ylim([-2.05, 1.55])
                ax.set_yticks(range(-2,2,1))

            if plot_ct < 3:
                ax.set_title(lr_time['data_lbl'][iCond][1])
            elif plot_ct == 24:
                ax.set_xticklabels(t_axis_lbl, rotation=90)

            if np.remainder(iCond, 3) == 0:
                ax.set_ylabel(lr_time['data_lbl'][iCond][0])

        utils.applyPlotDefaults(ax)

    return fig, figdata

# ===========================================================================
# posterior vs anterior comparison
def plot_lrtime_postVSfront(lr_time,plot_what='evid_diff_norm',doMedian=False,includeSim=True,ax=None):

    # colors
    lc = [[0,0,0,1],[.99,.35,.31,1]]
    sc = [[.5,.5,.5,.5],[.99,.3,.3,.5]]

    # area indices
    area_list     = params['inact_locations']
    post_sim_idx  = area_list.index('Post')
    front_sim_idx = area_list.index('Front')
    if includeSim:
        post_idx  = range(post_sim_idx+1)
        front_idx = range(post_sim_idx+1,front_sim_idx+1)
    else:
        post_idx  = range(post_sim_idx)
        front_idx = range(post_sim_idx+1,front_sim_idx)

    # overall time axis
    nan_array   = np.ones((1)) * np.nan
    axis_pre    = lr_time['results'][1]['evidence_vals'][0]
    nt_pre      = np.size(axis_pre)
    idx_pre     = range(0,nt_pre)
    axis_during = lr_time['results'][1]['evidence_vals'][1]
    nt_during   = np.size(axis_during)
    idx_during  = range(nt_pre,nt_pre+nt_during)
    axis_post   = lr_time['results'][1]['evidence_vals'][2]
    nt_post     = np.size(axis_post)
    idx_post    = range(nt_pre+nt_during,nt_pre+nt_during+nt_post)
    t_axis      = np.concatenate((-1 * np.flip(axis_pre), nan_array,- 1 * np.flip(axis_during), nan_array, axis_post), axis=0)

    t_axis_idx = np.arange(np.size(t_axis))
    t_axis_lbl = list()
    for lbl in list(t_axis):
        if np.isnan(lbl):
            t_axis_lbl.append('')
        else:
            t_axis_lbl.append(np.array2string(lbl))

    if ax is None:
        fig = plt.figure(figsize=(3,3))
        ax  = plt.gca()

    # zero line, lsr shade
    ax.plot([t_axis_idx[0], t_axis_idx[-1]], [0, 0], ':', color=[0, 0, 0])
    fillx  = [t_axis_idx[4]-.5, t_axis_idx[4]+.5, t_axis_idx[4]+.5, t_axis_idx[4]-.5]
    filly  = [-1.5, -1.5, 1, 1]
    fillcl = params['lsr_color']
    ax.fill(fillx,filly,fillcl,alpha=.25)

    figdata = [None] * 2
    labels  = ['Posterior','Frontal']
    for iPlot in range(2):

        if iPlot == 0:
            area_idx = post_idx
        else:
            area_idx = front_idx

        #  isolated inact
        area_ct = 0
        for iArea in area_idx:
            if area_ct == 0:
                vec      = lr_time['results'][iArea]['coeff']['{}'.format(plot_what)]
            else:
                vec      = np.vstack((vec,lr_time['results'][iArea]['coeff']['{}'.format(plot_what)]))
            area_ct = area_ct + 1

        # concatenate coefficients
        ev_vals  = lr_time['results'][0]['evidence_vals']
        if np.isnan(ev_vals[0][0]):
            this_nt_pre  = 0
            this_pre     = np.ones((area_ct,nt_pre)) * np.nan
        else:
            this_nt_pre  = nt_pre
            this_pre     = np.flip(vec[:,idx_pre])

        if np.isnan(ev_vals[1][0]):
            this_nt_dur  = 0
            this_dur     = np.ones((area_ct,nt_during)) * np.nan
        else:
            this_nt_dur  = nt_during
            this_idx     = range(this_nt_pre,this_nt_pre+this_nt_dur)
            this_dur     = np.flip(vec[:,this_idx])

        if np.isnan(ev_vals[2][0]):
            this_nt_post = 0
            this_post    = np.ones((area_ct,nt_post)) * np.nan
        else:
            this_nt_post = nt_post
            this_idx     = range(this_nt_pre+this_nt_dur,this_nt_pre+this_nt_dur+this_nt_post)
            this_post    = vec[:,this_idx]

        nan_array = np.ones((area_ct,1)) * np.nan
        vals      = np.hstack((this_pre, nan_array, this_dur, nan_array, this_post))
        figdata[iPlot] = vals
        if doMedian:
            thismean  = np.median(vals,axis=0)
            thissem   = sp.stats.iqr(vals,axis=0)
        else:
            thismean  = np.mean(vals,axis=0)
            thissem   = np.std(vals,axis=0,ddof=1) / np.sqrt(area_ct-1)

        # plot data
        for iT in range(np.size(vals,axis=0)):
            ax.plot(t_axis_idx,vals[iT,:],color=sc[iPlot],linewidth=.5, \
                    marker='x',markerfacecolor='none',markersize=4)
        ax.errorbar(t_axis_idx, thismean, thissem, marker='.', ls='-',color=lc[iPlot],linewidth=.75)

        plt.text(t_axis_idx[1],.8-.1*iPlot,labels[iPlot],color=lc[iPlot],fontsize=7)

    # labels etc
    ax.set_xlim([t_axis_idx[0]-.5, t_axis_idx[-1]+.5])
    ax.set_ylim([-1.5,.5])
    ax.set_xticks(t_axis_idx)
    ax.set_xticklabels(t_axis_lbl, rotation=90)

    utils.applyPlotDefaults(ax)

    # Do 2-way RM anova
    nr          = np.size(figdata[0],axis=0)
    data_post   = np.array(figdata[0][~np.isnan(figdata[0])])
    data_post   = data_post.reshape(nr,int(np.size(data_post)/nr))
    nr          = np.size(figdata[1],axis=0)
    data_front  = np.array(figdata[1][~np.isnan(figdata[1])])
    data_front  = data_front.reshape(nr,int(np.size(data_front)/nr))
    areas_post  = np.zeros(np.size(data_post))
    areas_front = np.ones(np.size(data_front))
    nc          = np.size(data_post,axis=1)
    nr          = np.size(data_post,axis=0)
    bins_post   = np.transpose(np.repeat(np.arange(nc),nr).reshape(nc,nr))
    ids_post    = np.repeat(np.arange(nr),nc).reshape(nr,nc)
    nc          = np.size(data_front,axis=1)
    nr          = np.size(data_front,axis=0)
    bins_front  = np.transpose(np.repeat(np.arange(nc),nr).reshape(nc,nr))
    ids_front   = np.repeat(np.arange(nr),nc).reshape(nr,nc)
    table       = pd.DataFrame({'time_bin'   : np.concatenate((bins_post.flatten(),bins_front.flatten())),
                                'cond'       : np.concatenate((ids_post.flatten(),ids_front.flatten())),
                                'area_group' : np.concatenate((areas_post.flatten(),areas_front.flatten())),
                                'coeff'      : np.concatenate((data_post.flatten(),data_front.flatten()))})

    anova_table = pg.rm_anova(dv='coeff', within=['time_bin','area_group'],subject='cond',data=table)

    pvals = anova_table['p-unc'].to_numpy()
    plt.text(t_axis_idx[5]+.5,-.9,'p(time)={:1.2g}'.format(pvals[0]),color='k',fontsize=7)
    plt.text(t_axis_idx[5]+.5,-1.1,'p(area)={:1.2g}'.format(pvals[1]),color='k',fontsize=7)
    plt.text(t_axis_idx[5]+.5,-1.3,'p(time*area)={:1.2g}'.format(pvals[2]),color='k',fontsize=7)

    return ax, anova_table

# ===========================================================================
# simulatenous vs avg (triggered logistic regression in space)
def plot_simultaneous_vs_avg(lr_time,plot_what='evid_diff_norm'):

    """
    fig, figdata = plot_simultaneous_vs_avg(lr_time,plot_what='evid_diff_norm')
    plots results of mixed effects logistic regression in time, comparing averages
    of individual areas with their simulatenous inactivation
    lr_time is dictionary output of batch_mixed_logRegr* functions
    plot_what what to plot on y axis, default 'evid_diff_norm'
    returns figure handle fig and dictionary figdata with source data and stats
    """

    # area indices
    area_list     = params['inact_locations']
    post_sim_idx  = area_list.index('Post')
    post_idx      = range(post_sim_idx)
    front_sim_idx = area_list.index('Front')
    front_idx     = range(post_sim_idx+1,front_sim_idx)

    # overall time axis
    nan_array   = np.ones((1)) * np.nan
    axis_pre    = lr_time['results'][1]['evidence_vals'][0]
    nt_pre      = np.size(axis_pre)
    idx_pre     = range(0,nt_pre)
    axis_during = lr_time['results'][1]['evidence_vals'][1]
    nt_during   = np.size(axis_during)
    idx_during  = range(nt_pre,nt_pre+nt_during)
    axis_post   = lr_time['results'][1]['evidence_vals'][2]
    nt_post     = np.size(axis_post)
    idx_post    = range(nt_pre+nt_during,nt_pre+nt_during+nt_post)
    t_axis      = np.concatenate((-1 * np.flip(axis_pre), nan_array,- 1 * np.flip(axis_during), nan_array, axis_post), axis=0)

    t_axis_idx = np.arange(np.size(t_axis))
    t_axis_lbl = list()
    for lbl in list(t_axis):
        if np.isnan(lbl):
            t_axis_lbl.append('')
        else:
            t_axis_lbl.append(np.array2string(lbl))

    # set up source data
    figdata = dict()
    figdata['coefficient_type']               = plot_what
    figdata['inactivation_condition_labels']  = ['Posterior_simultaneous','Frontal_simulatenous','Posterior_avg','Frontal_avg']
    figdata['time_axis']                      = t_axis
    figdata['time_axis_unit']                 = 'seconds'
    figdata['coefficient_mean']               = list()
    figdata['coefficient_sem']                = list()
    figdata['coefficient_pval']               = list()
    figdata['coefficient_FDR_isSig']          = list()
    figdata['coefficient_shuffle_mean']       = list()
    figdata['coefficient_shuffle_std']        = list()
    figdata['coefficient_randomEffects_mice'] = list()

    # first plot post vs frontal simulatenous comparison
    fig    = plt.figure(figsize=(6,1.5))
    ax     = fig.add_subplot(1,3,1)
    areacl = utils.getAreaColors(['Post','Front'])
    shade  = [[.5,.9,.5,.65],[.9,.5,.5,.65]]

    ax.plot([t_axis_idx[0], t_axis_idx[-1]], [0, 0], ':', color=[0, 0, 0])
    fillx  = [t_axis_idx[4]-.5, t_axis_idx[4]+.5, t_axis_idx[4]+.5, t_axis_idx[4]-.5]
    filly  = [-3, -3, 3, 3]
    fillcl = params['lsr_color']
    ax.fill(fillx,filly,fillcl,alpha=.25)

    for iPlot in range(2):
        if iPlot == 0:
            iCond = post_sim_idx
        else:
            iCond = front_sim_idx

        # concatenate vectors into a single time axis
        vec      = lr_time['results'][iCond]['coeff']['{}'.format(plot_what)]
        vec_sem  = lr_time['results'][iCond]['coeff']['{}_sem'.format(plot_what)]
        vec_re   = lr_time['results'][iCond]['coeff']['{}_re'.format(plot_what)]
        vec_p    = lr_time['results'][iCond]['coeff']['{}_pvals'.format(plot_what)]
        try:
            vec_sig  = lr_time['results'][iCond]['coeff']['{}_isSig'.format(plot_what)]
        except:
            vec_sig  = np.ones(np.size(vec_p)) * False

        ev_vals  = lr_time['results'][iCond]['evidence_vals']
        nmice    = np.size(vec_re,axis=0)
        nan_vec  = np.ones((nmice,1)) * np.nan

        if np.isnan(ev_vals[0][0]):
            this_nt_pre  = 0
            mean_pre     = np.ones(nt_pre) * np.nan
            sem_pre      = np.ones(nt_pre) * np.nan
            p_pre        = np.ones(nt_pre) * np.nan
            sig_pre      = np.ones(nt_pre) * np.nan
            re_pre       = np.ones((nmice,nt_pre)) * np.nan

        else:
            this_nt_pre  = nt_pre
            mean_pre     = np.flip(vec[idx_pre])
            sem_pre      = np.flip(vec_sem[idx_pre])
            p_pre        = np.flip(vec_p[idx_pre])
            sig_pre      = np.flip(vec_sig[idx_pre])
            re_pre       = np.flip(vec_re[:,idx_pre])

        if np.isnan(ev_vals[1][0]):
            this_nt_dur  = 0
            mean_dur     = np.ones(nt_during) * np.nan
            sem_dur      = np.ones(nt_during) * np.nan
            p_dur        = np.ones(nt_during) * np.nan
            sig_dur      = np.ones(nt_during) * np.nan
            re_dur       = np.ones((nmice,nt_during)) * np.nan

        else:
            this_nt_dur  = nt_during
            this_idx     = range(this_nt_pre,this_nt_pre+this_nt_dur)
            mean_dur     = np.flip(vec[this_idx])
            sem_dur      = np.flip(vec_sem[this_idx])
            p_dur        = np.flip(vec_p[this_idx])
            sig_dur      = np.flip(vec_sig[this_idx])
            re_dur       = np.flip(vec_re[:,this_idx])

        if np.isnan(ev_vals[2][0]):
            this_nt_post = 0
            mean_post    = np.ones(nt_post) * np.nan
            sem_post     = np.ones(nt_post) * np.nan
            p_post       = np.ones(nt_post) * np.nan
            sig_post     = np.ones(nt_post) * np.nan
            re_post      = np.ones((nmice,nt_post)) * np.nan

        else:
            this_nt_post = nt_post
            this_idx     = range(this_nt_pre+this_nt_dur,this_nt_pre+this_nt_dur+this_nt_post)
            mean_post    = vec[this_idx]
            sem_post     = vec_sem[this_idx]
            p_post       = vec_p[this_idx]
            sig_post     = vec_sig[this_idx]
            re_post      = vec_re[:,this_idx]

        thismean = np.concatenate((mean_pre, nan_array, mean_dur, nan_array, mean_post), axis=0)
        thissem  = np.concatenate((sem_pre, nan_array, sem_dur, nan_array, sem_post), axis=0)
        thispval = np.concatenate((p_pre, nan_array, p_dur, nan_array, p_post), axis=0)
        thissig  = np.concatenate((sig_pre, nan_array, sig_dur, nan_array, sig_post), axis=0)
        thisre   = np.hstack((re_pre, nan_vec, re_dur, nan_vec, re_post))

        figdata['coefficient_mean'].append(thismean)
        figdata['coefficient_sem'].append(thissem)
        figdata['coefficient_pval'].append(thispval)
        figdata['coefficient_FDR_isSig'].append(thissig)
        figdata['coefficient_randomEffects_mice'].append(thisre)

        # random effects (clip outliers for plotting only)
        # range calculated to clip datapoints beyond the [1 99] CI
        thisre[thisre>1.5] = np.nan
        thisre[thisre<-2]  = np.nan
        for iT in range(np.size(thisre,axis=0)):
            ax.plot(t_axis_idx,thisre[iT,:],color=shade[iPlot],linewidth=.5, \
                    marker='x',markerfacecolor='none',markersize=4)

        # avg +/- sem
        ax.errorbar(t_axis_idx, thismean, thissem, marker='.', ls='-', c=areacl[iPlot],linewidth=1)

    # compare coefficients between front and post using z stats, plot pvals
    coeff_post      = figdata['coefficient_mean'][0]
    coeff_post_sem  = figdata['coefficient_sem'][0]
    coeff_front     = figdata['coefficient_mean'][1]
    coeff_front_sem = figdata['coefficient_sem'][1]
    z        = (coeff_post - coeff_front)/(np.sqrt((coeff_post*coeff_post_sem)**2+(coeff_front*coeff_front_sem)**2))
    pvals    = sp.stats.norm.sf(abs(z))*2
    isSig, _ = utils.FDR(pvals)

    figdata['coeff_postVSfront_zstats'] = z
    figdata['coeff_postVSfront_pval']   = pvals
    figdata['coeff_postVSfront_isSig']  = isSig

    isSig[np.isnan(isSig)] = False
    thisy                  = coeff_front-coeff_front_sem
    utils.plotPvalCircles(ax, t_axis_idx, thisy, thispval, isSig.astype(bool), where='below',color='k')

    ax.set_xlim([t_axis_idx[0]-.5, t_axis_idx[-1]+.5])
    ax.set_xticklabels(t_axis_lbl, rotation=90)
    ax.set_xticks(t_axis_idx)

    # labels, titles, y limits
    if plot_what.find('norm') == -1:
        ax.set_ylim([-.5,.75])
    else:
        ax.set_ylim([-2.05, 1.55])
        ax.set_yticks(range(-2,2,1))

    utils.applyPlotDefaults(ax)

    # plot simultaneous vs avg comparison
    for iPlot in range(2):
        ax      = fig.add_subplot(1,3,iPlot+2)

        if iPlot == 0:
            vec_sim     = lr_time['results'][post_sim_idx]['coeff']['{}'.format(plot_what)]
            vec_sim_sem = lr_time['results'][post_sim_idx]['coeff']['{}_sem'.format(plot_what)]
            area_idx    = post_idx
        else:
            vec_sim     = lr_time['results'][front_sim_idx]['coeff']['{}'.format(plot_what)]
            vec_sim_sem = lr_time['results'][front_sim_idx]['coeff']['{}_sem'.format(plot_what)]
            area_idx    = front_idx

        # average isolated inact
        area_ct = 1
        for iArea in area_idx:
            # concatenate vectors into a single time axis
            if area_ct == 1:
                vals     = lr_time['results'][iArea]['coeff']['{}'.format(plot_what)]
                vec_sem  = lr_time['results'][iArea]['coeff']['{}_sem'.format(plot_what)]**2
            else:
                vals     = np.vstack((vals,lr_time['results'][iArea]['coeff']['{}'.format(plot_what)]))
                vec_sem  = vec_sem + lr_time['results'][iArea]['coeff']['{}_sem'.format(plot_what)]**2
            area_ct = area_ct + 1

        vec         = np.mean(vals,axis=0)
        vec_sem     = np.sqrt(vec_sem) / area_ct

        # concatenate coefficients
        ev_vals  = lr_time['results'][0]['evidence_vals']
        if np.isnan(ev_vals[0][0]):
            this_nt_pre  = 0
            mean_pre     = np.ones(nt_pre) * np.nan
            sem_pre      = np.ones(nt_pre) * np.nan
            vals_pre     = np.ones((np.size(vals,axis=0),nt_pre)) * np.nan
            mean_pre_sim = np.ones(nt_pre) * np.nan
            sem_pre_sim  = np.ones(nt_pre) * np.nan
        else:
            this_nt_pre  = nt_pre
            mean_pre     = np.flip(vec[idx_pre])
            vals_pre     = np.flip(vals[:,idx_pre])
            sem_pre      = np.flip(vec_sem[idx_pre])
            mean_pre_sim = np.flip(vec_sim[idx_pre])
            sem_pre_sim  = np.flip(vec_sim_sem[idx_pre])

        if np.isnan(ev_vals[1][0]):
            this_nt_dur  = 0
            mean_dur     = np.ones(nt_during) * np.nan
            vals_dur     = np.ones((np.size(vals,axis=0),nt_dur)) * np.nan
            sem_dur      = np.ones(nt_during) * np.nan
            mean_dur_sim = np.ones(nt_during) * np.nan
            sem_dur_sim  = np.ones(nt_during) * np.nan
        else:
            this_nt_dur  = nt_during
            this_idx     = range(this_nt_pre,this_nt_pre+this_nt_dur)
            mean_dur     = np.flip(vec[this_idx])
            vals_dur     = np.flip(vals[:,this_idx])
            sem_dur      = np.flip(vec_sem[this_idx])
            mean_dur_sim = np.flip(vec_sim[this_idx])
            sem_dur_sim  = np.flip(vec_sim_sem[this_idx])

        if np.isnan(ev_vals[2][0]):
            this_nt_post = 0
            mean_post    = np.ones(nt_post) * np.nan
            vals_post    = np.ones((np.size(vals,axis=0),nt_post)) * np.nan
            sem_post     = np.ones(nt_post) * np.nan
            mean_post_sim = np.ones(nt_post) * np.nan
            sem_post_sim  = np.ones(nt_post) * np.nan
        else:
            this_nt_post = nt_post
            this_idx     = range(this_nt_pre+this_nt_dur,this_nt_pre+this_nt_dur+this_nt_post)
            mean_post    = vec[this_idx]
            vals_post    = vals[:,this_idx]
            sem_post     = vec_sem[this_idx]
            mean_post_sim = vec_sim[this_idx]
            sem_post_sim  = vec_sim_sem[this_idx]

        thismean     = np.concatenate((mean_pre, nan_array, mean_dur, nan_array, mean_post), axis=0)
        thissem      = np.concatenate((sem_pre, nan_array, sem_dur, nan_array, sem_post), axis=0)
        thismean_sim = np.concatenate((mean_pre_sim, nan_array, mean_dur_sim, nan_array, mean_post_sim), axis=0)
        thissem_sim  = np.concatenate((sem_pre_sim, nan_array, sem_dur, nan_array, sem_post_sim), axis=0)

        figdata['coefficient_mean'].append(thismean)
        figdata['coefficient_sem'].append(thissem)
        figdata['coefficient_pval'].append(None)
        figdata['coefficient_FDR_isSig'].append(None)
        figdata['coefficient_randomEffects_mice'].append(None)

        nan_vec      = np.ones((np.size(vals,axis=0),1))*np.nan
        thisvals     = np.hstack((vals_pre, nan_vec, vals_dur, nan_vec, vals_post))

        # zero line and data
        ax.plot([t_axis_idx[0], t_axis_idx[-1]], [0, 0], ':', color=[0, 0, 0])
        fillx  = [t_axis_idx[4]-.5, t_axis_idx[4]+.5, t_axis_idx[4]+.5, t_axis_idx[4]-.5]
        filly  = [-1.5, -1.5, 1, 1]
        fillcl = params['lsr_color']
        ax.fill(fillx,filly,fillcl,alpha=.25)
        ax.errorbar(t_axis_idx, thismean, thissem, marker='.', ls='-',color='gray',linewidth=1)
        ax.errorbar(t_axis_idx, thismean_sim, thissem_sim, marker='.', ls='-',color=areacl[iPlot],linewidth=1)

        # compare coefficients between simulatenous and avg using z stats, plot pvals
        z        = (thismean_sim - thismean)/(np.sqrt((thismean_sim*thissem_sim)**2+(thismean*thissem)**2))
        pvals    = sp.stats.norm.sf(abs(z))*2
        isSig, _ = utils.FDR(pvals)

        if iPlot == 0:
            figdata['coeff_post_simultVSavg_zstats'] = z
            figdata['coeff_post_simultVSavg_pval']   = pvals
            figdata['coeff_post_simultVSavg_isSig']  = isSig
        else:
            figdata['coeff_front_simultVSavg_zstats'] = z
            figdata['coeff_front_simultVSavg_pval']   = pvals
            figdata['coeff_front_simultVSavg_isSig']  = isSig

        isSig[np.isnan(isSig)] = False
        thisy                  = thismean_sim-thissem_sim
        utils.plotPvalCircles(ax, t_axis_idx, thisy, pvals, isSig, where='below',color=[0,0,0])

        # labels etc
        ax.set_xlim([t_axis_idx[0]-.5, t_axis_idx[-1]+.5])
        ax.set_ylim([-1.5,.5])
        ax.set_xticks(t_axis_idx)
        ax.set_xticklabels(t_axis_lbl, rotation=90)

        plt.text(t_axis_idx[0],.7,'Simultaneous',color=areacl[iPlot],fontsize=7)
        plt.text(t_axis_idx[0],.6,'Single-area avg',color='gray',fontsize=7)

        if iPlot == 0:
            ax.set_title('Posterior')
        else:
            ax.set_title('Frontal')

        utils.applyPlotDefaults(ax)

    fig.subplots_adjust(left=.1, bottom=.1, right=.95, top=.9, wspace=.5, hspace=.5)

    return fig, figdata

# ==============================================================================
# plot laser-onset triggered logistic regression in space
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
# plot laser offset-triggered logistic regression in space
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
# simulatenous vs avg (triggered logistic regression in space)
def plot_simultaneous_vs_avg_trig(trig_logRegr):

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

# ==============================================================================
# avg and std of shuffled models
def shuffle_avg(lr_time_shuff,avg_what='evid_diff_norm'):

    num_shuff = len(lr_time_shuff['coeff'])
    num_cond  = len(lr_time_shuff['coeff'][0])
    num_coeff = np.size(lr_time_shuff['coeff'][0][0][avg_what])
    shuffs    = [None]*num_cond
    shuff_avg = [None]*num_cond

    for iCond in range(num_cond):
        for iShuff in range(num_shuff):
            if iShuff == 0:
                shuffs[iCond] = np.zeros((num_shuff,num_coeff))

            shuffs[iCond][iShuff,:] = lr_time_shuff['coeff'][iShuff][iCond][avg_what]

        shuff_avg[iCond] = dict()
        shuff_avg[iCond]['mean'] = np.mean(shuffs[iCond],axis=0)
        shuff_avg[iCond]['std']  = np.std(shuffs[iCond],axis=0,ddof=1) #/ np.sqrt(num_shuff-1)

    return shuff_avg

# ==============================================================================
# generate a table summarizing experiments
def diagnose_dataset(lg,areas=params['inact_locations'],epochs=params['inact_epochs'],convertToDf=False):

    """
    summary_table = diagnose_dataset(lg,areas=params['inact_locations'],epochs=params['inact_epochs'],convertToDf=False)
    calculates summary stats for all conditions (num trials, mice etc) and returns a dictionary or a pandas DataFrame
    """

    print('generating summary table...')

    num_areas  = len(areas)
    num_epochs = len(epochs)
    num_cond   = num_areas * num_epochs

    ct = 0
    summary = dict(area=[None]*num_cond, epoch=[None]*num_cond, num_mice=[None]*num_cond, num_sessions=[None]*num_cond, \
                   num_laser_trials=[None]*num_cond, num_control_trials=[None]*num_cond, num_total_trials=[None]*num_cond)

    # mouse, sessions, trials per condition
    for iArea in range(num_areas):
        for iEpoch in range(num_epochs):
            thisarea  = areas[iArea]
            thisepoch = epochs[iEpoch]

            numtrials = np.size(lg['choice'])
            isarea    = np.array([lg['loclbl'][iTrial]==thisarea for iTrial in range(numtrials)])
            isepoch   = np.array([lg['laserEpoch'][iTrial]==thisepoch for iTrial in range(numtrials)])
            lsridx    = isarea & isepoch

            mice     = np.unique(lg['mouseID'][lsridx])
            ctrlidx  = np.zeros(numtrials) > 1
            sessct   = 0
            for iMouse in range(np.size(mice)):
                midx     = lg['mouseID'] == mice[iMouse]
                sessions = np.unique(lg['sessionID'][np.logical_and(lsridx,midx)])
                sessct   = sessct + np.size(sessions)
                for iSession in range(np.size(sessions)):
                    sessidx = lg['sessionID'] == sessions[iSession]
                    thisidx = midx & sessidx & ~lg['laserON']
                    ctrlidx = np.logical_or(ctrlidx , thisidx)

            idx      = lsridx | ctrlidx

            summary['area'][ct]               = thisarea
            summary['epoch'][ct]              = thisepoch
            summary['num_mice'][ct]           = np.size(mice)
            summary['num_sessions'][ct]       = sessct
            summary['num_laser_trials'][ct]   = sum(lsridx)
            summary['num_control_trials'][ct] = sum(ctrlidx)
            summary['num_total_trials'][ct]   = sum(idx)

            ct = ct+1

    # total unique mice and sessions
    mice     = np.unique(lg['mouseID'])
    sessct   = 0
    for iMouse in range(np.size(mice)):
        midx     = lg['mouseID'] == mice[iMouse]
        sessions = np.unique(lg['sessionID'][midx])
        sessct   = sessct + np.size(sessions)

    summary['area'].append('total')
    summary['epoch'].append(None)
    summary['num_mice'].append(np.size(mice))
    summary['num_sessions'].append(sessct)
    summary['num_laser_trials'].append(sum(summary['num_laser_trials']))
    summary['num_control_trials'].append(sum(~lg['laserON']))
    summary['num_total_trials'].append(summary['num_laser_trials'][-1]+summary['num_control_trials'][-1])

    # convert to data DataFrame
    if convertToDf:
        summary_table = pd.DataFrame(summary)
    else:
        summary_table = summary

    return summary_table
