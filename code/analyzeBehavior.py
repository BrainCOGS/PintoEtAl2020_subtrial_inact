"""
Module to analyze behavior (of the mice or RNNs)
Includes pyschometrics, logistic regression, plotting, trial selection
Lucas Pinto 2020, lucas.pinto@northwestern.edu
"""
#!/usr/bin/env python

# Libraries
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import lp_utils as utils
import multiprocessing as mp
import deepdish as dd
import time
import copy
import os.path
import pingouin as pg
import numpy.matlib
from   os import path
from   scipy.io import loadmat
from   statsmodels.discrete.discrete_model import Logit
from   statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from   statsmodels.stats.proportion import proportion_confint

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
        'savePath'               : '/Users/lpinto/Dropbox/PintoEtAl2020_subtrial_inact/data/', # save path
        'removeThetaOutlierMode' : 'none',                 # remove outlier trials from log, 'soft', 'stringent', or 'none' (there's some pre-selection already in .mat file)
        'removeZeroEvidTrials'   : True,                   # remove control trials where #r : #l
        'excludeBadMice'         : True,                   # exclude mice with bad logistic regression functions
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
        filename = 'multiArea_inact_goodMice.hdf5'
        suffix   = 'goodMice'
    else:
        filename = 'multiArea_inact_allMice.hdf5'
        suffix   = 'allMice'

    filename = '{}{}'.format(params['savePath'],filename)

    # first check if file exists and key params match, in which case just load
    if path.exists(filename):
        data            = dd.io.load(filename)
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

            lg            = data['lg']
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
    lg = loadBehavLog(params['behavLogPath'])
    lg = cleanupBehavLog(lg, params['removeThetaOutlierMode'], params['removeZeroEvidTrials'], params['excludeBadMice'])

    # summary stats
    summary_table = diagnose_dataset(lg,convertToDf=False) # save as dictionary, deepdish crashes when saving this as pandas dataframe

    # analyze inactivations
    print('ANALYZING INACTIVATION EFFECTS...\n')
    inact_effects = multiArea_boot_perf(lg, True, params['logRegr_boot_numiter'], params['logRegr_nbins'], params['psych_bins'])

    if doSave == True:
        print('saving results...')
        data = {'lg': lg, 'params': params, 'inact_effects': inact_effects, 'summary_table': summary_table}
        dd.io.save(filename, data)

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
# Load .mat behavior log
def loadBehavLog(filepath = params['behavLogPath']):
    """
    lg = loadBehavLog(filepath = params['behavLogPath']):
    loads .mat file with behavioral log, returns log object
    """

    print('loading .mat behavior log...')

    # load behavior data as an object. Variables are attributes
    data       = loadmat(filepath, struct_as_record=False, squeeze_me=True)
    lg         = data['lg']
    lg.laserON = lg.laserON == 1 # convert to boolean

    return lg

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

    # if axisHandle is None:
        # plt.show()

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
# Analyze psychometrics and logistic regression by mouse
def behav_by_mouse(lg, doPlot=True, logRegr_nbins=params['logRegr_nbins'], psych_bins=params['psych_bins']):

    """
    mouse_data = behav_by_mouse(lg, doPlot=True, logRegr_nbins=params['logRegr_nbins'], psych_bins=params['psych_bins'])
    computes logistic regression (evidence_logRegr) and psychometrics (psychometrics) by mouse
    returns dictionary with lists of corresponding analysis outputs
    """

    mouse_ids = np.unique(lg.mouseID)
    num_mice  = np.size(mouse_ids)

    # do psychometrics by mouse
    mouse_data = dict(psych=list(), logRegr=list(), mouse_ids= mouse_ids)
    for iMouse in range(num_mice):
        mouseidx = np.logical_and(~lg.laserON,lg.mouseID == mouse_ids[iMouse])
        mouse_data['psych'].append(psychometrics(lg.choice[mouseidx],lg.nCues_RminusL[mouseidx], psych_bins))
        mouse_data['logRegr'].append(evidence_logRegr(lg.choice[mouseidx], lg.cuePos_R[mouseidx], lg.cuePos_L[mouseidx], logRegr_nbins))


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
# Flag bad trials based on view angle criteria etc
def findBadTrials(lg,removeThetaOutlierMode=params['removeThetaOutlierMode'],removeZeroEvidTrials=params['removeZeroEvidTrials']):

    """
    isBadTrial = findBadTrials(lg,removeThetaOutlierMode=params['removeThetaOutlierMode'],removeZeroEvidTrials=params['removeZeroEvidTrials'])
    takes flattened behavioral log structure and returns a 1 x num_trials boolean array
    where True indicates bad trials according to view angle criteria (refer to function
    body for details) and whether tower deletion control trials where #r = #l should be removed
    """

    # view angle outliers
    total_trials    = len(lg.choice)
    if removeThetaOutlierMode == 'soft': # exclude top 10 percentile of bad theta trials
        theta_std   = np.array([ np.std(lg.viewAngle_byYpos[iTrial]) for iTrial in np.arange(0,total_trials) ])
        large_theta = theta_std > np.percentile(theta_std,95)

    elif removeThetaOutlierMode == 'stringent': # soft + some absolute thresholds on theta values
        theta_std   = np.array([ np.std(lg.viewAngle_byYpos[iTrial]) for iTrial in np.arange(0,total_trials) ])
        large_theta = theta_std > np.percentile(theta_std,90)
        norm_theta  = np.array([ lg.viewAngle_byYpos[iTrial]/lg.viewAngle_byYpos[iTrial][-1] for iTrial in np.arange(0,total_trials) ])
        bad_norm    = np.array([ np.any(norm_theta[iTrial][0:200]>.4) or np.any(norm_theta[iTrial][200:250]>.5) or np.any(norm_theta[iTrial][250:290]>.9)
                               for iTrial in np.arange(0,total_trials) ])
        large_theta = large_theta | bad_norm

    elif removeThetaOutlierMode == 'none': # no exclusion
        large_theta = np.zeros(total_trials) > 1

    # remove #R = #L trials (from some control conditions)
    if removeZeroEvidTrials == True:
      badRL       = np.array([ np.size(lg.cuePos_R[iTrial]) == np.size(lg.cuePos_L[iTrial]) for iTrial in np.arange(0,total_trials) ])
    else:
      badRL       = np.zeros(total_trials) > 1

    isBadTrial    = large_theta | badRL
    isBadTrial    = np.logical_or(isBadTrial,lg.loclbl == 'unknown')

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

    mouse_ids  = np.unique(lg.mouseID)
    num_mice   = np.size(mouse_ids)
    isBadMouse = np.zeros(num_mice) > 1

    # do logistic regression by mouse, bad mice are ones that have logistic regression
    # coefficients not significantly different from zero
    for iMouse in range(num_mice):
        mouseidx = np.logical_and(~lg.laserON, lg.mouseID == mouse_ids[iMouse])
        lr       = boot_logRegr(lg.choice[mouseidx], lg.cuePos_R[mouseidx], lg.cuePos_L[mouseidx], numbins, numiter)
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
            trialidx = lg.mouseID == badmice[iMouse]
            isBadTrial[trialidx] = True

    # remove trials
    lg_clean = copy.deepcopy(lg)
    fields   = dir(lg_clean)
    for iF in range(len(fields)):
        if fields[iF][0] != '_':
            thisvar = getattr(lg_clean, fields[iF])
            if isinstance(thisvar, np.ndarray):
                if len(thisvar) == len(isBadTrial):
                    setattr(lg_clean, fields[iF], thisvar[isBadTrial == False])

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
    numtrials = np.size(thislg.choice)
    trialidx  = np.arange(numtrials)

    # trial difficulty is \delta / total
    nr           = np.array([np.size(thislg.cuePos_R[iTrial]) for iTrial in range(numtrials)])
    nl           = np.array([np.size(thislg.cuePos_L[iTrial]) for iTrial in range(numtrials)])
    total_towers = nr + nl
    trial_diff   = thislg.nCues_RminusL / total_towers
    isHard       = trial_diff > np.median(trial_diff)

    # bootstrap over mice, sessions, trials
    for iBoot in range(numiter):
        idx           = np.random.choice(trialidx, size=numtrials, replace=True)
        choice        = thislg.choice[idx]
        laserON       = thislg.laserON[idx]
        cuePos_L      = thislg.cuePos_L[idx]
        cuePos_R      = thislg.cuePos_R[idx]
        nCues_RminusL = thislg.nCues_RminusL[idx]
        trialType     = thislg.trialType[idx]
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
# get only relevant behavioral trials for area / epoch combination
def lg_singleCondition(lg,area,epoch):

    """
    sublg = lg_singleCondition(lg,area,epoch)
    called by multiArea_boot_perf()
    selects relevant laser and control trials for single condition
    """

    # laser trials
    numtrials = np.size(lg.choice)
    isarea    = np.array([lg.loclbl[iTrial]==area for iTrial in range(numtrials)])
    isepoch   = np.array([lg.laserEpoch[iTrial]==epoch for iTrial in range(numtrials)])
    lsridx    = isarea & isepoch
    # trialidx = np.logical_and(lg.laserEpoch == epoch, lg.loclbl == area) # this gives a future version warning

    # pick only control trials from the same mice and sessions as laser trials
    mice     = np.unique(lg.mouseID[lsridx])
    ctrlidx  = np.zeros(numtrials) > 1

    for iMouse in range(np.size(mice)):
        midx     = lg.mouseID == mice[iMouse]
        sessions = np.unique(lg.sessionID[np.logical_and(lsridx,midx)])
        for iSession in range(np.size(sessions)):
            sessidx = lg.sessionID == sessions[iSession]
            thisidx = midx & sessidx & ~lg.laserON
            ctrlidx = np.logical_or(ctrlidx , thisidx)

    idx      = lsridx | ctrlidx

    # leave just relevant trials in lg
    sublg  = copy.deepcopy(lg)
    fields = dir(sublg)
    for iF in range(len(fields)):
        if fields[iF][0] != '_':
            thisvar = getattr(sublg, fields[iF])
            if isinstance(thisvar, np.ndarray):
                if len(thisvar) == len(idx):
                    setattr(sublg, fields[iF], thisvar[idx])

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
def plot_multiArea(inact_effects,plot_category='logRegr',plot_var='coeffs_diff'):

    """
    figHandle = plot_multiArea(inact_effects,plot_category='logRegr',plot_var='coeffs_diff')
    plots inactivation effects for all inactivation conditions (one per panel)
    inact_effects is dictionary generated by multiArea_boot_perf
    plot_category is 'logRegr', 'psych' or 'percCorrect'
    plot_var is which variable to plot (diffs, ratios etc), does not apply to psych. Must match variable name exactly
    """

    data      = inact_effects[plot_category]
    areas     = inact_effects['area_list']
    numareas  = len(areas)
    epochs    = inact_effects['epoch_list']
    numepochs = len(epochs)

    if plot_category == 'percCorrect':
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
                # psych_temp['fit_y']          = data['fit_y_lsr_mean'][ct-1,:]
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

                # # print slope pvals
                # ptxt = 'p(slope) = {: 1.2g}'.format(data['slope_diff_p'][ct-1,:][0])
                # if data['slope_diff_p_isSig'][ct-1] == True:
                #     ptxt = ptxt + '*'
                # ax.text(-12,.8,ptxt,fontsize=6)

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

                # plot average ctrl and lsr data
                thismean = data['{}_mean'.format(plot_var)][idx,:]
                thisstd  = data['{}_std'.format(plot_var)][idx,:]

                ax.bar(iArea,thismean,facecolor=colors[iArea],edgecolor=colors[iArea])
                ax.errorbar(iArea,thismean,thisstd,color=colors[iArea],linewidth=2)

                p_y[iArea]   = thismean - thisstd
                p[iArea]     = data['{}_p'.format(plot_var)][idx,:]
                isSig[iArea] = data['{}_p_isSig'.format(plot_var)][idx]

            # plot pvals
            x     = np.arange(numareas)
            y     = p_y - .01
            where = 'below'
            ax    = utils.plotPvalCircles(ax, x, y, p, isSig, where, rotate=True)

            # title and axes
            ax.set_title(epochs[iEpoch],pad=0)
            ax.set_ylabel('$\Delta$ P correct')
            ax.set_xticks(x)
            ax.set_xticklabels(areas, rotation='vertical')

            # cosmetics
            ax.set_xlim([-.75, numareas-.25])
            ax.set_ylim([-.35, .01])
            utils.applyPlotDefaults(ax)

    if plot_category == 'percCorrect':
        fig.subplots_adjust(left=.1, bottom=.1, right=.95, top=.9, wspace=.5, hspace=.7)
    else:
        fig.subplots_adjust(left=.06, bottom=.05, right=.95, top=.95, wspace=.4, hspace=.7)

    # fig.show()

    return fig

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

            numtrials = np.size(lg.choice)
            isarea    = np.array([lg.loclbl[iTrial]==thisarea for iTrial in range(numtrials)])
            isepoch   = np.array([lg.laserEpoch[iTrial]==thisepoch for iTrial in range(numtrials)])
            lsridx    = isarea & isepoch

            mice     = np.unique(lg.mouseID[lsridx])
            ctrlidx  = np.zeros(numtrials) > 1
            sessct   = 0
            for iMouse in range(np.size(mice)):
                midx     = lg.mouseID == mice[iMouse]
                sessions = np.unique(lg.sessionID[np.logical_and(lsridx,midx)])
                sessct   = sessct + np.size(sessions)
                for iSession in range(np.size(sessions)):
                    sessidx = lg.sessionID == sessions[iSession]
                    thisidx = midx & sessidx & ~lg.laserON
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
    mice     = np.unique(lg.mouseID)
    sessct   = 0
    for iMouse in range(np.size(mice)):
        midx     = lg.mouseID == mice[iMouse]
        sessions = np.unique(lg.sessionID[midx])
        sessct   = sessct + np.size(sessions)

    summary['area'].append('total')
    summary['epoch'].append(None)
    summary['num_mice'].append(np.size(mice))
    summary['num_sessions'].append(sessct)
    summary['num_laser_trials'].append(sum(summary['num_laser_trials']))
    summary['num_control_trials'].append(sum(~lg.laserON))
    summary['num_total_trials'].append(summary['num_laser_trials'][-1]+summary['num_control_trials'][-1])

    # convert to data DataFrame
    if convertToDf:
        summary_table = pd.DataFrame(summary)
    else:
        summary_table = summary

    return summary_table
