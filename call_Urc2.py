# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:07:50 2024

@author: helme00l
"""

import datetime
import os
import pandas as pd
from degradation_toolbox import Urc2, example_data

# find some data somewhere ...
d=example_data()

# instantiate U_rc2 for evalauation
myU1=Urc2(d,label='testcase', configDict={'plotSummary':True, 'jref': [1, 1.5]})

# execute
resDF1=myU1.evaluate()


"""
U_r2 can be called in various ways:
arguments: dataSet=None,label=None,configDict=None):

    
    dataSet can be a fixed format pandas dataframe (as above) for onesingle cell (or avg of stack/group) with
    index(rows): calender time In in datetime format
    columns:
        'currentDensity' in A/cm^2
        'voltage' in V (cell level)
        'temperature' in °C representative cell temperature, preferably control paramter = stack exit temeprature
    the name may be given as additional argument
    if dataSet is NOT given (default):
        data can be handet over as list of files (multiple cells possible!)
        in *.parquet format with same naming convention as dataFrame above.
        in *.pkl format with same index as above and columns
        Tavg_C   represenatavie temperature
        Ucell_V  cell voltage
        j_A/cm2  current Density
        dtime_h  time since start in h 
        t.p.d    time per day ':0...24h in timestamp format
        
        the same list can be passed as item in configDict, specifiying other details of instance at the same time:
            
            'plotSummary':False # plot of all proccessed data sets and current densities including linear regressiona nd projection to 
            'plotFit':False # all individual log fits (huge amount of plots ...)
            'plotTS':False # plot the time series of voltage and currentDensity including fits
            'summary':False # some further processing plots
            'export':False # storage of central fitting data in *.csv files
            'FATflag':False # run evalaution after typica FAT run time (15h)
            'ReSample':1 # drop n points to reduce resolution, no averaging!
            'NfitLim ': 100 # min number of valid data points to start a fit at all
            'SaveFigs':False # automarted storage of plots
    
            'jOff':0.1 # [A/cm2] # threshold value to define on/off state
            'Uoff':1.3 # [V] # threshold value to define on/off state
            'dl':0.015 # load filter: bin size [A/cm2] # polcurve slope ~0.2v/(A/cm2) constrain scatter to U-meas err ~3mV --> .003/.2 ': .015 [A/cm2]
            'AddMoreStarts':True # flag to initiate set-up of 'artificial starts' during long operation periods without stops (chopping the time interval)
            't_aS':48 # [h] time to chop long periods add artificial starts
            'localT':True # use local Temperature fit instead of Michel's standard
            'artStart':False # use arificial starts for fitting: True , or real starts: False <':': use this one!! (True for review) True cuases the fake start to re set the log-time scale
            'addFilter':True # do som earbitrary filtering on raw data and Uref
            'myU':  'U60_V' # 'Ucell_V' # parameter consdered for regression # U_cell would skip T- correction alltogether
            'jEvalList':  'auto' #[1.55,0.6]# 'viewOne&quit' #  'viewAll&quit' # # [1.5,1.4,1.0,.95,.5,.4,.3]#,1,0.5,0.3]   # value # LinzM12:[0.292,0.307,0.941,0.951]; M10
            'file_suffix':'' # manipuilation of text in plots
            'opPressure':10.0 # for file label only
            'RunIn':[0,0]#[h] 3*168 # time fram for initial aging requires t_start':':0
            't_start':1 # [h] earliest possible start
            't_stop':np.inf # [h] truncation of evalutation
            'U_lim':[1.3,2.3] # limits for U_ref truncation (skip nonsense fits) 
            
            'fitLbl':'log' # label for fitting style
            # various functions required to set up fitting styl allowing for other then log-shape / currently exp(a*exp(bt)) is defined  as 'expexp
            'params0':'params0_log' # 
            'fit':'fit_log'
            'der':'der_log'
            'fitErr':'fit_logErr'
            'residual':'residual_log'
            
            # same including local temperature fit
            'params0_logT':'params0_logT'
            'fitT':'fitT'
            'fitT_err':'fitT_err'
            'residualT':'residualT'

            # reference time
            't_int':168/2 # ref time for U_rc defintion [h] 240
            # alternate referenc time
            't_init':5/60 # alternate evalaution e.g. shortly after restar [h]
            # reference temperature
            'Tref':60 # [°C] if None: average temperature of run dataset is used.
            # refence current density
            'jref':1.5 # [A/cm2]
    
            # label for filenames and in plots
            'addlabel':'' # first line on aging plots Uref vs t if not empty
"""
            
        
