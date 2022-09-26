# -*- coding: utf-8 -*-
"""
Basic download script of subx variables needed for calculating reference ET:
radiation, humidity, temperature, and wind speed.

Originally written by Shrard Shukla, modified by John Volk
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import datetime
from dateutil.relativedelta import relativedelta
import requests
import calendar
import urllib.request
from datetime import date, datetime



CURRENT_DATE, INIT_DATE = date.today(), date.today() + relativedelta(days=-28)
DAYS_TS = pd.date_range(INIT_DATE, CURRENT_DATE)

print ("Downloading data from {} to {}".format(INIT_DATE, CURRENT_DATE))

# Modeling Groups (must be same # elements as models below)
groups=['ECCC', 'EMC', 'GMAO'] 
# Model Name (must be same # of elements as groups above)
models=['GEPS7', 'GEFSv12', 'GEOS_V2p1'] 


## The following variables are to be downloaded daily
varnames=['tas', 'uas', 'vas', 'tdps', 'rad', 'dswrf']
plevstrs=['2m', '10m', '10m', '2m', 'sfc', 'sfc']

## Location of download and file format
BASE_INDIR = 'subx/forecast'
INFILE_template = 'http://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/.{}/.{}/.{}/.{}/S/%280000%20{:02d}%20{}%20{:04d}%29VALUES/{}'

for NUM, GROUP in enumerate(groups):
    for TSTEP_NUM, TSTEP in enumerate(DAYS_TS):
        DAY, MON, YEAR = TSTEP.day,  TSTEP.month, TSTEP.year
        for VAR_NUM, VAR in enumerate(varnames):
            DOD = INFILE_template.format(
                GROUP, models[NUM], 'forecast', VAR, DAY, 
                calendar.month_abbr[MON], YEAR, 'dods'
            )
            request = requests.get(DOD + '.dds')
            if request.status_code == 200:
                DATA = xr.open_dataset(DOD, cache=False)
                CHECK_VAL = float(DATA.isel(M=0, L=0).sum()[VAR].values)
                if np.isfinite(CHECK_VAL) and (CHECK_VAL>0) and len(DATA.S)==1:
                    DOWNLOAD_FILE = INFILE_template.format(
                        GROUP, models[NUM], 'forecast', VAR, DAY, 
                        calendar.month_abbr[MON], YEAR, 'data.nc'
                    )
                    OUTFILE_template =\
                            '{}/{}_{}_{}-{}_{:04d}{:02d}{:02d}.daily.nc'
                    OUTDIR_template = '{}/{}/daily/{}-{}/{:02d}'
                    OUTDIR = Path(
                        OUTDIR_template.format(
                            BASE_INDIR, VAR, GROUP, 
                            models[NUM], MON
                        )
                    )
                    if OUTDIR.is_dir():
                        pass
                    else:
                        OUTDIR.mkdir(exist_ok=True, parents=True)
                    OUTFILE = OUTFILE_template.format(
                        OUTDIR, VAR, plevstrs[VAR_NUM], GROUP, 
                        models[NUM], YEAR, MON, DAY
                    )
                    urllib.request.urlretrieve(DOWNLOAD_FILE, OUTFILE)
                    CHECK_DATA = xr.open_dataset(OUTFILE)
                    print ("Writing {} Check value {}".format(
                        OUTFILE, 
                        float(CHECK_DATA.isel(M=1, L=0).mean()[VAR].values))
                    )
                else:
                    print ("Data not valid for {} {} {} {}".format(
                        models[NUM], DAY, MON, YEAR)
                    )
                    pass
            else:
                print ("Couldn't access {}".format(DOD))
                pass
