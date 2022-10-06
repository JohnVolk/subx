# -*- coding: utf-8 -*-

"""
Main module for computing weekly forecasted ETo anomalies from subx. Input ETo
netCDFs that are pre-downloaded are used, using the most recent download date. NetCDFs, and plots of the 4 week anomalies are saved as a final result.

Note: currently, one model is run at a time using one ensemble member or initialization but this will be modified to take an ensemble mean or median for each member and across models.
"""

import re
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from dask.distributed import Client, LocalCluster
from refet import asce_refet, make_elev_like, make_jday_like, make_lat_like

########
## control variables, for now just assign here: 
elev_path = 'cfs_z_regrid_to_subx.nc' # path to regridded elevation netCDF
baseline_dir = '..' # directory with baseline anomalies
output_data_dir = None # if None, netCDFs will be written to "anomaly_data" dir
model = 'GMAO_GEOS_V2p1' # can run GMAO_GEOS_V2p1 or ECCC_GEPS7
chunks = {'X':100,'Y':100} # chunk size for dask (ETo)
########


# static parameters and lookups- may move to submodule:
model_var_data = {
    'EMC_GEFS':dict(
        tmean='tas', hum='tdps', rad='dswrf', uwind='uas', vwind='vas', 
        hum_type=2, rd_type=1
    ),
    'GMAO_GEOS_V2p1':dict(
        tmean='tas', hum='tdps', rad='dswrf', uwind='uas', vwind='vas', 
        hum_type=2, rd_type=1
    ),
    'ECCC_GEPS7':dict(
        tmean='tas', hum='tdps', rad='rad', uwind='uas', vwind='vas', 
        hum_type=2, rd_type=2
    ) # rad is net rad
}

var_heights = {
    'tas':'2m',
    'tdps':'2m',
    'dswrf':'sfc',
    'rad':'sfc',
    'uas':'10m',
    'vas':'10m'
}
ETo_vars = ['tmean', 'hum', 'rad', 'uwind', 'vwind']

varying_input_vars = [
    'tmean', 'hum', 'rad', 'uwind', 'vwind', 'hum_type', 'rd_type'
]
# other variables for ETo: lat, z (subx elevation), fjday (day of year), 
# ref (ASCE std. ref. grass (1) or alfalfa (2)),

# This dict. just maps names of models to baseline netCDF names 
# if model versions change may need to update/add
baseline_model_name = {
    'ECCC_GEPS7':'ECCC-GEPS6',
    'GMAO_GEOS_V2p1':'GMAO-GEOS_V2p1'
}

week_end_days = [7,14,21,28]

def wind_spd_u_v(u, v):
    return np.sqrt(u**2 + v**2)

def _chunk_inputs(da, chunk_dict):
    return da.chunk(chunk_dict)

################### Calculations below
if __name__ == "__main__":

    print(f'Running forecast ETo weekly anomalies for model: {model}')
    print('Starting dask client')
    # set up cluster and workers for parallelization
    cluster = LocalCluster(n_workers=8, threads_per_worker=1)
    client = Client(cluster)
    print(f'View dask dashboard at: {client.dashboard_link}')

    #### Collect ETo input netCDFs- assumes downloaded to "subx/forecasts/" 
    #### as a result of running download_forecasts.py

    download_data = {}
    vars_downloaded = [v.stem for v in  Path('subx/forecast/').glob('*')]
    vars_downloaded

    filename_date_re = re.compile('.*_(\d{8})\.daily\.nc')

    for v in vars_downloaded:
        models_per_var = Path(f'subx/forecast/{v}/daily/').glob('*')
        var_files = Path(f'subx/forecast/{v}/daily/').rglob('*.nc')
        for m in models_per_var:
            # replacing dash from download script between group-model
            model_name = m.stem.replace('-','_')
            if not model_name in download_data.keys():
                download_data[model_name] = {v:list(var_files)}
            else:
                download_data[model_name].update({v:list(var_files)})

            var_files = download_data[model_name].get(v)
            dates = [
                filename_date_re.search(f.name).group(1) for f in var_files
            ]
            download_data[model_name][f'{v}_dates'] = dates


    models_downloaded = [m for m in download_data.keys()]

    # check on vars downloaded make sure input for Ref ET was pulled (vary by model)
    needed_vars = set(
        filter(lambda x: type(x) == str, model_var_data.get(model).values())
    )

    ##TODO: if check fails skip everything further down
    have_needed_vars = needed_vars.issubset(vars_downloaded)

    # a check on each variable in needed_vars for the latest date
    var_dates = {}
    for v in needed_vars:
        file_date_strs = download_data.get(model).get(f'{v}_dates')
        var_dates[v] = file_date_strs

    # get the oldest date from each variable in case one is not updated
    most_recent_date = max(set.intersection(*map(set,var_dates.values())))

    year = most_recent_date[0:4]
    month = most_recent_date[4:6]
    day = most_recent_date[6:8]

    print(f'Found forecast data starting on {day}/{month}/{year}')


    ############# baseline ensemble ETo weekly totals
    # get week as int [0,3]
    week = int(day)//7

    print('Loading baseline ETo and computing ensemble mean of weekly sums')
    # load baseline ETo (local netCDF), pick week that matches start date 
    baseline_path = Path(
        f'{baseline_dir}/eto_climo_weekly_v2_'
        f'{baseline_model_name.get(model, model)}'
    )
    baseline_path = list(baseline_path.rglob(f'*_{month}.nc'))[0]
    base_ds = xr.open_dataset(baseline_path)

    # weekly sum of baseline ETo
    base_mov_sum_7day = base_ds['eto'].isel(week=week-1).rolling(
        L=7, min_periods=7).sum()

    base_mov_sum_7day_ens_mean = base_mov_sum_7day.mean(dim='member')

    week_sums = []
    for end_day in week_end_days:
        week_sums.append(base_mov_sum_7day_ens_mean.isel(L=end_day))

    base_weekly_sums = xr.concat(week_sums,'L')


    print('Computing daily forecasted ETo for all ensemble members')
    # get files using most recent valid date
    input_files = {}
    for v in needed_vars:
        var_files = download_data.get(model).get(v)
        input_files[v] = [f for f in var_files if most_recent_date in f.name][0]

    # load data in xarray/dask
    tmean = xr.open_dataset(input_files.get('tas'))
    hum_name = model_var_data.get(model).get('hum')
    hum_in = xr.open_dataset(input_files.get(hum_name))
    rad_name = model_var_data.get(model).get('rad')
    rd = xr.open_dataset(input_files.get(rad_name))
    u = xr.open_dataset(input_files.get('uas'))
    v = xr.open_dataset(input_files.get('vas'))

    # make elev, lat, and jday arrays
    ws = wind_spd_u_v(u['uas'],v['vas'])
    z = make_elev_like(tmean, elev_path)
    lat = make_lat_like(tmean)
    jday = make_jday_like(tmean)
    rad_type = model_var_data.get(model).get('rd_type')
    hum_type = model_var_data.get(model).get('hum_type')

    # ref ET data array inputs to tidy up before run
    ref_et_input = {
        'tmean':tmean['tas'],
        'hum_in':hum_in[hum_name],
        'rd':rd[rad_name],
        'ws':ws,
        'lat':lat,
        'z':z,
        'jday':jday
    }

    # chunk for dask workers
    for input_var, da in ref_et_input.items():
        da = da.isel(S=0) 
        ref_et_input[input_var] = _chunk_inputs(da,chunks) 

    tmean, hum_in, rd, ws, lat, z, jday = tuple(ref_et_input.values())

    ## apply ref ET with dask
    da_out = xr.Dataset() 
    stat_name = 'ETo'
    # eigth arg is 1 for ETo 2 for ETr
    da_out[stat_name] = xr.apply_ufunc(
        asce_refet, tmean, hum_in, rd, ws, lat, z, jday, 1, hum_type, rad_type,
        dask='parallelized',
        output_dtypes=[np.float32]
    )

    # compute and load ETo
    da_out.compute()

    print('Calculating forecasted weekly ETo sums and anomalies')
    # Calc weekly (7-day) moving total ETo
    mov_sum_7day = da_out['ETo'].rolling(L=7, min_periods=7).sum()

    week_sums = []
    for end_day in week_end_days:
        week_sums.append(mov_sum_7day.isel(L=end_day))

    weekly_sums = xr.concat(week_sums,'L')

    # anomalies, forecast minus base
    weekly_anomalies = weekly_sums - base_weekly_sums

    weekly_anomalies_ens_mean = weekly_anomalies.mean(dim='M')

    # save netCDFs
    if output_data_dir:
        output_data_dir = Path(output_data_dir)
    else: # assume folder name to save netCDFs
        output_data_dir = Path('anomaly_data')

    print(f'Writing output netCDFs to {output_data_dir}')

    output_data_dir.mkdir(parents=True, exist_ok=True)

    # save ensemble member and ensemble mean anomalies
    out_path = output_data_dir/f'{model}_{most_recent_date}.nc'
    weekly_anomalies.to_netcdf(str(out_path))

    out_path = output_data_dir/f'{model}_ensemble_mean_{most_recent_date}.nc'
    weekly_anomalies_ens_mean.to_netcdf(str(out_path))

    plot_path = f'{model}_{most_recent_date}.jpg'
    print(f'Saving plot to {plot_path}')
    #### Plot anomalies for 4 week forecast and save
    weekly_anomalies_ens_mean.plot(
        x="X", y="Y", col="L",col_wrap=2, robust=True,
        cbar_kwargs={"label": "ETo anomaly [mm/week]"}
    )
    f = plt.gcf()
    f.figsize=(10,8)
    plt.suptitle(
        f"{model.replace('_','-')}  starting {month}/{day}/{year}", y=1.02
    )
    plt.savefig(plot_path, bbox_inches='tight')
    
    print(f'Finished successfully')
