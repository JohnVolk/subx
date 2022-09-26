# -*- coding: utf-8 -*-

"""
Tools for calculation of reference ET from NOAA SubX drivers.
"""
import logging
import xarray as xr
import numpy as np
import pandas as pd
import datetime
from numba import vectorize, jit


model_var_data = {
    'EMC_GEFS':dict(
        tmean='tas', hum='tdps', rad='dswrf', uwind='uas', 
        vwind='vas', hum_type=2, rd_type=1),
    'GMAO_GEOS_V2p1':dict(
        tmean='tas', hum='tdps', rad='dswrf', uwind='uas', 
        vwind='vas', hum_type=2, rd_type=1),
    'ECCC_GEPS7':dict(
        tmean='tas', hum='tdps', rad='rad', uwind='uas', 
        vwind='vas', hum_type=2, rd_type=2) # rad is net rad
}

var_heights = {
    'tas':'2m',
    'tdps':'2m',
    'dswrf':'sfc',
    'rad':'sfc',
    'uas':'10m',
    'vas':'10m'
}

def make_elev_like(da, elev_nc_path):
    """
    Make elevation data array indexed in time like another data array.
    """
    elev = xr.open_dataset(elev_nc_path)
    elev = elev.rename_dims({'lat':'Y','lon':'X'})
    elev = elev.get('elevation')
    elev, dummy = xr.broadcast(elev, da.L) #L is lead time
    elev, dummy = xr.broadcast(elev, da.M) #L is lead time
    elev, dummy = xr.broadcast(elev, da.S) #L is lead time
    
    elev = elev.assign_coords({
        'X':da.X.values.astype(np.float32),
        'Y':da.Y.values.astype(np.float32),
    })

    return elev

def make_lat_like(da):
    # make lat array with same dims
    lat_da = da.Y.expand_dims(L=da.L, S=da.S, M=da.M, X=da.X, ) 
    return lat_da


def make_jday_like(da):
    # day of year 

    start_date = da.S.min().values
    l_vals = da.L.values
    # list of lead times (ns)
    lead_delta_ns = [
        datetime.timedelta(milliseconds=int(l)/1000000) for l in l_vals
    ]
    jdays = np.array([(pd.Timestamp(start_date) + l).dayofyear for l in lead_delta_ns])

    #########
#     j_dates = da.time.dt.dayofyear.values
    j_dates = np.broadcast_to(
        jdays[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis],
        (da.L.shape[0], da.Y.shape[0], da.X.shape[0], da.S.shape[0], da.M.shape[0])
    )

    jday = xr.DataArray(
        data = j_dates,
        dims = ["L", "Y", "X", "S", "M"],
        coords=dict(
            L = da.L,
            Y = da.Y,
            X = da.X,
            S = da.S,
            M = da.M

        )
    )
    
    return jday


@vectorize
def asce_refet(tmean, hum_in, rd, u, lat, z, jday, ref, hum_type, rd_type):
    """
    Compute daily ASCE standardized Reference ET from SubX drivers.

    Arguments:
        tmean (float): daily air temperature [K]
        hum_in (float): daily specific humidity (kg/kg) or dew point (K)
        rd (float):   daily downward shortwave radiation
                      at the surface [W m-2] or net radiation at the surface
        u (float):    daily wind speed [m s-1] at 10 meters 
        lat (float):  pixel latitude grid [dd]
        z (float):    pixel elevation [m]
        jday (float): Day of year
        ref (int):    1 for grass (ETo), 2 for alfalfa (ETr)
        hum_type (int): 1 for specific humidity (kg/kg), 2 for dew point (K)
        rd_type (int): 1 for shortwave, 2 for net radiation

    Returns:
        et (float):   array of ASCE ref. ET (mm/day)
    """

    # note some lines like print and del commands are commented out for numba

    # Conversions and adjustments
#    logging.debug('conversions')
    # Convert tair to [C]
    tmean = tmean - 273.15

    # Convert rd to [MJ m-2 d-1]
    rd = rd * 0.0864

    # Adjust wind speed from 10-m to 2-m
    u2 = u * ((4.87 / np.log(67.8 * 10. - 5.42)))

    # Convert tdew to [C]
    if hum_type == 2:
        hum_in = hum_in - 273.15

    #--------------------------------------
#    logging.debug('constants')
    # Constants

    # Solar constant [MJ m-2 min-1]
    Gsc = 0.082

    # Steffan-Boltzman Constant [MJ m-2 d-1 K-4]
    sig = 4.90 * 10. ** -9

    # Latitude in radians converted from latitude in decimal degrees
    phi = (np.pi * lat) / 180.

    # Barometric pressure in kPa as a function of elevation (z) in meters
    pressure = 101.3 * ((293. - 0.0065 * z) / 293.) ** 5.26

    # Latent heat of vaporization [MJ kg-1]
    lamda = 2.45

    # Psychrometric constant [kPa C-1]
    psyc = 0.00163 * (pressure / lamda)

    #--------------------------------------

    # Humidity and vapor pressure calculations
#    logging.debug('humidity')
    # Saturation vapor pressure [kPa]
    es = 0.6108 * np.exp((17.27 * tmean) / (tmean + 237.3))

    # Slope of the saturation vapor pressure/tair curve (kPaoC-1)
    delta_slope = ((4098. * es) / (tmean + 237.3) ** 2.)

    # Actual vapor pressure [kPa]
    if hum_type == 1: # from specific humidity
        ea = (hum_in * pressure) / (0.622 + 0.378 * hum_in)
    elif hum_type == 2: # from dew point
        ea = 0.6108 * np.exp((17.27 * hum_in) / (hum_in + 237.3))

    # Precipitable water
    w = 0.14 * ea * pressure + 2.1

    #--------------------------------------

    # Solar radiation calculations
#    logging.debug('solar')
    if rd_type == 1:
        # Correction for eccentricity of Earth's orbit around the sun
        dr = 1. + 0.033 * np.cos(((2. * np.pi) / 365.) * jday)

        # Declination of the sun above the celestial equator in radians
        delta = 0.40928 * np.sin(((2. * np.pi) / 365) * jday - 1.39435)

        # Sunrise hour angle [r] 
        #omega = np.arccos(-np.tan(phi) * np.tan(delta))
        #omega = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
        omega = -np.tan(phi) * np.tan(delta)
        omega = 1. if omega > 1 else omega
        omega = -1. if omega < -1 else omega
        omega = np.arccos(omega)

        # Angle of sun above horizon
        theta_24 = (omega * np.sin(phi) * np.sin(delta) +
                 np.cos(phi) * np.cos(delta) * np.sin(omega))
        theta_24 = 0.01 if theta_24 <= 0 else theta_24

        Ra = (24. / np.pi) * 4.92 * (dr) * theta_24

        # Clearness index for direct beam radiation (unitless)
        kb =\
            0.98*np.exp(((-0.00146*pressure)/theta_24)-0.075*(w/theta_24)**0.4)

        #del w, omega, theta_24

        # Transmissivity index for diffuse radiation (unitless)
        kd = 0.35 - 0.36 * kb

        # Clear sky total global solar radiation at the Earth's surface [MJ m-2 d-1]
        Rso = (kb + kd) * Ra
        #del Ra, kb, kd

        # Net solar radiation [MJ m-2 d-1]
        Rns = (1. - 0.23) * rd

        # Cloudiness function of rd and Rso
        f = 1.35 * rd / Rso - 0.35

        # Apparent "net" clear sky emmissivity
        net_emiss = 0.34 - 0.14 * np.sqrt(ea)

        # Net longwave radiation [MJ m-2 d-1]
        Rnl = f * net_emiss * sig * ((tmean + 273.15) ** 4)

        # Net radiation [MJ m-2 d-1]
        Rn = Rns - Rnl
    else:
        Rn = rd
        
    # Soil heat flux density (G; [MJ m-2 d-1])
    G = 0

    # Reference evapotranspiration calculation
#    logging.debug('ETo')

    # Determine long or short reference
    if ref == 1:
        Cn = 900.
        Cd = 0.34
    else:
        Cn = 1600.
        Cd = 0.38

    # ETo/ETr [mm/day]
    et_num = 0.408*delta_slope*(Rn-G)+psyc*(Cn/(tmean+273.))*u2*(es-ea)
    et_denom = delta_slope + psyc * (1. + Cd * u2)

    et = et_num / et_denom

    return et


