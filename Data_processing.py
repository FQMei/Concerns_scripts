#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@description:   data processing
@outline:
@Time    :   2025/06/19 08:58:58
@Author  :   	 Qiaomei Feng
'''
# %%
import numpy as np
import xarray as xr
from xarrayMannKendall import Mann_Kendall_test
from grid_area_utils import area_grid
# %%
# - 01 load data
# * mask data
# snowfree albedo trend>0 regions
mask = xr.open_dataset(r".\albedo_snowfree_trend_mask_1degree_land.nc")
sf_mask = xr.open_dataset(r".\albedo_snowfree_mask_1degree_land.nc")

mask = mask.where(sf_mask == 1)
mask
# %%
# * ET data
et_glm = xr.open_dataset(
    r"..\ET\Gleam\v4.2a\yearly\E_2000-2023_1deg.nc").sel(time=slice("2001-01-01", "2020-12-31"))

et_era = xr.open_dataset(
    r"..\ET\ERA5-Land\era5_2000-2022_mmyr_1deg.nc").sel(time=slice("2001-01-01", "2020-12-31"))
# %%
# * SM data
sm_glm = xr.open_dataset(r"..\ET\Gleam\v4.2a\yearly\SMs_2000-2024_1deg.nc").sel(time=slice("2001-01-01", "2020-12-31"))

sm_era = xr.open_dataset(
    r"..\ERA5\ERA5-Land monthly averaged data from 1950 to present\Volumetric soil water layer 1_1deg.nc").sel(valid_time=slice("2001-01-01", "2020-12-31"))
sm_era_yr = sm_era.groupby('valid_time.year').mean(dim='valid_time').rename({'year': 'time'})
# %%
# - 02 global pattern


def mk_trend(data, time, x, y, alpha=0.05, mk_modified=False):
    mk_test = Mann_Kendall_test(
        data,
        dim='time',
        alpha=alpha,
        method="theilslopes",
        MK_modified=mk_modified,
        coords_name={"time": time, "x": x, "y": y},
    )

    trend = mk_test.compute()
    trend = trend.rename({"x": x, "y": y})
    return trend


# %%
# spatial pattern
et_era_trend = mk_trend(et_era.e, time="time", x="lon", y="lat")
sm_era_trend = mk_trend(sm_era_yr.swvl1, time="time", x="lon", y="lat")
et_glm_trend = mk_trend(et_glm.E, time='time', x='lon', y='lat')
sm_glm_trend = mk_trend(sm_glm.SMs, time='time', x='lon', y='lat')
# %%
# mask
et_era_mask = et_era_trend.where(mask.mask == 1)
sm_era_mask = sm_era_trend.where(mask.mask == 1)
et_glm_mask = et_glm_trend.where(mask.mask == 1)
sm_glm_mask = sm_glm_trend.where(mask.mask == 1)
# %%
et_era_mask.to_netcdf(r".\et_era_trend_mk.nc")
sm_era_mask.to_netcdf(r".\sm_era_trend_mk.nc")
et_glm_mask.to_netcdf(r".\et_glm_trend_mk.nc")
sm_glm_mask.to_netcdf(r".\sm_glm_trend_mk.nc")
# %%
# - 03 annual mean LH changes for non-(SM trend<0 & albedo trend>0) regions

et_era_compl = et_era.where((~((mask.mask == 1) & (sm_era_trend.trend < 0))) & (sf_mask.mask == 1))

lh_era_compl = et_era_compl.e / 12.876
weights = np.cos(np.deg2rad(lh_era_compl.lat))
lh_era_ann_compl = lh_era_compl.weighted(weights).mean(dim=["lon", "lat"], skipna=True)
lh_era_ann_compl.to_dataframe().to_csv(r'.\lh_era_ann_complementary_mk.csv')
# %%
# - 04 annual mean LH changes for (SM trend>0 & albedo trend<0) regions

et_era_opst = et_era.where((mask.mask == 0) & (sm_era_trend.trend > 0))

lh_era_opst = et_era_opst.e / 12.876
weights = np.cos(np.deg2rad(lh_era_opst.lat))
lh_era_ann_opst = lh_era_opst.weighted(weights).mean(dim=["lon", "lat"], skipna=True)
lh_era_ann_opst.to_dataframe().to_csv(r'.\lh_era_ann_albneg_smpos_mk.csv')
# %%
# - 05 annual mean LH changes for (SM trend<0 & albedo trend>0) regions

et_era_smmask = et_era.where((mask.mask == 1) & (sm_era_trend.trend < 0))

lh_era_smmask = et_era_smmask.e / 12.876
weights = np.cos(np.deg2rad(lh_era_smmask.lat))
lh_era_ann = lh_era_smmask.weighted(weights).mean(dim=["lon", "lat"], skipna=True)
lh_era_ann.to_dataframe().to_csv(r'\lh_era_ann_smdecline_mk.csv')
# %%
# - 06 annual mean ET changes for (SM trend<0 & albedo trend>0) regions

et_glm_smmask = et_glm.where((mask.mask == 1) & (sm_era_trend.trend < 0))
# %%
weights = np.cos(np.deg2rad(et_era_smmask.lat))
et_era_ann = et_era_smmask.weighted(weights).mean(dim=["lon", "lat"], skipna=True)
et_glm_ann = et_glm_smmask.weighted(weights).mean(dim=['lon', "lat"], skipna=True)
# %%
et_era_ann.e.to_dataframe().to_csv(r'.\et_era_ann_albpos_smneg_mk.csv')
et_glm_ann.E.to_dataframe().to_csv(r'.\et_glm_ann_albpos_smneg_mk.csv')
# %%
# - 07 annual mean LH changes for snow free regions

et_era_mask = et_era.where(sf_mask.mask == 1)
lh_era_mask = et_era_mask.e / 12.876

weights = np.cos(np.deg2rad(lh_era_mask.lat))
lh_era_ann = lh_era_mask.weighted(weights).mean(dim=["lon", "lat"], skipna=True)
lh_era_ann.to_dataframe().to_csv(r".\lh_era_ann_snowfree_region.csv")
# %%
