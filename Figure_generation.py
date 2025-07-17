# %%
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@description:  Figure generation
@outline:
@Time    :   2025/06/25 20:51:24
@Author  :   	 Qiaomei Feng
'''
# %%
import numpy as np
import pandas as pd
import xarray as xr
import ultraplot as uplt
import scipy.stats as stats
import cartopy.crs as ccrs
import colormaps as cmaps
import pymannkendall as mk
import statsmodels.api as sm
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.ticker import LongitudeLocator, LatitudeLocator
# %%
# -------------------------------------Fig.c--------------------------------------
# * load data
df_lh_era_sf = pd.read_csv(r".\lh_era_ann_snowfree_region.csv", index_col=0)
df_lh_era_smd = pd.read_csv(r".\lh_era_ann_albpos_smneg_mk.csv", index_col=0)
df_lh_era_compl = pd.read_csv(r".\lh_era_ann_non_albpos_smneg_mk.csv", index_col=0)
df_lh_era_opst = pd.read_csv(r".\lh_era_ann_albneg_smpos_mk.csv", index_col=0)
# %%
df_et_era_smd = pd.read_csv(r".\et_era_ann_albpos_smneg_mk.csv")
df_et_glm_smd = pd.read_csv(r".\et_glm_ann_albpos_smneg_mk.csv")
# %%
# * lh trend
sf_result = mk.original_test(df_lh_era_sf["LH (W/m2)"] - df_lh_era_sf.iloc[0, 0])
smd_result = mk.original_test(df_lh_era_smd["LH (W/m2)"] - df_lh_era_smd.iloc[0, 0])
compl_result = mk.original_test(df_lh_era_compl['LH (W/m2)'] - df_lh_era_compl.iloc[0, 0])
opst_result = mk.original_test(df_lh_era_opst['LH (W/m2)'] - df_lh_era_opst.iloc[0, 0])
# %%
# * lh fit line
sf_fit = sf_result.slope * np.arange(0, 20) + sf_result.intercept
smd_fit = smd_result.slope * np.arange(0, 20) + smd_result.intercept
compl_fit = compl_result.slope * np.arange(0, 20) + compl_result.intercept
opst_fit = opst_result.slope * np.arange(0, 20) + opst_result.intercept
# %%
# * ET fit line
x = df_et_era_smd['ET (mm/yr)']
y = df_et_glm_smd['ET (mm/yr)']

slope, intercept, r, p, stderr = stats.linregress(x=x, y=y)
hat = slope * np.sort(x) + intercept
# %%
X = x
X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()

xmn = results.params[1]
xstd = results.bse[1]
results.summary()
# %%
# *plot
color_list = ['#324b4d', '#00ADB5', "#94b0b2", "#9696d8"]
label_list = ['Global snow-free regions', 'Albedo trend>0 & SM trend<0 regions',
              'Non-Albedo trend>0 & SM trend<0 regions', 'Albedo trend<0 & SM trend>0 regions']
df_list = [df_lh_era_sf, df_lh_era_smd, df_lh_era_compl, df_lh_era_opst]
fit_list = [sf_fit, smd_fit, compl_fit, opst_fit]

fig, ax = uplt.subplots(figsize=(7, 2.8), dpi=300)

for i in range(4):
    ax.plot(df_list[i].index, df_list[i]["LH (W/m2)"] - df_list[i].iloc[0, 0],
            label=label_list[i], color=color_list[i], lw=1.5)

    ax.plot(df_list[i].index, fit_list[i], ls='--', lw=0.8)

for i, result in enumerate([sf_result, smd_result, compl_result, opst_result]):
    ax.text(0.5,
            0.2 - i * 0.06,
            f"Slope = {result.slope:.3f} " + r"W/m$^2$ per year " +
            [r'$\it{p}$<0.001' if result.p < 0.001 else r'$\it{p}$<0.01' if result.p <
             0.01 else r'$\it{p}$<0.05' if result.p < 0.05 else r'$\it{p}$>0.05'][0],
            c=color_list[i], transform=ax.transAxes)

ax.legend(loc='ul', ncols=2, frameon=False)
ax.format(xlabel='Year', xlocator=4, ylabel=r'LH (W/m$^2$)', ylim=(-6, 4), ylocator=('maxn', 5), grid=False)
ax.spines[["top", "right"]].set_visible(False)

# inset
ix = ax.inset([0.08, 0.14, 0.18, 0.23], transform="axes", zoom=False)
ix.scatter(x=df_et_era_smd['ET (mm/yr)'], y=df_et_glm_smd['ET (mm/yr)'], s=23, c='#b6d3d5', alpha=0.7)
# fit line
ix.plot(
    np.sort(df_et_era_smd['ET (mm/yr)']),
    hat,
    lw=0.8,
    ls='--',
    c="#b6d3d5",
    alpha=0.7,
)

ix.text(
    x=0.56,
    y=0.08,
    s=r"$\it{R^2}$=0.819",
    c="#b6d3d5",
    transform=ix.transAxes,
)

ix.format(xlim=(593, 645), xlabel='ERA5-Land ET (mm/yr)', ylabel='GLEAM ET (mm/yr)', grid=False)

fig.savefig(r".\LH_albedo_pos_sm_neg6.png", dpi=500, bbox_inches='tight')
# %%
# -------------------------------------Fig.d,e--------------------------------------

# * load data
ds_et_era = xr.open_dataset(r".\et_era_trend_mk.nc").drop_vars(['band', 'spatial_ref'])
ds_sm_era = xr.open_dataset(r".\sm_era_trend_mk.nc").drop_vars(['band', 'spatial_ref'])
# %%
# ds_et_glm = xr.open_dataset(r".\et_glm_trend_mk.nc").drop_vars(['band', 'spatial_ref'])
# ds_sm_glm = xr.open_dataset(r".\sm_glm_trend_mk.nc").drop_vars(['band', 'spatial_ref'])
# %%
# *plot
ds_list = [ds_et_era, ds_sm_era]
cmap_list = [cmaps.prinsenvlag, "DryWet"]
label_list = ['ET trend (mm/yr)', r'SM trend(m$^3$/m$^3$)']

for ds, color, label in zip(ds_list, cmap_list, label_list):

    fig, ax = uplt.subplots(figsize=(3.5, 2.5), proj='robin', dpi=300)

    ax.format(
        lonlabels=False,
        latlabels=False,
        land=True,
        coast=False,
        ocean=False,
        landcolor=uplt.set_alpha("gray4", 0.6),
        coastcolor="k",
        coastlinewidth=0.25,
        grid=False,
    )

    m = ds.trend.plot.imshow(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=color,
        robust=True,
        center=0,
        extend="both",
        add_colorbar=False,
        zorder=1)

    # dot significant
    bool_data = ds.p < 0.05
    resampled_data = bool_data.coarsen(lat=4, lon=4, boundary='trim').max()

    dot_area = np.asarray(resampled_data)
    dot_lon, dot_lat = np.meshgrid(resampled_data.lon, resampled_data.lat)

    sc = ax.scatter(dot_lon[dot_area],
                    dot_lat[dot_area],
                    marker='.', color='gray7',
                    s=3.5, linewidths=0, zorder=2, transform=ccrs.PlateCarree())

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.2,
        color="0.4",
        linestyle=(0, (5, 5)),
        alpha=1,
        zorder=3,
    )

    gl.top_labels = True
    gl.bottom_labels = True
    gl.right_labels = True
    gl.left_labels = True

    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlocator = LongitudeLocator()
    gl.ylocator = LatitudeLocator()

    fig.colorbar(m, loc='b', label=label, length=0.5, width=0.08, pad=0.3, extend='both', extendsize=0.8)
    fig.format(linewidth=0.3, edgecolor='gray8')
