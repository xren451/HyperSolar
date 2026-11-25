"""Dataset adapters for the original grid16 V9 Ramp CSI/DDPM point task.

Targets remain physical irradiance, never electrical power. A raster cell
contains the mean of observed source nodes assigned to it, not interpolated
ground truth. Future Haurwitz geometry is calculated from time/coordinates.
"""
from pathlib import Path
import json
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

B = Path('/mnt/wangyishan_share/wangdongdong')
R = B / 'smf_registry_variants_20261105'
ROOT = B / 'smf_v9_four_datasets_20261111'
SPECS = {
    'CN-Solar': {
        'source': B / 'smf_utc_corrected_20261103/full_year_CN-SOLAR_20261103/data/CN-SOLAR_2021_full_common64_hourly.npz',
        'point': 866, 'radiative': ['Ch07', 'Ch08', 'SW_gbl'],
        'met': ['t2m', 'd2m', 'sp', 'u10', 'v10', 'u100', 'v100'],
        'target': 'archive grid-cell irradiance; not measured station GHI'},
    'NSRDB': {
        'source': R / 'data/NSRDB_2018_irradiance_fieldpoint.npz',
        'point': 0,
        'radiative': ['GHI', 'DNI', 'DHI', 'SolarZenith_deg', 'CloudType_code', 'ClearskyGHI'],
        'met': ['Temperature_C', 'RH_pct'],
        'target': 'NSRDB GHI resource product at sampled coordinates'},
    'SolarAlabama': {
        'source': R / 'data/SolarAlabama_2006_irradiance_fieldpoint.npz',
        'point': 0, 'radiative': ['CERES_all_sky_Wm2', 'CERES_clear_sky_Wm2'],
        'met': ['t2m', 'd2m', 'sp', 'u10', 'v10', 'u100', 'v100'],
        'target': 'CERES product proxy irradiance; not ground GHI or plant power'},
}


def prepare(dataset):
    import fcntl
    folder = ROOT/'cache'/dataset
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'.prepare.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        return _prepare(dataset)


def _prepare(dataset):
    if dataset == 'SolarCube':
        return prepare_cube_geometry()
    from scipy.sparse import csr_matrix
    import pandas as pd
    sys.path.insert(0, '/tmp/smf_pvlib_20261110')
    import pvlib
    out = ROOT / 'cache' / dataset
    if (out / 'READY.json').exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    spec = SPECS[dataset]
    z = np.load(spec['source'])
    names = list(z['feature_names'].astype(str))
    coords = z['coords'].astype(np.float32)
    idx = spec['point']
    assert idx < len(coords)
    if dataset == 'CN-Solar':
        assert 20 <= coords[idx, 0] <= 50 and 100 <= coords[idx, 1] <= 125
    stamps = np.asarray(z['timestamps']).astype('datetime64[s]')
    timestamp_shift_hours = 0.
    if dataset == 'NSRDB':
        # Archive timestamps are 7.5h later than the source solar-geometry
        # timestamps. Fixed by TRAIN-ONLY SZA/SPA audit (nine sampled nodes,
        # MAE < .04deg); labels/order/split indices are NOT moved or refitted.
        stamps = stamps-np.timedelta64(450, 'm')
        timestamp_shift_hours = -7.5
    assert np.all(np.diff(stamps).astype('timedelta64[s]').astype(int) == 3600)
    times = pd.DatetimeIndex(stamps).tz_localize('UTC')
    if dataset == 'NSRDB':
        check = np.arange(0, int(.7*len(times)), 37)
        reference_sza = z['features'][check, idx, names.index('SolarZenith_deg')]
        calculated = pvlib.solarposition.spa_python(times[check], float(coords[idx, 0]), float(coords[idx, 1]))['zenith'].to_numpy()
        clock_error = float(np.mean(abs(reference_sza-calculated)))
        if clock_error >= .1:
            raise ValueError(('NSRDB geometry/time audit failed', clock_error))
    length = len(times)
    cut = int(.7 * length)
    target = z['target_ghi'] if 'target_ghi' in z else z['target']
    target_mask = z['target_ghi_mask'] if 'target_ghi_mask' in z else z['target_mask']
    # Fixed geographic projection using coordinates only, not target values.
    latmax, latmin = float(coords[:, 0].max()), float(coords[:, 0].min())
    lonmin, lonmax = float(coords[:, 1].min()), float(coords[:, 1].max())
    position = np.array([(coords[idx, 1]-lonmin)/max(lonmax-lonmin, 1e-6),
                         (latmax-coords[idx, 0])/max(latmax-latmin, 1e-6)], np.float32)
    rr = np.rint((latmax-coords[:, 0])/max(latmax-latmin, 1e-6)*15).astype(int).clip(0, 15)
    cc = np.rint((coords[:, 1]-lonmin)/max(lonmax-lonmin, 1e-6)*15).astype(int).clip(0, 15)
    cell = rr*16+cc
    projection = csr_matrix((np.ones(len(coords), np.float32), (cell, np.arange(len(coords)))), shape=(256, len(coords)))
    gridlat, gridlon = np.meshgrid(np.linspace(latmax, latmin, 16), np.linspace(lonmin, lonmax, 16), indexing='ij')
    geometry_coords = np.concatenate([np.stack([gridlat.ravel(), gridlon.ravel()], -1), coords[idx:idx+1]], 0)
    def project(v):
        finite = np.isfinite(v)
        n, nodes, channels = v.shape
        numerator = projection @ np.where(finite, v, 0).transpose(1, 0, 2).reshape(nodes, -1)
        denominator = projection @ finite.astype(np.float32).transpose(1, 0, 2).reshape(nodes, -1)
        return np.where(denominator > 0, numerator/np.maximum(denominator, 1), np.nan).reshape(256, n, channels).transpose(1, 0, 2)
    arrays = {}
    for name, shape, dtype in [('features', (length, 256, len(names)), 'float32'),
                               ('field', (length, 256), 'float32'), ('field_mask', (length, 256), 'bool'),
                               ('clear', (length, 257), 'float32')]:
        arrays[name] = np.lib.format.open_memmap(out/(name+'.npy'), mode='w+', dtype=dtype, shape=shape)
    features = z['features']
    for lo in range(0, length, 168):
        hi = min(lo+168, length)
        arrays['features'][lo:hi] = project(features[lo:hi])
        yy = np.where(target_mask[lo:hi], target[lo:hi], np.nan)
        fg = project(yy[..., None])[..., 0]
        arrays['field'][lo:hi] = np.nan_to_num(fg)
        arrays['field_mask'][lo:hi] = np.isfinite(fg)
        st = times[lo:hi].repeat(257)
        lat = np.tile(geometry_coords[:, 0], hi-lo)
        lon = np.tile(geometry_coords[:, 1], hi-lo)
        spa = pvlib.solarposition.spa_python(st, lat, lon)
        cs = pvlib.clearsky.haurwitz(spa['apparent_zenith'])['ghi'].to_numpy().reshape(hi-lo, 257)
        arrays['clear'][lo:hi] = np.maximum(cs, 0).astype(np.float32)
        if lo % 1008 == 0:
            print('PREPARE', dataset, hi, '/', length, flush=True)
    del features
    valid = np.isfinite(arrays['features'][:cut])
    train = arrays['features'][:cut]
    mu = np.nanmean(train, axis=(0, 1))
    sd = np.maximum(np.nanstd(train, axis=(0, 1)), 1e-4)
    mu = np.nan_to_num(mu)
    sd = np.nan_to_num(sd, nan=1.)
    radiative_ix = [names.index(n) for n in spec['radiative']]
    met_ix = [names.index(n) for n in spec['met']]
    sat = np.lib.format.open_memmap(out/'sat.npy', mode='w+', dtype='float32', shape=(length, 6, 16, 16))
    nwp = np.lib.format.open_memmap(out/'nwp.npy', mode='w+', dtype='float32', shape=(length, 9, 16, 16))
    for lo in range(0, length, 168):
        hi = min(lo+168, length)
        raw = arrays['features'][lo:hi]
        vv = np.isfinite(raw)
        normalized = np.where(vv, (raw-mu)/sd, 0).astype(np.float32)
        sx = np.zeros((hi-lo, 256, 6), np.float32)
        sx[..., :len(radiative_ix)] = normalized[..., radiative_ix]
        if len(radiative_ix) <= 3:
            sx[..., 6-len(radiative_ix):] = vv[..., radiative_ix]
        wx = np.zeros((hi-lo, 256, 9), np.float32)
        wx[..., :len(met_ix)] = normalized[..., met_ix]
        wx[..., -1] = vv[..., met_ix].mean(-1)
        sat[lo:hi] = sx.transpose(0, 2, 1).reshape(-1, 6, 16, 16)
        nwp[lo:hi] = wx.transpose(0, 2, 1).reshape(-1, 9, 16, 16)
    point = np.asarray(target[:, idx], np.float32)
    pm = np.asarray(target_mask[:, idx], bool) & np.isfinite(point)
    pmean = float(point[:cut][pm[:cut]].mean())
    psd = max(float(point[:cut][pm[:cut]].std()), 1e-4)
    time_features = z['time_features'].astype(np.float32)
    if dataset == 'NSRDB':
        hour = np.asarray(times.hour+times.minute/60., np.float32)
        doy = np.asarray(times.dayofyear-1, np.float32)
        time_features = np.stack([np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
                                  np.sin(2*np.pi*doy/365), np.cos(2*np.pi*doy/365), hour/23, doy/364], -1).astype(np.float32)
    for name, value in [('point', np.nan_to_num(point)), ('point_mask', pm),
                        ('time', time_features), ('timestamps', stamps),
                        ('position', position), ('coords', coords), ('cell', cell)]:
        np.save(out/(name+'.npy'), value)
    for arr in list(arrays.values())+[sat, nwp]:
        arr.flush()
    info = {'dataset': dataset, 'source': str(spec['source']), 'point_index': idx,
            'point_name': str(z['site_names'][idx]), 'point_lat_lon': coords[idx].tolist(),
            'target': spec['target'], 'hours': length, 'cuts': [0, cut, int(.85*length), length],
            'feature_names': names, 'feature_mean': mu.tolist(), 'feature_std': sd.tolist(),
            'point_mean': pmean, 'point_std': psd, 'stat_fit_end_exclusive': cut,
            'radiative_channels': spec['radiative'], 'meteorological_channels': spec['met'],
            'source_modality_note': 'Observed radiative/product history is NOT falsely described as satellite imagery. Missing source variables are zero slots; meteorological availability channel explicit.',
            'field_grid': 16, 'field_target': 'mean of source nodes assigned by fixed nearest grid-cell coordinate; absent cells masked, no synthetic dense labels',
            'clear_sky': 'pvlib NREL SPA apparent zenith + Haurwitz, using UTC and coordinates only; no future met/irradiance',
            'history': 72, 'horizon': 24, 'stride': 3}
    info['timestamp_correction_hours'] = timestamp_shift_hours
    if dataset == 'NSRDB':
        info['training_only_sza_mae_after_clock_correction_deg'] = clock_error
    (out/'READY.json').write_text(json.dumps(info, indent=2))
    return out


INPUT_KEYS = ('sat', 'nwp', 'time_hist', 'time_future', 'point_hist', 'point_hist_mask', 'position')


class Windows(Dataset):
    def __new__(cls, dataset, split, requested_origin=None):
        if dataset == 'SolarCube':
            return CubeWindows(split, requested_origin)
        return super().__new__(cls)

    def __init__(self, dataset, split, requested_origin=None):
        self.path = ROOT/'cache'/dataset
        self.info = json.loads((self.path/'READY.json').read_text())
        self.arr = {n: np.load(self.path/(n+'.npy'), mmap_mode='r') for n in
                    ['sat', 'nwp', 'field', 'field_mask', 'point', 'point_mask', 'clear', 'time']}
        self.position = np.load(self.path/'position.npy')
        cuts = self.info['cuts']
        lo, hi = {'train': cuts[:2], 'val': cuts[1:3], 'test': cuts[2:4]}[split]
        candidates = list(range(lo+71, hi-24, 3))
        if requested_origin is not None:
            assert split == 'test' and lo+71 <= requested_origin < hi-24
            candidates.append(int(requested_origin))
        self.origins = sorted({t for t in candidates if self.arr['point_mask'][t+1:t+25].any()})
        self.split = split
        assert all(t-71 >= lo and t+24 < hi for t in self.origins)

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, i):
        t = self.origins[i]
        a = self.arr
        past = slice(t-71, t+1)
        future = slice(t+1, t+25)
        fy = np.array(a['field'][future])
        fm = np.array(a['field_mask'][future], bool)
        py = np.array(a['point'][future])
        pm = np.array(a['point_mask'][future], bool)
        cs = np.array(a['clear'][future])
        phm = np.array(a['point_mask'][past], bool)
        ph = np.where(phm, (a['point'][past]-self.info['point_mean'])/self.info['point_std'], 0).astype(np.float32)
        result = {'sat': np.array(a['sat'][past]), 'nwp': np.array(a['nwp'][past]),
                  'time_hist': np.array(a['time'][past]), 'time_future': np.array(a['time'][future]),
                  'point_hist': ph, 'point_hist_mask': phm, 'position': self.position.copy(),
                  'field_ghi': fy, 'field_mask': fm, 'point_ghi': py, 'point_mask': pm,
                  'field_cs': cs[:, :256], 'point_cs': cs[:, 256],
                  'field_csi': np.clip(fy/np.maximum(cs[:, :256], 1e-4), 0, 2.5),
                  'point_csi': np.clip(py/np.maximum(cs[:, 256], 1e-4), 0, 2.5),
                  'field_csi_mask': fm & (cs[:, :256] > 20),
                  'point_csi_mask': pm & (cs[:, 256] > 20), 'origin': np.array(t, np.int64),
                  'point_mean': np.array(self.info['point_mean'], np.float32),
                  'point_std': np.array(self.info['point_std'], np.float32)}
        return {k: torch.from_numpy(np.asarray(v).copy()) for k, v in result.items()}


def prepare_cube_geometry():
    """Known UTC/site geometry, independent of future satellite missingness."""
    import csv
    import pandas as pd
    sys.path.insert(0, '/tmp/smf_pvlib_20261110')
    import pvlib
    out = ROOT/'cache'/'SolarCube'
    out.mkdir(parents=True, exist_ok=True)
    sites = list(csv.DictReader(open('/mnt/wangyishan_share/paper_solar_datasets/solarcube/raw/SolarCube_sitelist.csv')))
    times = pd.date_range('2018-01-01', periods=8760, freq='h', tz='UTC')
    for site in sites:
        r = int(site['tile_id'])
        path = out/f'clear_region{r:02d}.npy'
        if path.exists() and (out/f'clear_region{r:02d}.ready').exists():
            continue
        lat_hi, lat_lo = float(site['lat_ulcnr']), float(site['lat_lrcnr'])
        lon_lo, lon_hi = float(site['lon_ulcnr']), float(site['lon_lrcnr'])
        lat = lat_hi-(np.arange(16)+.5)/16*(lat_hi-lat_lo)
        lon = lon_lo+(np.arange(16)+.5)/16*(lon_hi-lon_lo)
        la, lo = np.meshgrid(lat, lon, indexing='ij')
        la = np.r_[la.ravel(), float(site['lats'])]
        lo = np.r_[lo.ravel(), float(site['lons'])]
        result = np.lib.format.open_memmap(path, mode='w+', dtype='float32', shape=(8760, 257))
        for start in range(0, 8760, 168):
            end = min(start+168, 8760)
            position = pvlib.solarposition.spa_python(times[start:end].repeat(257), np.tile(la, end-start), np.tile(lo, end-start))
            result[start:end] = pvlib.clearsky.haurwitz(position['apparent_zenith'])['ghi'].to_numpy().reshape(end-start, 257)
        result.flush()
        (out/f'clear_region{r:02d}.ready').write_text('Known UTC and coordinates only; hourly origin timestamps; no future labels or weather.')
        print('PREPARE SolarCube geometry region', r, flush=True)
    return out


class CubeWindows(Dataset):
    def __init__(self, split, requested_origin=None):
        import csv
        from train_solarcube_v7_fieldpoint_csi import SolarCubeWindows
        self.base = SolarCubeWindows(
            R/'data/SolarCube_2018_full19_hourly', B/'smf_full_20261025/solarcube/prepared',
            '/mnt/wangyishan_share/paper_solar_datasets/solarcube/raw/SolarCube_sitelist.csv',
            R/'data/SolarCube_2018_era5_8x8_v7', split, stride=3)
        self.rows = self.base.rows
        self.path = ROOT/'cache'/'SolarCube'
        self.geometry = {r: np.load(self.path/f'clear_region{r:02d}.npy', mmap_mode='r') for r in range(1, 20)}
        self.times = np.arange(np.datetime64('2018-01-01T00:00:00'), np.datetime64('2019-01-01T00:00:00'), np.timedelta64(1, 'h'))
        self.sites = {int(v['tile_id']): v for v in csv.DictReader(open('/mnt/wangyishan_share/paper_solar_datasets/solarcube/raw/SolarCube_sitelist.csv'))}
        self.info = dict(dataset='SolarCube', target='actual station GHI and source SSR16 raster',
                         cuts=self.base.cuts, stat_fit_end_exclusive=self.base.cuts[1],
                         source='original full19 SolarCubeWindows; new known-geometry decoding',
                         clear_sky='pvlib SPA/Haurwitz at hourly timestamp, not future satellite SZA/missingness')
        self.split = split
        self.origins = [t for r, t in self.rows]
        self.requested_region = 9
        if requested_origin is not None and (8, requested_origin) not in self.rows:
            assert split == 'test' and self.base.cuts[2]+71 <= requested_origin < 8760-24
            self.rows.append((8, int(requested_origin)))
            self.origins.append(int(requested_origin))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        raw = self.base[i]
        r, t = self.rows[i]
        cs = torch.from_numpy(np.array(self.geometry[r+1][t+1:t+25]))
        raw['field_cs'], raw['point_cs'] = cs[:, :256], cs[:, 256]
        for scope in ['field', 'point']:
            y = raw[scope+'_ghi']
            mask = raw[scope+'_mask'].bool() & torch.isfinite(y)
            raw[scope+'_ghi'] = torch.nan_to_num(y)
            raw[scope+'_mask'] = mask
            clear = raw[scope+'_cs']
            raw[scope+'_csi'] = (torch.nan_to_num(y)/clear.clamp_min(1e-4)).clamp(0, 2.5)
            raw[scope+'_csi_mask'] = mask & (clear > 20)
        return raw
