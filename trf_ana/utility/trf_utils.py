"""
Utility helpers for TRF pipelines (Letswave I/O, envelopes, events, mapping,
trial extraction, and saving TRF kernels back to Letswave).

Drop this file in: code/utility/trf_utils.py
"""

import os, re
import numpy as np
import pandas as pd
from typing import List, Dict
from scipy.io import wavfile
from scipy.io import loadmat as _loadmat_sio
from scipy.io import savemat,loadmat
from scipy.signal import resample_poly
from scipy.io.matlab.mio5_params import mat_struct
from pathlib import Path

try:
    # mat73 is only needed for -v7.3 .mat files (HDF5-backed)
    from mat73 import loadmat as _loadmat_v73
except Exception:
    _loadmat_v73 = None


# ----------------------------- Basic helpers ------------------------------
def _squeeze_if_single_trial(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]  # [T x F]
    return arr
def trf_get_coef_lags(trf_obj):
    """Return (coef, lags_in_seconds) robustly across mTRFpy versions."""

    # ---- coefficients / weights / kernel ----
    coef = None
    for name in ['coef_', 'coef', 'w_', 'w', 'weights_', 'weights', 'kernel_', 'kernel']:
        if hasattr(trf_obj, name):
            arr = np.asarray(getattr(trf_obj, name))
            if arr.ndim in (2, 3) and arr.size > 0:
                coef = arr
                break
    if coef is None:
        # last resort: scan public attributes for an ndarray with plausible shape
        for name in dir(trf_obj):
            if name.startswith('_'):
                continue
            try:
                arr = np.asarray(getattr(trf_obj, name))
                if arr.ndim in (2, 3) and arr.size > 0:
                    coef = arr
                    break
            except Exception:
                pass
    if coef is None:
        raise AttributeError("TRF object: no coefficient-like ndarray attribute found.")

    # ---- try many lag attribute names (seconds or samples) ----
    lag_candidates = [
        'lags_', 'lags', 't',                 # seconds in many builds
        'lag_', 'lag', 'tau', 'taus',         # seconds in some builds
        'time_lags', 't_lags',                # seconds
        'lag_samples', 'lags_samples', 'lag_idx', 'idx_lags'  # samples (integers)
    ]
    lags = None
    for name in lag_candidates:
        if hasattr(trf_obj, name):
            val = getattr(trf_obj, name)
            # call if it's a method (rare)
            if callable(val):
                try:
                    val = val()
                except TypeError:
                    pass
            arr = np.asarray(val)
            if arr.size > 0:
                lags = arr
                break

    # Convert lag *samples* to seconds if needed
    if lags is not None and np.issubdtype(np.asarray(lags).dtype, np.integer):
        fs = getattr(trf_obj, 'fs', getattr(trf_obj, 'sr', None))
        if fs is not None:
            lags = np.asarray(lags, dtype=float) / float(fs)

    # ---- infer lags if still missing ----
    if lags is None or lags.size == 0:
        # number of lags from coef
        if coef.ndim == 3:
            # common shapes: [F, L, O], [O, L, F], etc. Pick the axis that matches L from metadata later,
            # otherwise assume the middle dimension is lags.
            L = coef.shape[1]
        else:  # 2D
            L = max(coef.shape)  # we will align below via metadata if available

        fs   = getattr(trf_obj, 'fs', getattr(trf_obj, 'sr', None))
        tmin = getattr(trf_obj, 'tmin', None)
        tmax = getattr(trf_obj, 'tmax', None)

        if (fs is not None) and (tmin is not None) and (tmax is not None):
            # Uniform grid from tmin..tmax with L points (≈ arange(tmin, tmax+1/fs, 1/fs))
            lags = np.linspace(float(tmin), float(tmax), num=int(L))
        elif fs is not None:
            # We only know fs and L: return 0..(L-1) samples converted to seconds
            lags = np.arange(int(L), dtype=float) / float(fs)
        else:
            # Last fallback: just 0..(L-1) in arbitrary units
            lags = np.arange(int(L), dtype=float)

    return coef, np.asarray(lags, dtype=float)


def kernels_as_nch_nlags(coef, lags):
    """
    Normalize coef into [nCh × nLags].
    Handles common shapes: [F × L × O], [L × O], [O × L].
    """
    import numpy as _np
    coef = _np.asarray(coef)
    L = None if lags is None else int(_np.asarray(lags).size)

    if coef.ndim == 3:
        F, Lc, O = coef.shape
        if F == 1:              # single feature
            return coef[0].T    # [O × L]
        # fallback: move lag axis to -1 then collapse to channels
        if L is not None and L in coef.shape:
            lag_ax = coef.shape.index(L)
            K = _np.moveaxis(coef, lag_ax, -1)   # [..., L]
            return K.reshape(-1, L)
        raise ValueError(f"Unrecognized 3D coef shape: {coef.shape}")

    if coef.ndim == 2:
        R, C = coef.shape
        if L is None:
            return coef if C >= R else coef.T
        if C == L:
            return coef
        if R == L:
            return coef.T
        return coef if abs(C - L) < abs(R - L) else coef.T

    raise ValueError(f"Unsupported coef ndim: {coef.ndim}")

def read_wav_float(path):
    fs, y = wavfile.read(path)               # y can be int16/int32/float
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.integer):
        y = y.astype(np.float64) / np.iinfo(y.dtype).max
    else:
        y = y.astype(np.float64)
    return y, fs


def normalize_z(x, axis=0):
    mu = np.mean(x, axis=axis, keepdims=True)
    sd = np.std(x, axis=axis, ddof=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (x - mu) / sd


def rms_envelope(x, fs, win_ms=10):
    """Sliding RMS: sqrt(moving average of x^2) with a rectangular window."""
    win = max(1, int(round(win_ms/1000 * fs)))
    filt = np.ones(win, dtype=np.float64) / win
    env2 = np.convolve(np.asarray(x, dtype=np.float64)**2, filt, mode='same')
    return np.sqrt(env2)


def short_code(s):
    """'S 83' -> 'S83'."""
    return re.sub(r'\s+', '', str(s).strip())


def find_stim_file(stim_root, fname):
    """Case-insensitive exact filename search anywhere under stim_root."""
    f_low = str(fname).lower()
    for root, _, files in os.walk(stim_root):
        for f in files:
            if f.lower() == f_low:
                return os.path.join(root, f)
    return None


# ---------------------- MATLAB struct -> Python dict ----------------------

def _is_mat_struct(x):
    """True if x is a scipy.io.loadmat MATLAB struct object."""
    # scipy returns objects with attribute _fieldnames
    return hasattr(x, '_fieldnames')

def _mat_to_dict(obj):
    """Recursively convert a MATLAB struct (and arrays of them) to dicts."""
    if _is_mat_struct(obj):
        d = {}
        for k in obj._fieldnames:
            d[k] = _mat_to_dict(getattr(obj, k))
        return d
    elif isinstance(obj, np.ndarray) and obj.dtype == object:
        # recursively convert each element
        return np.vectorize(_mat_to_dict, otypes=[object])(obj)
    else:
        return obj

def _as_header_dict(h):
    """Return a plain dict header regardless of loader (scipy or mat73)."""
    if isinstance(h, dict):
        return h
    if _is_mat_struct(h):
        return _mat_to_dict(h)
    # Some mat73 versions return a simple namespace-like object
    try:
        return dict(h)
    except Exception:
        return {'header': h}


# ----------------------------- Letswave I/O -------------------------------
# already present helper; leave as-is if you have it
def load_letswave_header(lw6_path):
    """
    Load a Letswave .lw6 header (MAT-file) and return the header struct as a dict-like object.
    Tries scipy.io.loadmat first; falls back to mat73 if v7.3.
    """
    try:
        from scipy.io import loadmat
        m = loadmat(lw6_path, squeeze_me=True, struct_as_record=False)
        hdr = m.get('header', None)
        if hdr is None:
            raise KeyError("No 'header' variable in lw6 file.")
        return hdr
    except NotImplementedError:
        import mat73
        m = mat73.loadmat(lw6_path)
        hdr = m.get('header', None)
        if hdr is None:
            raise KeyError("No 'header' variable in lw6 file.")
        return hdr

def _mat_struct_to_dict(s):
    """Best-effort conversion of MATLAB struct/object to Python dict."""
    if isinstance(s, dict):
        return s
    if hasattr(s, '__dict__'):
        return {k: getattr(s, k) for k in s.__dict__.keys()}
    # numpy.void from loadmat
    try:
        return {name: s[name] for name in s.dtype.names}
    except Exception:
        return s

def parse_raw_events(lw6_path: str) -> pd.DataFrame:
    """
    Read Letswave events from a **.lw6** file.
    Returns a pandas DataFrame with columns: ['code','latency','epoch'], sorted by latency.
    """
    hdr = load_letswave_header(lw6_path)
    # header.events can be a list/array of MATLAB structs
    evs = getattr(hdr, 'events', None) if hasattr(hdr, 'events') else hdr['events']
    # normalize to python list
    if isinstance(evs, np.ndarray):
        ev_list = list(evs.ravel())
    elif isinstance(evs, (list, tuple)):
        ev_list = list(evs)
    else:
        ev_list = [evs]

    out: List[Dict] = []
    for e in ev_list:
        d = _mat_struct_to_dict(e)
        # fields can be attributes, dict entries, or numpy scalars
        code    = d.get('code', '')
        latency = d.get('latency', np.nan)
        epoch   = d.get('epoch', np.nan)

        # make them python-native
        code_str = str(code)
        if code_str.startswith("b'") and code_str.endswith("'"):
            code_str = code_str[2:-1]
        try:
            latency = float(np.asarray(latency).astype(float))
        except Exception:
            latency = np.nan
        try:
            epoch = int(np.asarray(epoch).astype(int))
        except Exception:
            epoch = np.nan

        out.append({'code': code_str, 'latency': latency, 'epoch': epoch})

    df = pd.DataFrame(out)
    if 'latency' in df.columns:
        df = df.sort_values('latency').reset_index(drop=True)
    return df

def load_mat_any(path):
    """
    Load a MATLAB .mat file with either scipy (v7.2 and below) or mat73 (v7.3).
    Returns a plain Python object (dict or similar).
    """
    try:
        return _loadmat_sio(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        if _loadmat_v73 is None:
            raise RuntimeError(
                "This .mat file is v7.3 (HDF5). Install `mat73` (and h5py) "
                "or save the file as < v7.3."
            )
        return _loadmat_v73(path)

def find_subject_ids(raw_dir: Path):
    """Return ['DLS32_E', 'DLS46_E', ...] by scanning for *.lw6."""
    ids = []
    for f in raw_dir.iterdir():
        if f.is_file() and SUBJ_RE.match(f.name):
            ids.append(f.stem)            # 'DLS54_E'
    return sorted(ids)

def load_letswave_mat(mat_path, lw6_path=None):
    """
    Load LetsWave data (.mat + optional .lw6 header).

    Returns
    -------
    EEG : np.ndarray [nEp, nCh, nT]
    hdr : dict (Letswave header) or {}
    """
    m = load_mat_any(mat_path)

    # Numeric array lives under 'data' (float32) as [ep,ch,1,1,1,t] etc.
    if isinstance(m, dict) and 'data' in m:
        arr = np.asarray(m['data'])
    elif isinstance(m, dict) and 'lwdata' in m and isinstance(m['lwdata'], dict) and 'data' in m['lwdata']:
        arr = np.asarray(m['lwdata']['data'])
    else:
        # Sometimes scipy returns a struct with field 'data'
        try:
            arr = np.asarray(getattr(m, 'data'))
        except Exception as e:
            raise KeyError(f"'data' not found in {mat_path}") from e

    arr = np.squeeze(arr)
    if arr.ndim == 3:                      # [ep, ch, t]
        EEG = arr
    elif arr.ndim == 4:                    # [ep, ch, 1, t]
        EEG = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[-1])
    elif arr.ndim == 6:                    # [ep, ch, 1, 1, 1, t]
        EEG = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[-1])
    else:
        raise ValueError(f"Unexpected LetsWave data shape: {arr.shape}")

    # Optional header
    hdr = {}
    if lw6_path is not None and os.path.isfile(lw6_path):
        hfile = load_mat_any(lw6_path)
        if isinstance(hfile, dict) and 'header' in hfile:
            hdr_raw = hfile['header']
        else:
            hdr_raw = hfile
        hdr = _as_header_dict(hdr_raw)

    return EEG, hdr


def chan_labels_from_header(header):
    """Extract channel labels (np.array of strings) from a Letswave header."""
    h = _as_header_dict(header)
    labels = None
    if isinstance(h, dict) and 'chanlocs' in h and h['chanlocs'] is not None:
        cl = h['chanlocs']
        # chanlocs can be array of dicts with 'labels', or array of objects
        out = []
        for c in np.ravel(cl):
            if isinstance(c, dict) and 'labels' in c:
                out.append(str(c['labels']))
            elif _is_mat_struct(c) and hasattr(c, 'labels'):
                out.append(str(c.labels))
            else:
                out.append('Ch?')
        labels = np.array(out, dtype=object)
    return labels


def eeg_fs_from_header(header):
    """Return (fsEEG, xstep_seconds) from a Letswave header (dict or mat-struct)."""
    h = _as_header_dict(header)
    # xstep is seconds/sample
    if isinstance(h, dict) and 'xstep' in h:
        xstep = float(h['xstep'])
    elif _is_mat_struct(h) and hasattr(h, 'xstep'):
        xstep = float(h.xstep)
    else:
        raise KeyError("xstep not found in header.")
    fs = int(round(1.0 / xstep))
    return fs, xstep


def parse_raw_events(raw_mat_path):
    """Read Letswave RAW header and return sorted [{'code','latency'}]."""
    m = load_mat_any(raw_mat_path)
    # header under 'header' for most files
    hdr = m.get('header', m) if isinstance(m, dict) else m
    hdr = _as_header_dict(hdr)
    events = hdr.get('events', [])
    ev = []
    for e in np.ravel(events):
        if isinstance(e, dict):
            code = str(e.get('code', ''))
            lat  = float(e.get('latency', 0.0))
        else:
            # MATLAB struct-like
            code = str(getattr(e, 'code', ''))
            lat  = float(getattr(e, 'latency', 0.0))
        ev.append({'code': code, 'latency': lat})
    ev = sorted(ev, key=lambda d: d['latency'])
    return ev


def read_stim_mapping(xlsx_path):
    """
    Robustly read the 2-column stimulus table (filename, ID 101..240).
    Returns: dict id->filename (int->str)
    """
    T = pd.read_excel(xlsx_path, header=None)
    # First text-like col as names
    text_cols = T.select_dtypes(include=['object']).columns.tolist()
    name_col = text_cols[0] if text_cols else 0
    # A numeric column for IDs (or coerce another column)
    num_cols = T.select_dtypes(include=[np.number]).columns.tolist()
    id_col = num_cols[0] if num_cols else ([c for c in T.columns if c != name_col][0])
    ids = pd.to_numeric(T[id_col], errors='coerce')
    fn  = T[name_col].astype(str)
    keep = (~ids.isna()) & (ids.between(101, 240)) & (fn != '')
    return dict(zip(ids[keep].astype(int).tolist(), fn[keep].tolist()))


# --------------------- Trial extractor (epoch_stim_resp) -------------------

def epoch_stim_resp(ep_idx, EEG, nT, fsEEG, idx_sel, events, rawGenreIdx,
                    id2name, stim_root, genre_names, audio_delay_ms=130, envWin_ms=10):
    """
    Build (Stim, Resp, pretty_genre, sound_id, wav_path) for a single epoch.
    """
    rg = rawGenreIdx[ep_idx]

    # walk back to the nearest sound ID (101..240)
    sound_id = None
    k = rg - 1
    while k >= 0:
        code = str(events[k]['code']).strip()
        m = re.match(r'^\s*S\s*(\d+)', code)
        if m:
            num = int(m.group(1))
            if 101 <= num <= 240:
                sound_id = num
                break
        k -= 1
    if sound_id is None or sound_id not in id2name:
        return None

    wav_rel  = id2name[sound_id]
    wav_path = find_stim_file(stim_root, wav_rel)
    if wav_path is None:
        return None

    # envelope -> resample to EEG rate
    y, fsA = read_wav_float(wav_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    env = rms_envelope(y, fsA, win_ms=envWin_ms)

    # rational resample fsA -> fsEEG
    from math import gcd
    g = gcd(fsEEG, fsA)
    P, Q = fsEEG // g, fsA // g
    env_eeg = resample_poly(env, P, Q).astype(np.float64)

    offs  = int(round(audio_delay_ms/1000 * fsEEG))
    Tkeep = min(nT - offs, env_eeg.size)
    if Tkeep <= 200:
        return None

    stim = env_eeg[:Tkeep]
    X    = EEG[ep_idx, idx_sel, :]                 # [nSel x nT]
    resp = X[:, offs:offs+Tkeep].T                 # [Tkeep x nSel]

    gcode = str(events[rg]['code']).strip()
    gcode_norm = re.sub(r'\s+', ' ', gcode)
    pretty = genre_names.get(gcode_norm, gcode_norm)
    return stim, resp, pretty, sound_id, wav_path


# -------------------- Write TRF kernels back to Letswave -------------------

def _chanlocs_struct(labels):
    """Build a MATLAB-like struct array with field 'labels' for Letswave."""
    cl = np.empty((len(labels),), dtype=object)
    for i, lb in enumerate(labels):
        cl[i] = {'labels': str(lb)}
    return cl

def _lw_header(name, datasize, fs, tmin_ms, labels):
    """
    Minimal Letswave header compatible with Letswave 7:
      - datasize: [ep, ch, y, z, ???, x]
      - xstep = 1/fs (lag step)
      - xstart = tmin (seconds)
      - chanlocs with labels
    """
    header = {
        'name': name,
        'datasize': np.array(datasize, dtype=np.int32),
        'xstep': 1.0/float(fs),
        'xstart': float(tmin_ms)/1000.0,
        'ystep': 1.0, 'zstep': 1.0, 'ystart': 0.0, 'zstart': 0.0,
        'events': np.empty((0,)),
        'chanlocs': _chanlocs_struct(labels),
        'filetype': 'time_series',
    }
    return header

def _clone_mat_struct(src):
    """Shallow-clone a MATLAB mat_struct so nested struct arrays stay structs."""
    dst = mat_struct()
    for f in src._fieldnames:
        setattr(dst, f, getattr(src, f))
    return dst

def _set_scalar_field(h, name, value):
    # MATLAB likes double scalars; datasize should be row 1×6 double.
    if name == 'datasize':
        arr = np.asarray(value, dtype=np.float64).reshape(1, 6)
        setattr(h, name, arr)
    else:
        setattr(h, name, float(value) if isinstance(value, (int, np.integer, np.floating)) else value)

def _trf_stack_to_letswave_tensor(W_list, tLags_ms):
    tLags_ms = np.asarray(tLags_ms, dtype=float)
    nLags = int(tLags_ms.size)
    W0 = np.asarray(W_list[0])
    if W0.shape[0] == nLags:
        nCh = W0.shape[1]; orient = "rows"
    elif W0.shape[1] == nLags:
        nCh = W0.shape[0]; orient = "cols"
    else:
        raise ValueError(f"First TRF shape {W0.shape} incompatible with nLags={nLags}.")
    nEp = len(W_list)
    data = np.zeros((nEp, nCh, 1, 1, 1, nLags), dtype=np.float32)
    for i, W in enumerate(W_list):
        W = np.asarray(W)
        if orient == "rows":
            if W.shape != (nLags, nCh): raise ValueError
            data[i, :, 0, 0, 0, :] = W.T
        else:
            if W.shape != (nCh, nLags): raise ValueError
            data[i, :, 0, 0, 0, :] = W
    return data


def _set_header_field(hdr, name, value, is_datasize=False):
    """
    Works for numpy.void (MATLAB struct) and for mat_struct (attribute style).
    """
    # MATLAB-friendly formats
    if is_datasize:
        value = np.asarray(value, dtype=np.float64).reshape(1, 6)  # 1x6 double row
    elif isinstance(value, (np.integer, int, np.floating, float)):
        value = float(value)  # scalar double
    # Try attribute style (mat_struct), else field indexing (numpy.void)
    try:
        getattr(hdr, name)  # exists?
        setattr(hdr, name, value)
    except Exception:
        try:
            hdr[name] = value
        except Exception:
            # Some lw6 write name as cell-char, but SciPy will handle str -> char
            hdr[name] = value
            
def _short_code(name: str) -> str:
    return 'TRF_' + name.replace(' ', '_').replace('/', '_').replace('\\', '_')

def save_trf_trialwise_letswave_pergenre(
    out_base: str,
    W_by_genre: dict[str, list[np.ndarray]],
    tLags_ms: np.ndarray,
    template_lw6_path: str,
    update_time_axis: bool = True,
):
    """
    Save one Letswave dataset PER GENRE using a template .lw6 header.

    Writes
    ------
    For each genre g:
      out_base + f" TRF <genre with spaces>.lw6"  -> header struct
      out_base + f" TRF <genre with spaces>.mat"  -> data tensor only
    """
    # Load template as 1x1 MATLAB struct array
    mat = loadmat(template_lw6_path, struct_as_record=False, squeeze_me=False)
    if 'header' not in mat:
        raise ValueError(f"Template '{template_lw6_path}' has no 'header'.")
    header_arr = mat['header']              # shape (1,1), element is numpy.void or mat_struct
    hdr = header_arr[0, 0]

    tLags_ms = np.asarray(tLags_ms, dtype=float)
    nLags = int(tLags_ms.size)
    xstart_s = float(tLags_ms[0] / 1000.0)
    xstep_s  = float((tLags_ms[1] - tLags_ms[0]) / 1000.0) if nLags > 1 else 1.0

    def _code(g):
        # keep spaces; replace underscores with spaces; avoid slashes in filenames
        s = str(g).replace('_', ' ').replace('/', '-').replace('\\', '-')
        s = 'TRF ' + s
        return ' '.join(s.split())  # collapse repeats

    for gname, W_list in W_by_genre.items():
        if not W_list:  # empty genre
            continue

        data = _trf_stack_to_letswave_tensor(W_list, tLags_ms)
        nEp, nCh = int(data.shape[0]), int(data.shape[1])

        # Modify only the scalar fields on a COPY of the 1x1 struct array
        header_copy = header_arr.copy()      # keeps struct type/shape
        hdr_edit = header_copy[0, 0]

        # NOTE: space instead of underscore in filenames
        base = f"{out_base}/trial_wise {_code(gname)}"

        _set_header_field(hdr_edit, 'name', os.path.basename(base))
        _set_header_field(hdr_edit, 'datasize', [nEp, nCh, 1, 1, 1, nLags], is_datasize=True)
        if update_time_axis:
            _set_header_field(hdr_edit, 'xstart', xstart_s)
            _set_header_field(hdr_edit, 'xstep',  xstep_s)

        # Save: header as original 1x1 struct; data numeric only
        savemat(base + '.lw6', {'header': header_copy[0, 0]}, do_compression=False, long_field_names=True)
        savemat(base + '.mat', {'data': data}, do_compression=False)
        
def save_trf_pergenre_letswave(out_base, W, tLags_ms, fs, genre_name, template_lw6_path, update_time_axis=True):
    """Save a single-epoch Letswave file for one genre using a template .lw6 header.
    .lw6 gets only the header (struct); .mat gets only the numeric data.
    Data stored as [1, nCh, 1, 1, 1, nLags] with lag on the last axis.
    """
    import os
    import numpy as _np
    from scipy.io import loadmat, savemat

    W = _np.asarray(W)
    tLags_ms = _np.asarray(tLags_ms, dtype=float)
    if W.ndim != 2:
        raise ValueError("W must be 2D (nLags x nCh) or (nCh x nLags)." )

    # Orient to [nCh x nLags]
    if W.shape[1] == tLags_ms.size:
        W_ch_x_lag = W
    elif W.shape[0] == tLags_ms.size:
        W_ch_x_lag = W.T
    else:
        raise ValueError(f"W shape {W.shape} incompatible with tLags length {tLags_ms.size}")

    nCh, nLags = W_ch_x_lag.shape
    data = _np.zeros((1, nCh, 1, 1, 1, nLags), dtype=_np.float32)
    data[0, :, 0, 0, 0, :] = W_ch_x_lag

    # Load template header (1x1 struct array)
    mat = loadmat(template_lw6_path, struct_as_record=False, squeeze_me=False)
    if 'header' not in mat:
        raise ValueError(f"Template '{template_lw6_path}' has no 'header'.")
    header_arr = mat['header']
    hdr_copy = header_arr.copy()
    hdr = hdr_copy[0, 0]

    # Build a spacified base and display name
    safe_genre = str(genre_name).replace('_', ' ').replace('/', '-').replace('\\', '-')
    base = f"{out_base}/genre_pooled TRF {safe_genre}"          # filenames with spaces
    name = os.path.basename(base)                  # dataset name with spaces

    # Basic fields
    try: setattr(hdr, 'name', name)
    except Exception: hdr['name'] = name

    datasize = _np.asarray([1, nCh, 1, 1, 1, nLags], dtype=_np.float64).reshape(1, 6)
    try: setattr(hdr, 'datasize', datasize)
    except Exception: hdr['datasize'] = datasize

    if update_time_axis:
        xstart_s = float(tLags_ms[0] / 1000.0)
        xstep_s  = float((tLags_ms[1] - tLags_ms[0]) / 1000.0) if nLags > 1 else 1.0/float(fs)
        try:
            setattr(hdr, 'xstart', xstart_s); setattr(hdr, 'xstep',  xstep_s)
        except Exception:
            hdr['xstart'] = xstart_s; hdr['xstep']  = xstep_s

    # Save header struct and data (filenames have spaces)
    savemat(base + '.lw6', {'header': hdr}, do_compression=False, long_field_names=True)
    savemat(base + '.mat', {'data': data}, do_compression=False)
    
    
# -------------------- Write reconstructed EEG back to Letswave -------------------

def _stack_trials_to_letswave_ts(Y_list):
    """
    Y_list: list of arrays [T_i x nCh]; T_i may differ by <= a few samples.
    -> data: [nEp, nCh, 1, 1, 1, T_min]
    """
    Ys = [np.asarray(y) for y in Y_list]
    # ensure 2D and collect nCh, T
    for i, y in enumerate(Ys):
        if y.ndim != 2:
            raise ValueError(f"Recon[{i}] must be 2D [T x nCh], got {y.ndim}D.")
    nChs = {y.shape[1] for y in Ys}
    if len(nChs) != 1:
        raise ValueError(f"Inconsistent nCh across trials: {sorted(nChs)}")
    nCh = nChs.pop()
    T_min = min(y.shape[0] for y in Ys)

    # truncate to T_min and stack into Letswave layout
    data = np.zeros((len(Ys), nCh, 1, 1, 1, T_min), dtype=np.float32)
    for i, y in enumerate(Ys):
        y_use = y[:T_min, :]      # truncate if needed
        data[i, :, 0, 0, 0, :] = y_use.T
    return data


def save_recon_trialwise_letswave_pergenre(
    out_base: str,
    Yhat_by_genre: dict[str, list[np.ndarray]],   # {genre: [T x nCh] per trial}
    fs: float,
    template_lw6_path: str,
    prefix: str = "recon trial_wise",
):
    """
    Save per-genre trial-wise EEG reconstructions (forward model).
    Each genre gets one dataset: data [nEp, nCh, 1,1,1, T].
    """
    mat = loadmat(template_lw6_path, struct_as_record=False, squeeze_me=False)
    if 'header' not in mat:
        raise ValueError(f"Template '{template_lw6_path}' has no 'header'.")
    header_arr = mat['header']          # (1,1)
    for gname, Y_list in Yhat_by_genre.items():
        if not Y_list:
            continue
        data = _stack_trials_to_letswave_ts(Y_list)
        nEp, nCh, _, _, _, T = data.shape

        hdr_copy = header_arr.copy()
        hdr = hdr_copy[0, 0]

        safe_genre = str(gname).replace('_', ' ').replace('/', '-').replace('\\', '-')
        base = f"{out_base}/{prefix} TRF {safe_genre}"
        name = os.path.basename(base)

        # header fields
        try: setattr(hdr, 'name', name)
        except Exception: hdr['name'] = name
        try: setattr(hdr, 'datasize', np.asarray([nEp, nCh, 1, 1, 1, T], float).reshape(1, 6))
        except Exception: hdr['datasize'] = np.asarray([nEp, nCh, 1, 1, 1, T], float).reshape(1, 6)
        # time axis (seconds)
        try:
            setattr(hdr, 'xstart', 0.0); setattr(hdr, 'xstep', 1.0/float(fs))
        except Exception:
            hdr['xstart'] = 0.0; hdr['xstep'] = 1.0/float(fs)

        savemat(base + '.lw6', {'header': hdr}, do_compression=False, long_field_names=True)
        savemat(base + '.mat', {'data': data}, do_compression=False)


def save_recon_pergenre_letswave(
    out_base: str,
    Yhat_by_genre: dict[str, list[np.ndarray]],   # {genre: [T x nCh] per trial}
    fs: float,
    template_lw6_path: str,
    prefix: str = "recon genre_pooled",
):
    """Same as above but different prefix in names."""
    return save_recon_trialwise_letswave_pergenre(
        out_base, Yhat_by_genre, fs, template_lw6_path, prefix=prefix
    )
