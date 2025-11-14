"""
Core entry points to run:
  1) TRF (trial-wise): one TRF per epoch with time-wise CV inside each epoch.
  2) TRF (per-genre pooled): one TRF per genre using trial-wise CV (no concatenation).

Requirements:
  pip install mtrf scipy numpy pandas soundfile openpyxl
"""

import sys,os
import numpy as np
from mtrf.model import TRF
from mtrf.stats import nested_crossval
import warnings
from scipy.io.wavfile import WavFileWarning   # <-- add this
warnings.filterwarnings("ignore", category=WavFileWarning)
from mtrf.model import TRF
import numpy as np
import glob
from utility.trf_utils import (
    load_letswave_mat, chan_labels_from_header, eeg_fs_from_header,
    parse_raw_events, read_stim_mapping, epoch_stim_resp, read_wav_float,
    normalize_z, short_code,
    save_trf_trialwise_letswave_pergenre, save_trf_pergenre_letswave,
    trf_get_coef_lags, kernels_as_nch_nlags,_squeeze_if_single_trial,save_recon_pergenre_letswave,
    save_recon_trialwise_letswave_pergenre
)

# ---------------------------- User settings -------------------------------


template_lw6 = "../data/preprocessed/DLS14_E.lw6"
genre_codes = ['S 81','S 82','S 83','S 84','S 85','S 86','S 87']
genre_names = {
    'S 81':'Pop-inst','S 82':'Pop-voc','S 83':'Blues-inst',
    'S 84':'Blues-voc','S 85':'Metal-inst','S 86':'Metal-voc','S 87':'Control'
}


audio_delay_ms  = 130
envWin_ms     = 10
alpha = 1e-1
lambda_grid   = np.logspace(-3, 3, 20)
tmin_ms       = 0
tmax_ms       = 600
Dir           = 1
kFolds        = 2
wanted_ch = np.array(['F3','F4','C3','C4','P3','P4','O1','O2'])
tmin_s, tmax_s = tmin_ms/1000.0, tmax_ms/1000.0


# -------------------------- TRF: trial-wise (per epoch) ---------------------
def trf_trialwise():
    # out_dir = os.path.join(out_root, 'trialwise')
    # os.makedirs(out_dir, exist_ok=True)

    # ---- small helpers ----
    def _contig_folds(T, k):
        lens = [T // k] * k
        lens[-1] = T - sum(lens[:-1])
        idx = np.arange(T)
        blocks, start = [], 0
        for L in lens:
            blocks.append(idx[start:start+L])
            start += L
        return blocks

    def _as_3d_trial(x):
        """[T] or [T x F] -> [1 x T x F]"""
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[None, :, None]
        elif x.ndim == 2:
            x = x[None, :, :]
        return x

    # before epoch loop
    recon_by_genre = {v: [] for v in genre_names.values()}
    all_w_by_genre = {v: [] for v in genre_names.values()}
    lag_ref = None
    lag_ref = None

    for ep in range(nEp):
        pack = epoch_stim_resp(ep, EEG, nT, fsEEG, idx_sel, events, rawGenreIdx,
                               id2name, stim_root, genre_names,
                               audio_delay_ms=audio_delay_ms, envWin_ms=envWin_ms)
        if pack is None:
            print(f'ep {ep+1:03d}: skipped (no audio/mismatch)')
            continue

        stim, resp, pretty, sid, _ = pack
        stim_z = normalize_z(stim[:, None], axis=0).ravel()   # [T]
        resp_z = normalize_z(resp, axis=0)                    # [T x nChSel]
        Tkeep, nChSel = resp_z.shape
        folds = _contig_folds(Tkeep, kFolds)
        # ---- CV over lambda per channel (time-wise) ----
        best_alpha = np.empty(nChSel, dtype=float)
        
        # No cross-validation being performed
        # for ch in range(nChSel):
        #     r_mean_per_lambda = []
        #     for L in lambda_grid:
        #         r_folds = []
        #         for test_idx in folds:
        #             train_idx = np.setdiff1d(np.arange(Tkeep), test_idx, assume_unique=True)

        #             Xtr = _as_3d_trial(stim_z[train_idx])        # [1 x Tr x 1]
        #             Ytr = _as_3d_trial(resp_z[train_idx, [ch]])  # [1 x Tr x 1]
        #             Xte = _as_3d_trial(stim_z[test_idx])         # [1 x Te x 1]
        #             Yte = _as_3d_trial(resp_z[test_idx, [ch]])   # [1 x Te x 1]

        #             trf_ch = TRF(direction=Dir, kind='single', method='ridge')
        #             trf_ch.train(_squeeze_if_single_trial(Xtr), _squeeze_if_single_trial(Ytr), fsEEG, tmin_s, tmax_s, float(L))
        #             _, r = trf_ch.predict(_squeeze_if_single_trial(Xte), _squeeze_if_single_trial(Yte))
        #             r_folds.append(float(np.asarray(r).squeeze()))

        #         r_mean_per_lambda.append(np.mean(r_folds))

        #     best_alpha[ch] = float(lambda_grid[int(np.argmax(r_mean_per_lambda))])
        
        best_alpha = alpha*np.ones(nChSel, dtype=float)
        print(f'ep {ep+1:03d}: OK ({pretty}, ID {sid})')
        # ---- final fit on full trial with best lambda per ch ----
        W = []
        Yhat_trial = np.zeros_like(resp_z)  # [T x nChSel]
        for ch in range(nChSel):
            X = _as_3d_trial(stim_z)              # [1 x T x 1]
            Y = _as_3d_trial(resp_z[:, [ch]])     # [1 x T x 1]
            trf_ch = TRF(direction=Dir, kind='single', method='ridge')
            trf_ch.train(_squeeze_if_single_trial(X), _squeeze_if_single_trial(Y), fsEEG, tmin_s, tmax_s, float(best_alpha[ch]))
            coef, lag_ref = trf_get_coef_lags(trf_ch)
            W.append(np.asarray(coef).ravel())
            Yhat, _ = trf_ch.predict(_squeeze_if_single_trial(X), _squeeze_if_single_trial(Y))
            Yhat_trial[:, ch] = np.asarray(Yhat).squeeze()                 # seconds
        
        W = np.stack(W, axis=1)                   # [nLags x nChSel]
        all_w_by_genre[pretty].append(W)
        recon_by_genre[pretty].append(Yhat_trial) 

    tLags_ms = np.asarray(lag_ref) * 1000.0
    save_trf_trialwise_letswave_pergenre(out_root,
                                     all_w_by_genre,
                                     tLags_ms,
                                     template_lw6_path=template_lw6,
                                     update_time_axis=True)    
    save_recon_trialwise_letswave_pergenre(
    out_root, recon_by_genre, fsEEG, template_lw6, prefix="recon trial_wise"
)
    print(f'Trial-wise TRFs saved to: {out_root}.lw6/.mat')

# -------------------------- TRF: pooled per-genre (LOTO) --------------------

def trf_per_genre():
    """
    Fit TRFs (ridge, encoding) with trial-wise CV:
      1) per-subgenre  (Pop-inst, Pop-voc, Blues-inst, Blues-voc, Metal-inst, Metal-voc, Control)
      2) per-main-genre (Pop, Blues, Metal)   <-- NEW

    Uses nested_crossval to choose λ per-channel, then fits a single TRF
    on all trials of that group. Saves kernels and per-trial reconstructions
    back to Letswave (one dataset per group).
    """
    # ---------- small helpers ----------
    def _as_lists_same_length(S_list, R_list):
        """Make all trials equal length (truncate to shortest)."""
        S_list = [np.asarray(s).squeeze() for s in S_list]      # [T]
        R_list = [np.asarray(r)            for r in R_list]      # [T x nSel]
        min_T = min(min(s.shape[0] for s in S_list),
                    min(r.shape[0] for r in R_list))
        S_list = [s[:min_T] for s in S_list]
        R_list = [r[:min_T, ...] for r in R_list]
        return S_list, R_list

    def _fit_group_and_save(gname, S_list, R_list, recon_bucket):
        """Choose λ per channel via nested CV, fit once on all trials, save."""
        if len(S_list) == 0:
            print(f'{gname}: no trials.')
            return

        # equalize length
        S_list, R_list = _as_lists_same_length(S_list, R_list)

        # lists for nested_crossval: X is [T x 1], Y is [T x nSel]
        X_list = [s.reshape(-1, 1).astype(float) for s in S_list]
        Y_list_full = [r.astype(float) for r in R_list]          # [T x nSel]
        nSel = Y_list_full[0].shape[1]

        # --- λ selection per channel
        best_alpha = alpha*np.ones(nSel, dtype=float)
        # for ch in range(nSel):
        #     Y_list_ch = [r[:, [ch]] for r in Y_list_full]        # [T x 1]
        #     trf = TRF(direction=1, kind='single', method='ridge')
        #     cv_out = nested_crossval(trf, X_list, Y_list_ch,
        #                              fsEEG, tmin_s, tmax_s, lambda_grid)
        #     if isinstance(cv_out, tuple):
        #         r_unbiased, alpha = cv_out
        #     else:
        #         r_unbiased = cv_out.get('r', cv_out.get('score'))
        #         alpha = cv_out.get('alpha', cv_out.get('regularization'))
        #     best_alpha[ch] = float(np.asarray(alpha).ravel()[0])

        # --- final fit per channel with its best λ
        kernels = []
        Yhat_trials = [np.zeros((X_list[0].shape[0], 1)) for _ in range(len(X_list))]
        for ch in range(nSel):
            Y_list_ch = [r[:, [ch]] for r in Y_list_full]
            trf = TRF(direction=1, kind='single', method='ridge')
            trf.train(X_list, Y_list_ch, fsEEG, tmin_s, tmax_s, float(best_alpha[ch]))
            # recon per trial, stack across channels
            for t_idx, Xseg in enumerate(X_list):
                Yhat_ch, _ = trf.predict(Xseg, Y_list_ch[t_idx])
                Yhat_ch = np.asarray(Yhat_ch).reshape(-1, 1)
                if ch == 0:
                    Yhat_trials[t_idx] = Yhat_ch
                else:
                    Yhat_trials[t_idx] = np.hstack([Yhat_trials[t_idx], Yhat_ch])
            coef, lags = trf_get_coef_lags(trf)
            K = kernels_as_nch_nlags(coef, lags)     # -> [nCh x nLags]
            kernels.append(K[0])                     # current channel's [nLags]

        # stack per-channel -> [nLags x nSel]
        W = np.stack(kernels, axis=1)
        lags_ms = (np.asarray(lags) * 1000.0).astype(float)

        # collect recon for this group, then save
        recon_bucket[gname].extend(Yhat_trials)
        save_trf_pergenre_letswave(
            out_root, W, lags_ms, fsEEG, gname,
            template_lw6_path=template_lw6, update_time_axis=True
        )
        print(f"{gname}: done. best λ median={np.median(best_alpha):.3g} "
              f"(per-ch range {best_alpha.min():.3g}..{best_alpha.max():.3g})")

    # ---------- gather trials into two kinds of buckets ----------
    # subgenres (existing)
    buckets_sub = {v: {'stim': [], 'resp': []} for v in genre_names.values()}
    # main genres (NEW)
    main_keys = ['Pop', 'Blues', 'Metal']
    buckets_main = {k: {'stim': [], 'resp': []} for k in main_keys}

    # z-scored per trial for stable CV
    for ep in range(nEp):
        pack = epoch_stim_resp(ep, EEG, nT, fsEEG, idx_sel, events, rawGenreIdx,
                               id2name, stim_root, genre_names,
                               audio_delay_ms=audio_delay_ms, envWin_ms=envWin_ms)
        if pack is None:
            continue
        stim, resp, pretty, _, _ = pack
        stim_z = normalize_z(stim, axis=0).ravel()
        resp_z = normalize_z(resp, axis=0)

        # fill subgenre bucket
        buckets_sub[pretty]['stim'].append(stim_z)
        buckets_sub[pretty]['resp'].append(resp_z)

        # fill main-genre bucket (ignore Control)
        main = pretty.split('-')[0]
        if main in buckets_main:
            buckets_main[main]['stim'].append(stim_z)
            buckets_main[main]['resp'].append(resp_z)

    # ---------- fit & save: subgenres ----------
    recon_sub = {k: [] for k in buckets_sub.keys()}
    for gname, B in buckets_sub.items():
        if gname == 'Control':
            continue
        else:
            _fit_group_and_save(gname, B['stim'], B['resp'], recon_sub)
    # write recon datasets for subgenres
    save_recon_pergenre_letswave(
        out_root, recon_sub, fsEEG, template_lw6, prefix="recon genre_pooled"
    )

    # ---------- fit & save: main genres (Pop/Blues/Metal) ----------
    recon_main = {k: [] for k in buckets_main.keys()}
    for gname, B in buckets_main.items():
        _fit_group_and_save(gname, B['stim'], B['resp'], recon_main)
    # write recon datasets for main genres
    save_recon_pergenre_letswave(
        out_root, recon_main, fsEEG, template_lw6, prefix="recon genre_main"
    )


if __name__ == '__main__':
    # Choose what you want to run:
    ls = glob.glob('../data/preprocessed/DLS*.lw6')   
    for f in sorted(ls[0:]):
        filename      = os.path.basename(f)
        print(filename)
        ep_mat        = os.path.join('..','data','preprocessed', f'ep ds butt {filename[:-4]}.mat')
        ep_lw6        = os.path.join('..','data','preprocessed', f'ep ds butt {filename[:-4]}.lw6')
        raw_mat       = os.path.join('..','data','preprocessed', f'{filename[:-4]}.mat')
        raw_lw6       = os.path.join('..','data','preprocessed', f'{filename[:-4]}.lw6')
        stim_xlsx     = os.path.join('..','stimuli','Stimuli_Triggers.xlsx')
        stim_root     = os.path.join('..','stimuli')
        out_root      = os.path.join('..','out_trf', filename[:-4])
        os.makedirs(out_root, exist_ok=True)
        EEG, ep_hdr = load_letswave_mat(ep_mat,ep_lw6)          # [nEp x nCh x nT]
        fsEEG, _    = eeg_fs_from_header(ep_hdr)
        ch_all      = chan_labels_from_header(ep_hdr)
        nEp, nCh_all, nT = EEG.shape
        if ch_all is None:
            ch_all = np.array([f'Ch{i+1}' for i in range(nCh_all)])

        # channel selection
        ch_all    = np.asarray(ch_all)
        wanted_ch = np.asarray(wanted_ch)

        # Upper-case string versions (works for any dtype)
        ch_all_up   = np.char.upper(ch_all.astype(str))
        wanted_up   = np.char.upper(wanted_ch.astype(str))

        # Find indices in the order of wanted_ch; collect missing for a helpful warning
        idx_sel = []
        missing = []
        for w in wanted_up:
            hits = np.where(ch_all_up == w)[0]
            if hits.size:
                idx_sel.append(hits[0])     # take first match
            else:
                missing.append(w)

        idx_sel = np.asarray(idx_sel, dtype=int)
        ch_labels = ch_all[idx_sel]

        events = parse_raw_events(raw_lw6)
        genre_set_short = set(short_code(c) for c in genre_codes)
        rawGenreIdx = [i for i, e in enumerate(events) if short_code(e['code']) in genre_set_short]
        id2name = read_stim_mapping(stim_xlsx)
        trf_trialwise()
        # trf_per_genre()
