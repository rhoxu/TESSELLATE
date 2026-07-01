"""
Sector-wide Bazin event search and clustering for tessellate.

Workflow
--------
1. fit_cut_events()  : fit Bazin to every positive event in one cut and write a
   candidate CSV.  Run one per cut (e.g. as a slurm job) across a sector.
2. aggregate_fits()  : concatenate the per-cut CSVs into one table.
3. cluster_events()  : cluster the well-fit events in Bazin-parameter space
   (log tau_rise, log tau_fall by default) to see whether they form groups.
4. template_cluster(): return the group that contains the studied event, i.e.
   all events similar to it.
"""

import os
import glob
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Per-cut fitting (one slurm job per cut)
# ---------------------------------------------------------------------------

def fit_cut_events(sector, cam, ccd, cut, data_path='/fred/oz335/TESSdata', n=8,
                   units='mJy', method='psf', n_durations=3, min_duration=3,
                   n_jobs=-1, savepath=None, **fit_kwargs):
    """
    Fit Bazin to every positive event in one cut and return / save the table.

    Thin wrapper around Navigator.fit_events (which splits the events across
    n_jobs worker processes) so it can be driven from a slurm script.  savepath
    defaults to the cut folder.
    """
    from .navigator import Navigator

    nav = Navigator(sector=sector, cam=cam, ccd=ccd, data_path=data_path, n=n)
    nav.gather_data(cut)
    nav.gather_results(cut)
    if nav.events is None or len(nav.events) == 0:
        print(f'No events for S{sector}C{cam}C{ccd}C{cut}.')
        return None

    if savepath is None:
        savepath = (f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/'
                    f'Cut{cut}of{n**2}/bazin_events.csv')
    df = nav.fit_events(cut=cut, units=units, method=method, n_durations=n_durations,
                        min_duration=min_duration, n_jobs=n_jobs,
                        savepath=savepath, **fit_kwargs)
    if df is not None:
        # Tag provenance so a sector-wide concatenation stays unambiguous
        df.insert(0, 'sector', sector)
        df.insert(1, 'cam', cam)
        df.insert(2, 'ccd', ccd)
        df.insert(3, 'cut', cut)
        df.to_csv(savepath, index=False)
    return df


# ---------------------------------------------------------------------------
# Sector-wide slurm dispatch (one job per cut)
# ---------------------------------------------------------------------------

def submit_sector_search(sector, cams=(1, 2, 3, 4), ccds=(1, 2, 3, 4), cuts=None,
                         n=8, data_path='/fred/oz335/TESSdata',
                         script_dir=None, log_dir=None,
                         units='mJy', method='psf', n_durations=3, min_duration=3,
                         time='01:00:00', cpus=8, mem=4, account='oz335',
                         submit=True):
    """
    Dispatch one slurm job per cut that fits Bazin to every positive event and
    writes <cut>/bazin_events.csv.  After the jobs finish, collect them with
    aggregate_fits('<data_path>/Sector<sector>') and cluster with cluster_events.

    cuts defaults to all n**2 cuts.  Set submit=False to only write the scripts.
    Returns the list of submitted job ids.
    """
    import sys
    import subprocess

    venv = sys.prefix
    cuts = list(cuts) if cuts is not None else list(range(1, n ** 2 + 1))
    script_dir = script_dir or f'{data_path}/Sector{sector}/bazin_search_scripts'
    log_dir = log_dir or f'{data_path}/Sector{sector}/bazin_search_logs'
    os.makedirs(script_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    job_ids = []
    for cam in cams:
        for ccd in ccds:
            for cut in cuts:
                base = f'{script_dir}/S{sector}C{cam}C{ccd}C{cut}_bazin'
                with open(base + '.py', 'w') as f:
                    f.write(
                        'from tessellate.event_search import fit_cut_events\n'
                        f'fit_cut_events(sector={sector}, cam={cam}, ccd={ccd}, '
                        f'cut={cut}, data_path="{data_path}", n={n}, '
                        f'units="{units}", method="{method}", n_durations={n_durations}, '
                        f'min_duration={min_duration}, n_jobs={cpus})\n'
                    )
                with open(base + '.sh', 'w') as f:
                    f.write(
                        '#!/bin/bash\n'
                        f'#SBATCH --job-name=TESS_S{sector}_C{cam}_C{ccd}_C{cut}_bazin\n'
                        f'#SBATCH --output={log_dir}/%A_%x_out.txt\n'
                        f'#SBATCH --error={log_dir}/%A_%x_err.txt\n'
                        '#SBATCH --ntasks=1\n'
                        f'#SBATCH --time={time}\n'
                        f'#SBATCH --cpus-per-task={cpus}\n'
                        f'#SBATCH --mem-per-cpu={mem}G\n'
                        f'#SBATCH --account={account}\n\n'
                        'PYTHONUNBUFFERED=1\n'
                        f'source {venv}/bin/activate\n'
                        f'python {base}.py\n'
                    )
                if submit:
                    r = subprocess.run(f'sbatch {base}.sh', shell=True,
                                       capture_output=True, text=True)
                    if r.returncode == 0 and r.stdout.strip():
                        job_ids.append(r.stdout.strip().split()[-1])
                    else:
                        print(f'sbatch failed S{sector}C{cam}C{ccd}C{cut}: '
                              f'{r.stderr.strip()}')

    print(f'{"Submitted" if submit else "Wrote"} {len(cams)*len(ccds)*len(cuts)} '
          f'cut jobs ({len(job_ids)} submitted).')
    return job_ids


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_fits(paths):
    """
    Concatenate per-cut Bazin CSVs into one table.

    paths : a directory (searched recursively for bazin_events.csv), a glob
    pattern, or a list of file paths.
    """
    if isinstance(paths, str):
        if os.path.isdir(paths):
            files = glob.glob(os.path.join(paths, '**', 'bazin_events.csv'),
                              recursive=True)
        else:
            files = glob.glob(paths)
    else:
        files = list(paths)
    if not files:
        raise FileNotFoundError('No Bazin event CSVs found.')
    frames = [pd.read_csv(f) for f in files]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _quality_mask(df, delta_bic=-6.0, redchi2_max=3.0, min_asnr=5.0):
    """Well-fit, significant, Bazin-like events to include in the clustering."""
    keep = df.get('fit_ok', pd.Series(True, index=df.index)).fillna(False).to_numpy(dtype=bool)
    if delta_bic is not None and 'delta_bic' in df:
        keep &= (df['delta_bic'] <= delta_bic).fillna(False).to_numpy()
    if redchi2_max is not None and 'redchi2_region' in df:
        keep &= (df['redchi2_region'] <= redchi2_max).fillna(False).to_numpy()
    if min_asnr is not None and 'A_snr' in df:
        keep &= (df['A_snr'] >= min_asnr).fillna(False).to_numpy()
    return keep


def cluster_events(df, features=('tau_rise', 'tau_fall', 'tau_ratio',
                                 'fwhm', 'rise_time', 'decay_time'),
                   log=True, whiten=True, pca_variance=0.99,
                   delta_bic=-6.0, redchi2_max=3.0, min_asnr=5.0,
                   min_cluster_size=15, min_samples=None):
    """
    Cluster the well-fit events in Bazin-parameter space.

    Adds a 'cluster' column to df:
       >=0 : cluster label
        -1 : clustered as noise (passed quality, not in a dense group)
        -2 : excluded from clustering (failed the quality cuts)

    Density clustering (HDBSCAN if available, else DBSCAN) is used so groups need
    not be specified in advance and outliers fall out as noise.  Features are
    log-scaled (default) and standardised before clustering.  The default feature
    set is shape-only (timescales and morphology) -- brightness features (peak,
    fluence, A) are excluded since events span a range of brightnesses.  Note the
    defaults are correlated (tau_ratio, fwhm, rise/decay_time all follow from the
    timescales), which up-weights the shape axes.
    """
    df = df.copy()
    df['cluster'] = -2

    keep = _quality_mask(df, delta_bic, redchi2_max, min_asnr)
    for feat in features:
        keep &= np.isfinite(df[feat].to_numpy())
        if log:
            keep &= (df[feat].to_numpy() > 0)
    if keep.sum() < min_cluster_size:
        print(f'Only {int(keep.sum())} events pass quality cuts; '
              'too few to cluster.')
        return df

    X = np.column_stack([
        np.log10(df.loc[keep, feat].to_numpy()) if log else df.loc[keep, feat].to_numpy()
        for feat in features
    ])
    Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

    # De-correlate: PCA-whiten onto the independent directions, dropping the
    # near-degenerate ones (correlated features -> a lower-dimensional basis).
    if whiten and Xs.shape[1] > 1:
        try:
            from sklearn.decomposition import PCA
            Xs = PCA(n_components=pca_variance, whiten=True,
                     random_state=0).fit_transform(Xs)
            print(f'  PCA-whitened to {Xs.shape[1]} independent component(s) '
                  f'(>= {pca_variance:.0%} variance).')
        except Exception as exc:
            print(f'  PCA whitening skipped ({exc}); clustering on raw features.')

    labels = None
    try:
        from sklearn.cluster import HDBSCAN
        labels = HDBSCAN(min_cluster_size=min_cluster_size,
                         min_samples=min_samples).fit_predict(Xs)
    except Exception:
        try:
            from sklearn.cluster import DBSCAN
            labels = DBSCAN(eps=0.5, min_samples=min_cluster_size).fit_predict(Xs)
        except Exception as exc:
            raise ImportError('scikit-learn is required for clustering '
                              f'(HDBSCAN/DBSCAN): {exc}')

    df.loc[keep, 'cluster'] = labels
    nclus = len(set(labels) - {-1})
    print(f'Clustered {int(keep.sum())} events into {nclus} group(s) '
          f'({int(np.sum(labels == -1))} noise).')
    return df


def template_cluster(df, objid, eventid, sector=None, cam=None, ccd=None, cut=None):
    """
    Return the subset of df in the same cluster as the template event -- i.e. all
    events similar to it.  Returns None if the template was not clustered.
    """
    sel = (df['objid'] == objid) & (df['eventid'] == eventid)
    for col, val in (('sector', sector), ('cam', cam), ('ccd', ccd), ('cut', cut)):
        if val is not None and col in df:
            sel &= df[col] == val
    match = df[sel]
    if len(match) == 0:
        print('Template event not found in the table.')
        return None
    lbl = int(match['cluster'].iloc[0])
    if lbl < 0:
        print(f'Template event was not assigned to a cluster (label {lbl}).')
        return None
    group = df[df['cluster'] == lbl].copy()
    print(f'Template is in cluster {lbl} with {len(group)} events.')
    return group


def plot_clusters(df, features=('tau_rise', 'tau_fall'), log=True,
                  template=None, savepath=None):
    """
    Scatter the clustered events in the two feature axes, coloured by cluster.
    template = (objid, eventid) marks the studied event.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    fx, fy = features[0], features[1]
    clustered = df[df['cluster'] >= -1]
    fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)

    noise = clustered[clustered['cluster'] == -1]
    ax.scatter(noise[fx], noise[fy], s=8, c='0.7', alpha=0.5, label='noise')
    for lbl in sorted(set(clustered['cluster']) - {-1}):
        g = clustered[clustered['cluster'] == lbl]
        ax.scatter(g[fx], g[fy], s=14, alpha=0.8, label=f'cluster {lbl}')

    if template is not None:
        tsel = (df['objid'] == template[0]) & (df['eventid'] == template[1])
        if tsel.any():
            tr = df[tsel].iloc[0]
            ax.scatter([tr[fx]], [tr[fy]], marker='*', s=260, c='k',
                       edgecolors='w', zorder=6, label='template')

    if log:
        ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(fx); ax.set_ylabel(fy)
    ax.set_title('Bazin event clusters')
    ax.legend(fontsize=7)
    if savepath is not None:
        fig.savefig(savepath, bbox_inches='tight')
        print(f'Saved: {savepath}')
    else:
        plt.show()
    plt.close(fig)
