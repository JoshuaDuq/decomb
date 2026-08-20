"""comb_db on the UNCLEANED recording: the denominator the arm table never had."""
import os, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v,"1")
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
CONFIG="/Users/joduq24/Desktop/decomb/decomb.yaml"; COMB_HZ=1.2

def _one(name):
    import mne
    from scipy.signal import medfilt
    from decomb import recordings
    from decomb.config import load_config
    config=load_config(CONFIG)
    raw=recordings.read_bids_raw(config.path("bids_root")/name.split("_")[0]/"eeg"/f"{name}.vhdr")
    if recordings.acquisition_segments(raw)!=((0,raw.n_times),): return None
    sfreq=float(raw.info["sfreq"]); picks=mne.pick_types(raw.info,eeg=True,exclude="bads")
    n_fft=recordings.estimation_window_samples(sfreq,10.0); nyq=np.nextafter(sfreq/2.0,0.0)
    p,fr=mne.time_frequency.psd_array_welch(raw.get_data(picks=picks),sfreq,fmin=1.0,
        fmax=min(100.0,nyq),n_fft=n_fft,n_per_seg=n_fft,n_overlap=n_fft//2,average="mean",
        window="hamming",remove_dc=True,verbose="ERROR")
    db=10*np.log10(p.mean(axis=0)*1e12)
    res=db-medfilt(db,41); vals=[]
    for k in range(int(np.ceil(20.0/COMB_HZ)), int(95.0/COMB_HZ)+1):
        off=np.abs(fr-k*COMB_HZ); pk=off<=0.11; gp=(off>=0.35*COMB_HZ)&(off<=0.5*COMB_HZ)
        if pk.any() and gp.sum()>=2: vals.append(res[pk].max()-np.median(res[gp]))
    return {"recording":name,"raw_comb_db":round(float(np.median(vals)),2) if vals else np.nan}

def main():
    names=[l.strip() for l in open(sys.argv[1]) if l.strip()]
    rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(_one,n) for n in names]):
            r=f.result()
            if r: rows.append(r)
    d=pd.DataFrame(rows); d.to_csv(sys.argv[2],sep="\t",index=False)
    print(f"n={len(d)}  raw comb_db: median {d.raw_comb_db.median():+.2f}  mean {d.raw_comb_db.mean():+.2f}  p10 {d.raw_comb_db.quantile(.1):+.2f}  p90 {d.raw_comb_db.quantile(.9):+.2f}")


if __name__ == '__main__':
    main()
