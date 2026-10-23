import sys, glob, os, zipfile
import pandas as pd

COLS = ["date", "time", "open", "high", "low", "close", "vol"]


def _read(f):
    d = pd.read_csv(f, header=None, names=COLS, dtype={"date": str, "time": str})
    # HistData = fixed EST (UTC-5) -> convert to UTC
    ts = pd.to_datetime(d["date"] + " " + d["time"], format="%Y.%m.%d %H:%M")
    d.index = ts + pd.Timedelta(hours=5)
    return d[["open", "high", "low", "close"]].astype("float64")


def load_m1(src):
    frames = []
    if src.lower().endswith(".zip"):
        with zipfile.ZipFile(src) as z:
            for n in sorted(z.namelist()):
                if n.lower().endswith(".csv"):
                    with z.open(n) as f:
                        frames.append(_read(f))
    else:
        for p in sorted(glob.glob(os.path.join(src, "**", "*.csv"), recursive=True)):
            frames.append(_read(p))
    m1 = pd.concat(frames)
    m1 = m1[~m1.index.duplicated(keep="last")].sort_index()
    return m1


def to_tf(m1, rule="15min"):
    return (
        m1.resample(rule, label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    m1 = load_m1(src)
    print("M1 rows:", len(m1), m1.index.min(), "->", m1.index.max())
    m15 = to_tf(m1, "15min")
    m15.to_parquet(out, compression="zstd")
    print("M15 rows:", len(m15), "saved:", out, f"{os.path.getsize(out)/1e6:.1f} MB")
