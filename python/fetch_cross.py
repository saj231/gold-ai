import os, sys, time, json, gzip, csv, urllib.request, urllib.parse
from datetime import datetime, timedelta

SYMS = ["EUR/USD", "USD/JPY", "GBP/USD", "USD/CAD", "USD/SEK", "USD/CHF", "XAG/USD"]
OPTIONAL = ["DXY", "US10Y", "SPX", "VIX", "WTI/USD"]   # may not exist on the free plan: failures are skipped
START = "2009-03-01 00:00:00"
FMT = "%Y-%m-%d %H:%M:%S"
PAUSE = 8          # free plan: 8 requests per minute


def call(params):
    url = "https://api.twelvedata.com/time_series?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "gold-scanner"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def fetch_symbol(sym, key):
    rows, start, tries = {}, START, 0
    while True:
        js = call(dict(symbol=sym, interval="1h", outputsize=5000, timezone="UTC",
                       order="ASC", start_date=start, apikey=key))
        if js.get("status") == "error" or "values" not in js:
            msg = str(js.get("message", js))
            if ("credits" in msg.lower() or "limit" in msg.lower()) and tries < 4:
                tries += 1
                print(sym, "rate limit, waiting 65s:", msg[:80], flush=True)
                time.sleep(65)
                continue
            print(sym, "stopped:", msg[:160], flush=True)
            break
        tries = 0
        vals = js["values"]
        if not vals:
            break
        for v in vals:
            rows[v["datetime"]] = (v["open"], v["high"], v["low"], v["close"])
        print(sym, "got", len(vals), "up to", vals[-1]["datetime"], flush=True)
        if len(vals) < 5000:
            break
        start = (datetime.strptime(vals[-1]["datetime"], FMT) + timedelta(hours=1)).strftime(FMT)
        time.sleep(PAUSE)
    return rows


FRED = ["DFII10", "DGS10", "DGS2", "T10YIE", "VIXCLS", "SP500", "DTWEXBGS"]   # real yield, 10y, 2y, breakeven, VIX, S&P, broad dollar


def fetch_fred(sid):
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + sid
    req = urllib.request.Request(url, headers={"User-Agent": "gold-scanner"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode()


def main(outdir):
    key = os.environ["TWELVEDATA_KEY"]
    os.makedirs(outdir, exist_ok=True)
    for sym in SYMS + OPTIONAL:
        rows = fetch_symbol(sym, key)
        name = sym.replace("/", "")
        if rows:
            ks = sorted(rows)
            with gzip.open(os.path.join(outdir, name + "_H1.csv.gz"), "wt", newline="") as f:
                w = csv.writer(f)
                w.writerow(["datetime", "open", "high", "low", "close"])
                for k in ks:
                    w.writerow((k,) + rows[k])
            print(f"SAVED {name}: {len(ks)} bars, {ks[0]} -> {ks[-1]}", flush=True)
        else:
            print(f"FAILED {name}: no data", flush=True)
        time.sleep(PAUSE)
    for sid in FRED:
        try:
            txt = fetch_fred(sid)
            lines = [l for l in txt.strip().splitlines() if l]
            with gzip.open(os.path.join(outdir, "FRED_" + sid + ".csv.gz"), "wt") as f:
                f.write("\n".join(lines) + "\n")
            print(f"SAVED FRED {sid}: {len(lines) - 1} rows, last: {lines[-1]}", flush=True)
        except Exception as e:
            print(f"FAILED FRED {sid}: {str(e)[:120]}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])


# END OF FILE
