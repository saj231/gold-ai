
import sys
import pandas as pd
import smc_engine as se
import common as cm

TFS = {"H1": ("1h", 60, 24, 48), "H4": ("4h", 240, 12, 30)}

ICT_EXTRA = [
    "zone_size",
    "zone_disp",
    "bars_since_sweep",
    "sweep_same",
    "bars_since_bos",
    "bos_same",
]

ALLOWED_FAMILY = "ICT"

ALLOWED_SETUPS = {
    "BOS_FVG",
    "BOS_OB",
    "BREAKER",
    "IFVG",
    "JUDAS_SWING",
    "OTE",
    "SILVER_BULLET",
    "SWEEP_CHOCH_OB",
    "SWEEP_DISP_FVG",
    "SWEEP_RECLAIM",
}


def build(m15, tf):
    if tf not in TFS:
        raise ValueError(f"Unsupported timeframe: {tf}")

    rule, mins, expire, hold = TFS[tf]

    df = m15.resample(
        rule,
        label="left",
        closed="left",
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }).dropna()

    df = se.prepare(df)

    df = cm.context(
        df,
        int(365 * 24 * 60 / mins * 0.7),
    )

    se.BAR_MIN = mins
    se.MIN_RISK_ATR = cm.MIN_RISK_ATR
    se.MAX_RISK_ATR = cm.MAX_RISK_ATR

    # تنها موتور ICT/SMC اجازه تولید سفارش دارد.
    ict = se.detect(df)

    if ict.empty:
        raise RuntimeError(
            f"{tf}: smc_engine produced no ICT/SMC orders. "
            "Training dataset was not generated."
        )

    required = {
        "bar",
        "setup",
        "dir",
        "limit",
        "stop",
    }

    missing = required.difference(ict.columns)

    if missing:
        raise RuntimeError(
            f"{tf}: smc_engine output is missing columns: "
            f"{sorted(missing)}"
        )

    # فقط نام ستاپ‌های مجاز وارد دیتاست می‌شوند.
    ict = ict[
        ict["setup"].isin(ALLOWED_SETUPS)
    ].copy()

    if ict.empty:
        raise RuntimeError(
            f"{tf}: no orders matched the ICT/SMC allow-list."
        )

    orders = pd.DataFrame({
        "bar": ict["bar"].to_numpy(),
        "family": ALLOWED_FAMILY,
        "setup": ict["setup"].astype(str).to_numpy(),
        "dir": ict["dir"].to_numpy(),
        "mode": "limit",
        "limit": ict["limit"].to_numpy(),
        "stop": ict["stop"].to_numpy(),
        "ref": ict["limit"].to_numpy(),
    })

    # این ویژگی‌ها مستقیماً از موتور SMC می‌آیند.
    for column in ICT_EXTRA:
        if column in ict.columns:
            orders[column] = ict[column].to_numpy()
        else:
            orders[column] = float("nan")

    orders = cm.add_features(
        df,
        orders,
        mins,
    )

    orders = cm.label(
        df,
        orders,
        expire,
        hold,
    )

    orders = orders.dropna(
        subset=["filled"]
    ).reset_index(drop=True)

    orders["tf"] = tf

    # فیلتر نهایی قبل از ذخیره داده‌های آموزشی.
    orders = orders[
        orders["family"].eq(ALLOWED_FAMILY)
        & orders["setup"].isin(ALLOWED_SETUPS)
    ].copy()

    if orders.empty:
        raise RuntimeError(
            f"{tf}: no labeled ICT/SMC orders remain after simulation."
        )

    print(
        f"{tf}: ICT/SMC orders={len(orders)}"
    )

    return orders


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python build_multi.py "
            "<m15.parquet> <output_prefix>"
        )

    source, prefix = sys.argv[1], sys.argv[2]

    m15 = pd.read_parquet(source)

    for tf in TFS:
        orders = build(m15, tf)

        output_path = f"{prefix}_{tf}.parquet"

        orders.to_parquet(
            output_path,
            compression="zstd",
        )

        filled = orders[
            orders["filled"] == 1
        ]

        print(
            f"{tf}: total={len(orders)} "
            f"filled={len(filled)}"
        )

        if filled.empty:
            print(
                f"{tf}: no filled ICT/SMC orders "
                "available for performance reporting."
            )
            continue

        grouped = filled.groupby(
            ["family", "setup"]
        )

        report = pd.DataFrame({
            "n": grouped.size(),
            "R1.5": grouped["R_1.5"].mean(),
            "R2": grouped["R_2.0"].mean(),
            "R3": grouped["R_3.0"].mean(),
        })

        print(
            report.round(3)
            .sort_values("R2", ascending=False)
            .to_string()
        )

        print(
            "ALL:",
            {
                rr: round(
                    float(filled[f"R_{rr}"].mean()),
                    3,
                )
                for rr in cm.RRS
            },
        )
