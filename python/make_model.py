
import sys
import os
import json

import numpy as np
import pandas as pd
import lightgbm as lgb

from train_multi import (
    NUM,
    TOP_Q,
    RR,
    TRAIN_END,
    VALID_END,
)

ALLOWED_FAMILY = {"ICT"}

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


def main(source, output_dir):
    df = pd.read_parquet(source)

    required = {
        "family",
        "setup",
        "filled",
        "time",
        f"R_{RR}",
    }

    missing = required.difference(df.columns)

    if missing:
        raise RuntimeError(
            f"Dataset missing columns: {sorted(missing)}"
        )

    original_count = len(df)

    # Strict allow-list: ICT family and approved setups only.
    df = df[
        df["family"].isin(ALLOWED_FAMILY)
        & df["setup"].isin(ALLOWED_SETUPS)
    ].copy()

    print(
        f"ICT/SMC filter: kept {len(df)} "
        f"of {original_count} rows"
    )

    df = df[df["filled"] == 1].copy()

    if df.empty:
        raise RuntimeError(
            "No filled ICT/SMC examples. "
            "Model training cancelled."
        )

    df["time"] = pd.to_datetime(
        df["time"], utc=True
    ).dt.tz_localize(None)

    families = sorted(df["family"].unique())
    setups = sorted(df["setup"].unique())

    if families != ["ICT"]:
        raise RuntimeError(
            f"Unexpected training families: {families}"
        )

    df["fam_code"] = df["family"].map({
        name: i
        for i, name in enumerate(families)
    })

    df["setup_code"] = df["setup"].map({
        name: i
        for i, name in enumerate(setups)
    })

    for column in NUM:
        if column not in df.columns:
            df[column] = np.nan

    features = NUM + [
        "fam_code",
        "setup_code",
    ]

    target = (
        df[f"R_{RR}"] > 0
    ).astype(int)

    return_column = f"R_{RR}"

    train = df[df["time"] < TRAIN_END]

    valid = df[
        (df["time"] >= TRAIN_END)
        & (df["time"] < VALID_END)
    ]

    test = df[df["time"] >= VALID_END]

    print(
        f"ICT/SMC only | "
        f"train={len(train)} | "
        f"valid={len(valid)} | "
        f"test={len(test)}"
    )

    print("Allowed setups in this dataset:")
    for setup in setups:
        print(f"  - {setup}")

    if len(train) < 200 or len(valid) < 50:
        raise RuntimeError(
            "Not enough ICT/SMC training examples. "
            f"train={len(train)}, valid={len(valid)}. "
            "No new model was saved."
        )

    params = {
        "objective": "binary",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 10.0,
        "verbose": -1,
        "seed": 7,
    }

    model = lgb.train(
        params,
        lgb.Dataset(
            train[features],
            target.loc[train.index],
            categorical_feature=[
                "fam_code",
                "setup_code",
            ],
        ),
        num_boost_round=250,
    )

    # Threshold is chosen using validation data,
    # not the out-of-sample test set.
    valid_scores = model.predict(
        valid[features]
    )

    threshold = float(
        np.quantile(
            valid_scores,
            1 - TOP_Q,
        )
    )

    if not test.empty:
        test_scores = model.predict(
            test[features]
        )

        selected = test_scores >= threshold
        selected_returns = test.loc[
            selected, return_column
        ]

        selected_mean = (
            float(selected_returns.mean())
            if len(selected_returns)
            else None
        )

        baseline_mean = float(
            test[return_column].mean()
        )

        print(
            f"Threshold={threshold:.4f} | "
            f"test baseline mean R={baseline_mean:+.3f} | "
            f"selected trades={int(selected.sum())} | "
            f"selected mean R={selected_mean}"
        )
    else:
        print(
            "No out-of-sample test rows available."
        )

    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(
        output_dir, "h1_model.txt"
    )

    metadata_path = os.path.join(
        output_dir, "meta.json"
    )

    model.save_model(model_path)

    metadata = {
        "version": 2,
        "threshold": threshold,
        "features": features,
        "families": families,
        "setups": setups,
        "rr": RR,
        "trained_until": TRAIN_END,
        "strategy_scope": "ICT/SMC only",
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "Saved ICT/SMC-only model:",
        model_path,
    )

    print(
        "Saved metadata:",
        metadata_path,
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python make_model.py "
            "<dataset.parquet> <output_dir>"
        )

    main(sys.argv[1], sys.argv[2])
