
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

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
            f"Dataset is missing required columns: "
            f"{sorted(missing)}"
        )

    initial_rows = len(df)

    # فقط معاملات پرشده و خانواده ICT/SMC مجازند.
    df = df[
        df["family"].isin(ALLOWED_FAMILY)
        & df["setup"].isin(ALLOWED_SETUPS)
        & (df["filled"] == 1)
    ].copy()

    print(
        f"ICT/SMC-only filter: kept "
        f"{len(df)} of {initial_rows} rows"
    )

    if df.empty:
        raise RuntimeError(
            "No filled ICT/SMC examples found. "
            "Refusing to train or overwrite the model."
        )

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(
        subset=["time", f"R_{RR}"]
    ).copy()

    df["time"] = df["time"].dt.tz_localize(None)

    families = sorted(
        df["family"].astype(str).unique()
    )

    setups = sorted(
        df["setup"].astype(str).unique()
    )

    if families != ["ICT"]:
        raise RuntimeError(
            f"Training data contains unexpected families: "
            f"{families}"
        )

    if not setups:
        raise RuntimeError(
            "No allowed ICT/SMC setup types remain."
        )

    family_codes = {
        name: index
        for index, name in enumerate(families)
    }

    setup_codes = {
        name: index
        for index, name in enumerate(setups)
    }

    df["fam_code"] = (
        df["family"]
        .map(family_codes)
        .astype(int)
    )

    df["setup_code"] = (
        df["setup"]
        .map(setup_codes)
        .astype(int)
    )

    for column in NUM:
        if column not in df.columns:
            df[column] = np.nan

    features = NUM + [
        "fam_code",
        "setup_code",
    ]

    target_column = f"R_{RR}"

    target = (
        df[target_column] > 0
    ).astype(int)

    train = df[
        df["time"] < TRAIN_END
    ]

    valid = df[
        (df["time"] >= TRAIN_END)
        & (df["time"] < VALID_END)
    ]

    test = df[
        df["time"] >= VALID_END
    ]

    print(
        f"ICT/SMC dataset | "
        f"train={len(train)} "
        f"valid={len(valid)} "
        f"test={len(test)}"
    )

    print(
        "Setups present in dataset:",
        ", ".join(setups),
    )

    if len(train) < 200 or len(valid) < 50:
        raise RuntimeError(
            "Not enough ICT/SMC examples for training "
            f"(train={len(train)}, valid={len(valid)}). "
            "The existing model has not been overwritten."
        )

    if target.loc[train.index].nunique() < 2:
        raise RuntimeError(
            "Training target has only one class. "
            "Refusing to train an unusable model."
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
        "verbosity": -1,
        "seed": 7,
    }

    train_set = lgb.Dataset(
        train[features],
        label=target.loc[train.index],
        categorical_feature=[
            "fam_code",
            "setup_code",
        ],
        free_raw_data=False,
    )

    model = lgb.train(
        params,
        train_set,
        num_boost_round=250,
    )

    # آستانه فقط با بخش اعتبارسنجی تعیین می‌شود.
    validation_scores = model.predict(
        valid[features]
    )

    threshold = float(
        np.quantile(
            validation_scores,
            1 - TOP_Q,
        )
    )

    if not test.empty:
        test_scores = model.predict(
            test[features]
        )

        selected_mask = (
            test_scores >= threshold
        )

        selected = test.loc[
            selected_mask,
            target_column,
        ]

        selected_mean = (
            float(selected.mean())
            if len(selected)
            else None
        )

        baseline_mean = float(
            test[target_column].mean()
        )

        print(
            f"Threshold={threshold:.4f} | "
            f"test baseline mean R={baseline_mean:+.3f} | "
            f"selected n={len(selected)} | "
            f"selected mean R={selected_mean}"
        )
    else:
        print(
            "No out-of-sample test rows available."
        )

    # فقط بعد از عبور از کنترل‌ها، مدل جدید ذخیره می‌شود.
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    model_path = os.path.join(
        output_dir,
        "h1_model.txt",
    )

    meta_path = os.path.join(
        output_dir,
        "meta.json",
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
        meta_path,
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
        f"Saved ICT/SMC-only model: {model_path}"
    )

    print(
        f"Saved model metadata: {meta_path}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python make_model.py "
            "<dataset.parquet> <output_dir>"
        )

    main(
        sys.argv[1],
        sys.argv[2],
    )
