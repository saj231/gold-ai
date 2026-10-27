
import os
import json
import urllib.request
import urllib.parse
import urllib.error

import numpy as np
import pandas as pd

import smc_engine as se
import common as cm
import tracker
from build_multi import ICT_EXTRA


MINS = 60
SYMBOL = "XAU/USD"
MD = os.environ.get("MODEL_DIR", "../model")

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

FAM_FA = {
    "ICT": "ICT/SMC",
}


def http_json(url, data=None, headers=None, timeout=40):
    body = (
        json.dumps(data, ensure_ascii=False).encode("utf-8")
        if data is not None
        else None
    )

    request_headers = {
        "User-Agent": "gold-ai-scanner/2.0",
    }
    request_headers.update(headers or {})

    request = urllib.request.Request(
        url,
        data=body,
        headers=request_headers,
    )

    with urllib.request.urlopen(
        request,
        timeout=timeout,
    ) as response:
        text = response.read().decode(
            "utf-8",
            errors="replace",
        )

    return json.loads(text) if text else None


def telegram(token, chat, text):
    if not token or not chat:
        raise RuntimeError(
            "Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID"
        )

    remaining = str(text)
    chunks = []

    while len(remaining) > 3900:
        split_at = remaining.rfind(
            "\n",
            0,
            3900,
        )

        if split_at < 1000:
            split_at = 3900

        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip()

    if remaining:
        chunks.append(remaining)

    for chunk in chunks:
        result = http_json(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={
                "chat_id": chat,
                "text": chunk,
                "disable_web_page_preview": True,
            },
            headers={
                "Content-Type": "application/json",
            },
            timeout=20,
        )

        if (
            not isinstance(result, dict)
            or result.get("ok") is not True
        ):
            description = (
                result.get(
                    "description",
                    "Unknown Telegram API error",
                )
                if isinstance(result, dict)
                else "Invalid Telegram response"
            )

            raise RuntimeError(
                f"Telegram sendMessage failed: {description}"
            )


def fetch_bars(key, count=5000):
    if not key:
        raise RuntimeError(
            "TWELVEDATA_KEY is missing"
        )

    query = urllib.parse.urlencode({
        "symbol": SYMBOL,
        "interval": "1h",
        "outputsize": count,
        "timezone": "UTC",
        "order": "ASC",
        "apikey": key,
    })

    result = http_json(
        "https://api.twelvedata.com/time_series?" + query,
        timeout=40,
    )

    if not isinstance(result, dict) or "values" not in result:
        message = (
            result.get("message", "Unknown Twelve Data error")
            if isinstance(result, dict)
            else "Invalid response"
        )

        raise RuntimeError(
            f"Twelve Data failed: {message}"
        )

    df = pd.DataFrame(result["values"])

    if df.empty:
        raise RuntimeError(
            "Twelve Data returned no candles"
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True,
    ).dt.tz_localize(None)

    df = (
        df.set_index("datetime")
        .sort_index()
    )

    for column in ("open", "high", "low", "close"):
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df[
        ["open", "high", "low", "close"]
    ]

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    df = df[
        ~df.index.duplicated(keep="last")
    ]

    if len(df) < 200:
        raise RuntimeError(
            f"Not enough usable candles: {len(df)}"
        )

    return df


def compute_orders(raw):
    df = se.prepare(raw)

    df = cm.context(
        df,
        int(365 * 24 * 60 / MINS * 0.7),
    )

    se.BAR_MIN = MINS
    se.MIN_RISK_ATR = cm.MIN_RISK_ATR
    se.MAX_RISK_ATR = cm.MAX_RISK_ATR

    # تنها موتور مجاز: ICT/SMC
    ict = se.detect(df)

    if ict.empty:
        return df, pd.DataFrame()

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
            f"SMC engine missing columns: {sorted(missing)}"
        )

    ict = ict[
        ict["setup"].isin(ALLOWED_SETUPS)
    ].copy()

    if ict.empty:
        return df, pd.DataFrame()

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

    for column in ICT_EXTRA:
        if column in ict.columns:
            orders[column] = ict[column].to_numpy()
        else:
            orders[column] = np.nan

    orders = orders[
        orders["family"].eq(ALLOWED_FAMILY)
        & orders["setup"].isin(ALLOWED_SETUPS)
    ].copy()

    if orders.empty:
        return df, pd.DataFrame()

    orders = cm.add_features(
        df,
        orders,
        MINS,
    )

    return df, orders


def make_X(orders, meta):
    X = pd.DataFrame(
        index=orders.index
    )

    family_codes = {
        name: index
        for index, name in enumerate(meta["families"])
    }

    setup_codes = {
        name: index
        for index, name in enumerate(meta["setups"])
    }

    for column in meta["features"]:
        if column == "fam_code":
            X[column] = (
                orders["family"]
                .map(family_codes)
                .fillna(-1)
            )

        elif column == "setup_code":
            X[column] = (
                orders["setup"]
                .map(setup_codes)
                .fillna(-1)
            )

        elif column in orders.columns:
            X[column] = orders[column]

        else:
            X[column] = np.nan

    return X.astype(float)


def select(orders, meta, booster, first_bar):
    X = make_X(
        orders,
        meta,
    )

    scored = orders.copy()

    scored["score"] = booster.predict(
        X[meta["features"]]
    )

    scored = scored.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    scored = scored[
        (scored["bar"] >= first_bar)
        & scored["score"].notna()
    ]

    return scored.sort_values(
        "score",
        ascending=False,
    ).copy()


def group_picks(picks):
    # ستاپ‌های هم‌زمان و هم‌جهت را در یک گروه گزارش می‌کنیم.
    groups = []

    for _, group in picks.groupby(
        ["time", "dir"],
        sort=False,
    ):
        top = group.iloc[0].copy()

        top["setup"] = " + ".join(
            group["setup"].astype(str).tolist()[:3]
        )

        groups.append(top)

    return sorted(
        groups,
        key=lambda row: -float(row["score"]),
    )


def safe_float(value, digits=3):
    try:
        number = float(value)

        if not np.isfinite(number):
            return None

        return round(number, digits)

    except (TypeError, ValueError):
        return None


def recent_candles(df, count=10):
    result = []

    for timestamp, row in df.tail(count).iterrows():
        result.append({
            "time_utc": pd.Timestamp(
                timestamp
            ).strftime("%Y-%m-%d %H:%M"),
            "open": safe_float(row["open"], 2),
            "high": safe_float(row["high"], 2),
            "low": safe_float(row["low"], 2),
            "close": safe_float(row["close"], 2),
        })

    return result


def candidate_payload(row, index, threshold):
    direction = int(row["dir"])
    entry = float(row["ref"])
    stop = float(row["stop"])
    risk = abs(entry - stop)

    return {
        "id": f"C{index}",
        "family": "ICT/SMC",
        "setup": str(row["setup"]),
        "direction": (
            "BUY" if direction == 1 else "SELL"
        ),
        "entry_type": str(row["mode"]),
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "tp_1_5R": round(
            entry + direction * 1.5 * risk,
            2,
        ),
        "tp_2R": round(
            entry + direction * 2.0 * risk,
            2,
        ),
        "tp_3R": round(
            entry + direction * 3.0 * risk,
            2,
        ),
        "lightgbm_score": round(
            float(row["score"]),
            4,
        ),
        "lightgbm_threshold": round(
            float(threshold),
            4,
        ),
        "passed_lightgbm_threshold": bool(
            float(row["score"]) >= threshold
        ),
        "bar_time_utc": pd.Timestamp(
            row["time"]
        ).strftime("%Y-%m-%d %H:%M"),
        "risk_atr": safe_float(
            row.get("risk_atr")
        ),
        "with_trend": safe_float(
            row.get("with_trend"),
            0,
        ),
    }


def gemini_review(
    api_key,
    candles,
    candidates,
    threshold,
    latest_close,
):
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured"
        )

    model = (
        os.environ.get(
            "GEMINI_MODEL",
            "gemini-2.5-flash",
        ).strip()
        or "gemini-2.5-flash"
    )

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + urllib.parse.quote(model, safe="-.")
        + ":generateContent?key="
        + urllib.parse.quote(api_key)
    )

    system = (
        "تو تحلیل‌گر کمکی بازار XAUUSD هستی و باید فارسی پاسخ بدهی. "
        "فقط از داده‌های ورودی استفاده کن. "
        "قیمت، خبر، اندیکاتور یا ستاپ جدید جعل نکن. "
        "فقط ستاپ‌های ICT/SMC ارائه‌شده در ورودی مجازند. "
        "ستاپ واجد شرایط فقط نامزدی است که "
        "passed_lightgbm_threshold=true داشته باشد. "
        "اگر هیچ نامزدی از آستانه عبور نکرده، "
        "صریح بگو ستاپ واجد شرایطی وجود ندارد و معامله پیشنهاد نکن. "
        "امتیاز LightGBM احتمال قطعی برد نیست. "
        "خروجی کوتاه، فارسی و مناسب تلگرام باشد."
    )

    # درخواست کوچک‌تر برای کاهش احتمال Timeout
    candles = candles[-10:] if isinstance(candles, list) else []
    candidates = candidates[:5] if isinstance(candidates, list) else []

    prompt = {
        "task": (
            "کندل‌های اخیر و نامزدهای واقعی ICT/SMC را خلاصه کن. "
            "فقط داده‌های ورودی را بررسی کن. "
            "اگر هیچ نامزدی از آستانه عبور نکرده، "
            "سیگنال نساز و این موضوع را صریح بگو. "
            "ریسک‌ها و محدودیت اطلاعات را کوتاه توضیح بده."
        ),
        "latest_closed_bar_end_utc": latest_close,
        "lightgbm_threshold": threshold,
        "candles_ohlc": candles,
        "candidates": candidates,
    }

    body = {
        "systemInstruction": {
            "parts": [{
                "text": system,
            }],
        },
        "contents": [{
            "role": "user",
            "parts": [{
                "text": json.dumps(
                    prompt,
                    ensure_ascii=False,
                ),
            }],
        }],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 600,
        },
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(
            body,
            ensure_ascii=False,
        ).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=55,
        ) as response:
            response_text = response.read().decode(
                "utf-8",
                errors="replace",
            )

    except urllib.error.HTTPError as exc:
        details = exc.read(1200).decode(
            "utf-8",
            errors="replace",
        )

        try:
            parsed = json.loads(details)
            details = parsed.get(
                "error",
                {},
            ).get("message", details)
        except (ValueError, TypeError):
            pass

        # URL درخواست چاپ نمی‌شود تا API key افشا نشود.
        print(
            f"Gemini HTTP {exc.code}: "
            f"{str(details)[:400]}"
        )

        raise RuntimeError(
            f"Gemini API returned HTTP {exc.code}"
        ) from None

    except Exception as exc:
        print(
            f"Gemini request failed: "
            f"{type(exc).__name__}: {str(exc)[:200]}"
        )

        raise RuntimeError(
            f"Gemini request failed ({type(exc).__name__})"
        ) from None

    try:
        data = json.loads(response_text)

        answer = "\n".join(
            part.get("text", "")
            for part in data["candidates"][0]["content"]["parts"]
            if isinstance(part, dict)
        ).strip()

    except (ValueError, KeyError, IndexError, TypeError):
        raise RuntimeError(
            "Gemini response could not be parsed"
        ) from None

    if not answer:
        raise RuntimeError(
            "Gemini returned an empty response"
        )

    return answer[:2200]


def format_candidate(row, threshold, tag=""):
    direction = int(row["dir"])
    entry = float(row["ref"])
    stop = float(row["stop"])
    risk = abs(entry - stop)

    side = (
        "🟢 خرید (BUY)"
        if direction == 1
        else "🔴 فروش (SELL)"
    )

    if str(row["mode"]) == "limit":
        entry_text = f"لیمیت روی {entry:.2f}"
    else:
        entry_text = f"ورود بازار نزدیک {entry:.2f}"

    timestamp = pd.Timestamp(
        row["time"]
    ).strftime("%Y-%m-%d %H:%M")

    tp15 = entry + direction * 1.5 * risk
    tp2 = entry + direction * 2.0 * risk
    tp3 = entry + direction * 3.0 * risk

    return (
        f"{tag}📊 ستاپ XAUUSD H1\n"
        f"{side}\n"
        f"سبک: ICT/SMC\n"
        f"ستاپ: {row['setup']}\n"
        f"ورود: {entry_text}\n"
        f"حد ضرر: {stop:.2f}\n"
        f"ریسک قیمتی: {risk:.2f}\n"
        f"اهداف محاسباتی: "
        f"1.5R={tp15:.2f} | "
        f"2R={tp2:.2f} | "
        f"3R={tp3:.2f}\n"
        f"امتیاز LightGBM: {float(row['score']):.3f}\n"
        f"آستانه مدل: {threshold:.3f}\n"
        f"زمان کندل: {timestamp} UTC\n"
        "⚠️ امتیاز مدل تضمین سود یا احتمال قطعی برد نیست."
    )


def persist_candidates(
    supabase_url,
    service_key,
    picks,
    meta,
):
    if (
        not supabase_url
        or not service_key
        or picks.empty
    ):
        return

    rows = []

    for _, row in picks.iterrows():
        direction = int(row["dir"])
        entry = float(row["ref"])
        stop = float(row["stop"])
        risk = abs(entry - stop)

        rows.append({
            "bar_time": (
                pd.Timestamp(row["time"]).isoformat() + "Z"
            ),
            "symbol": "XAUUSD",
            "setup": str(row["setup"]),
            "direction": (
                "BUY" if direction == 1 else "SELL"
            ),
            "entry": entry,
            "stop": stop,
            "tp1": entry + direction * 1.5 * risk,
            "tp2": entry + direction * 2.0 * risk,
            "tp3": entry + direction * 3.0 * risk,
            "probability": float(row["score"]),
            "model_version": meta.get("version", 1),
            "mode": str(row["mode"]),
            "family": "ICT",
        })

    headers = {
        "apikey": service_key,
        "Authorization": "Bearer " + service_key,
        "Content-Type": "application/json",
        "Prefer": (
            "resolution=ignore-duplicates,return=representation"
        ),
    }

    endpoint = (
        supabase_url.rstrip("/")
        + "/rest/v1/signals"
        + "?on_conflict=symbol,bar_time,setup,direction"
    )

    try:
        http_json(
            endpoint,
            data=rows,
            headers=headers,
            timeout=20,
        )

        print(
            f"Supabase persistence attempted for {len(rows)} candidates"
        )

    except Exception as exc:
        # خطای دیتابیس نباید ارسال گزارش تلگرام را متوقف کند.
        print(
            f"Supabase persistence failed: "
            f"{type(exc).__name__}: {str(exc)[:250]}"
        )


def main():
    import lightgbm as lgb

    mode = (
        os.environ.get("MODE", "scan")
        .strip()
        .lower()
    )

    token = os.environ.get(
        "TELEGRAM_TOKEN",
        "",
    ).strip()

    chat_id = os.environ.get(
        "TELEGRAM_CHAT_ID",
        "",
    ).strip()

    twelve_key = os.environ.get(
        "TWELVEDATA_KEY",
        "",
    ).strip()

    gemini_key = os.environ.get(
        "GEMINI_API_KEY",
        "",
    ).strip()

    supabase_url = os.environ.get(
        "SUPABASE_URL",
        "",
    ).strip()

    service_key = os.environ.get(
        "SUPABASE_SERVICE_KEY",
        "",
    ).strip()

    if not token or not chat_id:
        raise RuntimeError(
            "Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID"
        )

    if mode not in ("scan", "test"):
        mode = "scan"

    try:
        telegram(
            token,
            chat_id,
            "⏳ اسکن XAUUSD شروع شد؛ "
            "موتور ICT/SMC و مدل LightGBM در حال بررسی‌اند.",
        )
    except Exception as exc:
        print(
            f"Scan-start Telegram notification failed: "
            f"{type(exc).__name__}: {str(exc)[:200]}"
        )

    try:
        meta_path = os.path.join(
            MD,
            "meta.json",
        )

        model_path = os.path.join(
            MD,
            "h1_model.txt",
        )

        with open(
            meta_path,
            encoding="utf-8",
        ) as file:
            meta = json.load(file)

        # مدل قدیمی که خانواده‌های مختلف داشته، مجاز نیست.
        trained_families = set(
            meta.get("families", [])
        )

        trained_setups = set(
            meta.get("setups", [])
        )

        if trained_families != {"ICT"}:
            raise RuntimeError(
                "The installed model was not trained on ICT/SMC only. "
                "Run GitHub Actions -> train-model first. "
                f"Model families: {sorted(trained_families)}"
            )

        if (
            not trained_setups
            or not trained_setups.issubset(ALLOWED_SETUPS)
        ):
            raise RuntimeError(
                "Model metadata has an empty or unexpected ICT/SMC setup list. "
                f"Model setups: {sorted(trained_setups)}"
            )

        threshold = float(
            meta["threshold"]
        )

        booster = lgb.Booster(
            model_file=model_path
        )

        raw = fetch_bars(twelve_key)

        now = pd.Timestamp.now(
            tz="UTC"
        ).tz_localize(None)

        closed = raw[
            raw.index + pd.Timedelta(minutes=MINS) <= now
        ]

        if closed.empty:
            raise RuntimeError(
                "No fully closed H1 candles were returned"
            )

        last_close = (
            closed.index[-1]
            + pd.Timedelta(minutes=MINS)
        )

        age_hours = (
            now - last_close
        ).total_seconds() / 3600

        print(
            f"Last closed bar ends {last_close} UTC; "
            f"age={age_hours:.2f}h; candles={len(closed)}"
        )

        if age_hours > 2.5:
            telegram(
                token,
                chat_id,
                "⚠️ اسکن انجام شد، اما دادهٔ آخرین کندل بسته‌شده "
                f"{age_hours:.1f} ساعت قدمت دارد.\n"
                "برای جلوگیری از تحلیل داده کهنه، "
                "ستاپ تازه پیشنهاد نمی‌شود.",
            )
            return

        df, orders = compute_orders(closed)

        # فقط ستاپ‌هایی امتیاز می‌گیرند که مدل قبلاً دیده است.
        if not orders.empty:
            orders = orders[
                orders["family"].eq("ICT")
                & orders["setup"].isin(trained_setups)
            ].copy()

        candle_count = len(df)

        lookback = (
            72 if mode == "test" else 3
        )

        if orders.empty:
            print(
                "No eligible ICT/SMC candidates generated"
            )

            report = (
                "🔎 گزارش اسکن XAUUSD H1\n"
                f"کندل‌های بررسی‌شده: {len(closed)}\n"
                f"آخرین کندل بسته‌شده: {last_close} UTC\n"
                "موتور ICT/SMC هیچ ستاپ آموزش‌دیدهٔ واجد بررسی پیدا نکرد.\n"
                "هیچ معامله‌ای پیشنهاد نمی‌شود."
            )

            try:
                ai = gemini_review(
                    gemini_key,
                    recent_candles(closed),
                    [],
                    threshold,
                    str(last_close),
                )

                report += (
                    "\n\n🤖 بررسی Gemini:\n"
                    + ai
                )

            except Exception as exc:
                report += (
                    "\n\n🤖 بررسی Gemini انجام نشد: "
                    f"{str(exc)[:250]}\n"
                    "گزارش مستقل اسکنر ارسال می‌شود."
                )

            telegram(
                token,
                chat_id,
                report,
            )
            return

        first_bar = max(
            0,
            candle_count - lookback,
        )

        scored = select(
            orders,
            meta,
            booster,
            first_bar,
        )

        if scored.empty:
            telegram(
                token,
                chat_id,
                "🔎 گزارش اسکن XAUUSD H1\n"
                "در پنجره زمانی انتخاب‌شده، "
                "نامزد قابل امتیازدهی پیدا نشد.\n"
                "هیچ معامله‌ای پیشنهاد نمی‌شود.",
            )
            return

        qualified_all = scored[
            scored["score"] >= threshold
        ].copy()

        ranked = scored.head(8).copy()

        qualified = ranked[
            ranked["score"] >= threshold
        ].copy()

        print(
            f"ICT/SMC candidates={len(scored)}; "
            f"above threshold={len(qualified_all)}; "
            f"threshold={threshold:.4f}"
        )

        # فقط ستاپ‌های عبورکرده از آستانه ذخیره می‌شوند.
        if mode != "test":
            persist_candidates(
                supabase_url,
                service_key,
                qualified_all,
                meta,
            )

        candidates = [
            candidate_payload(
                row,
                index + 1,
                threshold,
            )
            for index, (_, row) in enumerate(
                ranked.iterrows()
            )
        ]

        try:
            ai_review = gemini_review(
                gemini_key,
                recent_candles(closed),
                candidates,
                threshold,
                str(last_close),
            )

            ai_status = "موفق"

        except Exception as exc:
            ai_review = (
                "تحلیل Gemini انجام نشد: "
                f"{str(exc)[:250]}\n"
                "نتیجه مستقل اسکنر در ادامه آمده است."
            )

            ai_status = "ناموفق"

            print(
                f"Gemini review failed: "
                f"{type(exc).__name__}: {str(exc)[:250]}"
            )

        report = (
            "🔎 گزارش نهایی اسکن XAUUSD H1\n"
            f"کندل‌های دریافت‌شده: {len(closed)}\n"
            f"آخرین کندل بسته‌شده: {last_close} UTC\n"
            f"نامزدهای ICT/SMC در پنجره: {len(scored)}\n"
            f"نامزدهای بالاتر از آستانه LightGBM "
            f"({threshold:.3f}): {len(qualified_all)}\n"
        )

        if qualified.empty:
            report += (
                "\n❌ هیچ ستاپ ICT/SMC از آستانه مدل عبور نکرد.\n"
                "بنابراین ستاپ قابل‌معامله‌ای معرفی نمی‌شود.\n"
            )

            if not ranked.empty:
                report += (
                    "\nبهترین نامزدهای زیر آستانه "
                    "(صرفاً اطلاعاتی، نه سیگنال):\n"
                )

                for _, row in ranked.head(3).iterrows():
                    report += (
                        f"• ICT/SMC / {row['setup']} — "
                        f"{'BUY' if int(row['dir']) == 1 else 'SELL'} — "
                        f"امتیاز {float(row['score']):.3f}\n"
                    )

        else:
            groups = group_picks(qualified)

            report += (
                "\n✅ ستاپ‌های واجد شرایط ICT/SMC:\n"
            )

            report += "\n\n".join(
                format_candidate(
                    row,
                    threshold,
                )
                for row in groups[:3]
            )

            if len(groups) > 3:
                report += (
                    f"\n\n{len(groups) - 3} گروه ستاپ دیگر "
                    "هم واجد شرایط بود."
                )

        report += (
            f"\n\n🤖 بررسی Gemini ({ai_status}):\n"
            f"{ai_review}\n\n"
            "⚠️ این گزارش تضمین سود نیست؛ "
            "پیش از معامله، قیمت و ریسک را دستی بررسی کن."
        )

        telegram(
            token,
            chat_id,
            report,
        )

        # Tracker اختیاری است و خطای آن نباید گزارش اسکن را حذف کند.
        if (
            mode == "scan"
            and supabase_url
            and service_key
        ):
            try:
                tracked = tracker.update(
                    supabase_url,
                    service_key,
                    closed,
                )

                for item in tracked[:5]:
                    telegram(
                        token,
                        chat_id,
                        tracker.close_message(item),
                    )

                if (
                    now.weekday() == 4
                    and now.hour == 20
                ):
                    telegram(
                        token,
                        chat_id,
                        tracker.weekly_report(
                            supabase_url,
                            service_key,
                        ),
                    )

            except Exception as exc:
                print(
                    f"Tracker error (non-fatal): "
                    f"{type(exc).__name__}: {str(exc)[:300]}"
                )

    except Exception as exc:
        print(
            f"SCAN FAILED: {type(exc).__name__}: "
            f"{str(exc)[:500]}"
        )

        try:
            telegram(
                token,
                chat_id,
                "❌ اجرای اسکن XAUUSD کامل نشد.\n"
                f"نوع خطا: {type(exc).__name__}\n"
                f"شرح: {str(exc)[:400]}\n"
                "گزارش GitHub Actions را بررسی کن.",
            )

        except Exception as notify_exc:
            print(
                f"Failure notification failed: "
                f"{type(notify_exc).__name__}: "
                f"{str(notify_exc)[:150]}"
            )

        raise


if __name__ == "__main__":
    main()
