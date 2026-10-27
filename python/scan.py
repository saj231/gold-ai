
import os
import sys
import json
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb

import smc_engine as se
import common as cm
import fam_trend
import fam_pa
import fam_rtm
import tracker
from build_multi import ICT_EXTRA


MINS = 60
SYMBOL = "XAU/USD"
MD = os.environ.get("MODEL_DIR", "../model")

FAM_FA = {
    "ICT": "ICT/SMC",
    "PA": "پرایس اکشن",
    "RTM": "RTM",
    "TREND": "ترند/مومنتوم",
}


def http_json(url, data=None, headers=None, timeout=40):
    body = (
        json.dumps(data, ensure_ascii=False).encode("utf-8")
        if data is not None
        else None
    )
    request_headers = {"User-Agent": "gold-ai-scanner/1.0"}
    request_headers.update(headers or {})

    request = urllib.request.Request(
        url,
        data=body,
        headers=request_headers,
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        content = response.read().decode("utf-8", errors="replace")

    return json.loads(content) if content else None


def telegram(token, chat, text):
    if not token or not chat:
        raise RuntimeError(
            "Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID"
        )

    text = str(text)
    chunks = []

    while len(text) > 3900:
        split_at = text.rfind("\n", 0, 3900)

        if split_at < 1000:
            split_at = 3900

        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()

    if text:
        chunks.append(text)

    for chunk in chunks:
        result = http_json(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={
                "chat_id": chat,
                "text": chunk,
                "disable_web_page_preview": True,
            },
            headers={"Content-Type": "application/json"},
            timeout=20,
        )

        if not isinstance(result, dict) or result.get("ok") is not True:
            description = (
                result.get("description", "Unknown Telegram error")
                if isinstance(result, dict)
                else "Invalid Telegram response"
            )
            raise RuntimeError(
                f"Telegram sendMessage failed: {description}"
            )


def fetch_bars(key, count=5000):
    if not key:
        raise RuntimeError("TWELVEDATA_KEY is missing")

    query = urllib.parse.urlencode({
        "symbol": SYMBOL,
        "interval": "1h",
        "outputsize": count,
        "timezone": "UTC",
        "order": "ASC",
        "apikey": key,
    })

    result = http_json(
        "https://api.twelvedata.com/time_series?" + query
    )

    if not isinstance(result, dict) or "values" not in result:
        message = (
            result.get("message", "Unknown Twelve Data error")
            if isinstance(result, dict)
            else "Invalid response"
        )
        raise RuntimeError(f"Twelve Data failed: {message}")

    df = pd.DataFrame(result["values"])

    if df.empty:
        raise RuntimeError("Twelve Data returned no candles")

    df["datetime"] = pd.to_datetime(
        df["datetime"], utc=True
    ).dt.tz_localize(None)

    df = df.set_index("datetime").sort_index()

    for column in ("open", "high", "low", "close"):
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df = df[["open", "high", "low", "close"]]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[~df.index.duplicated(keep="last")]

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

    parts = []

    ict = se.detect(df)

    if len(ict):
        orders = pd.DataFrame({
            "bar": ict["bar"],
            "family": "ICT",
            "setup": ict["setup"],
            "dir": ict["dir"],
            "mode": "limit",
            "limit": ict["limit"],
            "stop": ict["stop"],
            "ref": ict["limit"],
        })

        for column in ICT_EXTRA:
            orders[column] = ict[column].values

        parts.append(orders)

    for family_module in (fam_trend, fam_pa, fam_rtm):
        orders_builder = cm.Orders(df)
        family_module.run(orders_builder, df)
        family_orders = orders_builder.frame()

        if not family_orders.empty:
            parts.append(family_orders)

    parts = [part for part in parts if not part.empty]

    if not parts:
        return df, pd.DataFrame()

    orders = pd.concat(parts, ignore_index=True)
    orders = cm.add_features(df, orders, MINS)

    return df, orders


def make_X(orders, meta):
    X = pd.DataFrame(index=orders.index)

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
                orders["family"].map(family_codes).fillna(-1)
            )
        elif column == "setup_code":
            X[column] = (
                orders["setup"].map(setup_codes).fillna(-1)
            )
        elif column in orders.columns:
            X[column] = orders[column]
        else:
            X[column] = np.nan

    return X.astype(float)


def score_orders(orders, meta, booster, last_bar):
    X = make_X(orders, meta)

    scored = orders.copy()
    scored["score"] = booster.predict(
        X[meta["features"]]
    )

    scored = scored.replace(
        [np.inf, -np.inf], np.nan, subset=None
    )

    scored = scored[
        (scored["bar"] >= last_bar)
        & scored["score"].notna()
    ]

    return scored.sort_values(
        "score", ascending=False
    ).copy()


def safe_float(value, digits=3):
    try:
        number = float(value)

        if not np.isfinite(number):
            return None

        return round(number, digits)
    except (TypeError, ValueError):
        return None


def recent_candles(df, count=18):
    result = []

    for timestamp, row in df.tail(count).iterrows():
        result.append({
            "time_utc": pd.Timestamp(timestamp).strftime(
                "%Y-%m-%d %H:%M"
            ),
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
        "family": FAM_FA.get(
            str(row["family"]), str(row["family"])
        ),
        "setup": str(row["setup"]),
        "direction": "BUY" if direction == 1 else "SELL",
        "entry_type": str(row["mode"]),
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "tp_1_5R": round(entry + direction * 1.5 * risk, 2),
        "tp_2R": round(entry + direction * 2.0 * risk, 2),
        "tp_3R": round(entry + direction * 3.0 * risk, 2),
        "lightgbm_score": round(float(row["score"]), 4),
        "lightgbm_threshold": round(float(threshold), 4),
        "passed_lightgbm_threshold": bool(
            float(row["score"]) >= threshold
        ),
        "bar_time_utc": pd.Timestamp(
            row["time"]
        ).strftime("%Y-%m-%d %H:%M"),
        "risk_atr": safe_float(row.get("risk_atr")),
        "with_trend": safe_float(row.get("with_trend"), 0),
    }


def gemini_review(api_key, candles, candidates, threshold, latest_close):
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing from GitHub Actions secrets"
        )

    model = (
        os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
        or "gemini-2.5-flash"
    )

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + urllib.parse.quote(model, safe="-.")
        + ":generateContent?key="
        + urllib.parse.quote(api_key)
    )

    system_instruction = (
        "تو تحلیل‌گر کمکی بازار XAUUSD هستی و باید فارسی پاسخ بدهی. "
        "فقط از داده‌ها و نامزدهای ارائه‌شده استفاده کن. "
        "قیمت، خبر، اندیکاتور یا ستاپ جدید جعل نکن. "
        "امتیاز LightGBM را احتمال قطعی برد معرفی نکن. "
        "ستاپ واجد شرایط فقط نامزدی است که "
        "passed_lightgbm_threshold=true داشته باشد. "
        "اگر چنین نامزدی وجود ندارد، صریحاً بگو ستاپ واجد شرایطی "
        "برای پیشنهاد وجود ندارد و معامله‌ای پیشنهاد نکن. "
        "اگر نامزد واجد شرایط وجود دارد، بهترین را فقط از میان "
        "همان نامزدها انتخاب کن و دلایل، ریسک‌ها و عدم قطعیت "
        "را توضیح بده. اطلاعات ورودی ممکن است قیمت لحظه‌ای نباشد."
    )

    payload = {
        "task": (
            "کندل‌های اخیر H1 و خروجی واقعی موتورهای قوانین و LightGBM "
            "را مقایسه کن. نامزدهای واجد شرایط را از نظر جهت، محل ورود، "
            "حد ضرر، اهداف و هم‌راستایی با شواهد موجود بررسی کن. "
            "اگر هیچ نامزدی از آستانه عبور نکرده، سیگنال نساز. "
            "اگر شواهد برای رتبه‌بندی کافی نیست، این محدودیت را اعلام کن."
        ),
        "latest_closed_bar_end_utc": latest_close,
        "lightgbm_threshold": threshold,
        "candles_ohlc": candles,
        "candidates": candidates,
    }

    body = {
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        },
        "contents": [{
            "role": "user",
            "parts": [{
                "text": json.dumps(
                    payload, ensure_ascii=False
                )
            }],
        }],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 700,
        },
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response_text = response.read().decode(
                "utf-8", errors="replace"
            )

    except urllib.error.HTTPError as exc:
        details = exc.read(1200).decode(
            "utf-8", errors="replace"
        )

        try:
            parsed = json.loads(details)
            details = parsed.get("error", {}).get(
                "message", details
            )
        except (ValueError, TypeError):
            pass

        # Never print the request URL or API key.
        print(
            f"Gemini HTTP {exc.code}: {str(details)[:500]}"
        )
        raise RuntimeError(
            f"Gemini API returned HTTP {exc.code}: "
            f"{str(details)[:250]}"
        ) from None

    except Exception as exc:
        print(
            f"Gemini network error: {type(exc).__name__}: "
            f"{str(exc)[:250]}"
        )
        raise RuntimeError(
            f"Gemini network error ({type(exc).__name__})"
        ) from None

    try:
        data = json.loads(response_text)
        answer = "\n".join(
            part.get("text", "")
            for part in data["candidates"][0]["content"]["parts"]
            if isinstance(part, dict)
        ).strip()
    except (ValueError, KeyError, IndexError, TypeError):
        print("Gemini returned an unrecognized response")
        raise RuntimeError(
            "Gemini response could not be parsed"
        ) from None

    if not answer:
        raise RuntimeError("Gemini returned an empty response")

    return answer[:2500]


def format_candidate(row, threshold, tag=""):
    direction = int(row["dir"])
    entry = float(row["ref"])
    stop = float(row["stop"])
    risk = abs(entry - stop)

    side = "🟢 خرید (BUY)" if direction == 1 else "🔴 فروش (SELL)"
    family = FAM_FA.get(
        str(row["family"]), str(row["family"])
    )

    if str(row["mode"]) == "limit":
        entry_text = f"لیمیت روی {entry:.2f}"
    else:
        entry_text = f"ورود بازار نزدیک {entry:.2f}"

    timestamp = pd.Timestamp(row["time"]).strftime(
        "%Y-%m-%d %H:%M"
    )

    return (
        f"{tag}📊 نامزد ستاپ XAUUSD H1\n"
        f"{side}\n"
        f"سبک: {family}\n"
        f"قانون/ستاپ: {row['setup']}\n"
        f"ورود پیشنهادی موتور: {entry_text}\n"
        f"حد ضرر: {stop:.2f}\n"
        f"ریسک قیمتی: {risk:.2f}\n"
        f"اهداف محاسباتی: "
        f"1.5R={entry + direction * 1.5 * risk:.2f} | "
        f"2R={entry + direction * 2 * risk:.2f} | "
        f"3R={entry + direction * 3 * risk:.2f}\n"
        f"امتیاز LightGBM: {float(row['score']):.3f}\n"
        f"آستانه مدل: {threshold:.3f}\n"
        f"زمان کندل: {timestamp} UTC\n"
        "این خروجی تحلیلی است؛ قبل از هر تصمیم، داده و ریسک را بررسی کن."
    )


def persist_candidates(supabase_url, service_key, picks, meta):
    if not supabase_url or not service_key or picks.empty:
        return

    rows = []

    for _, row in picks.iterrows():
        direction = int(row["dir"])
        entry = float(row["ref"])
        stop = float(row["stop"])
        risk = abs(entry - stop)

        rows.append({
            "bar_time": pd.Timestamp(row["time"]).isoformat() + "Z",
            "symbol": "XAUUSD",
            "setup": str(row["setup"]),
            "direction": "BUY" if direction == 1 else "SELL",
            "entry": entry,
            "stop": stop,
            "tp1": entry + direction * 1.5 * risk,
            "tp2": entry + direction * 2.0 * risk,
            "tp3": entry + direction * 3.0 * risk,
            "probability": float(row["score"]),
            "model_version": meta.get("version", 1),
            "mode": str(row["mode"]),
            "family": str(row["family"]),
        })

    headers = {
        "apikey": service_key,
        "Authorization": "Bearer " + service_key,
        "Content-Type": "application/json",
        "Prefer": "resolution=ignore-duplicates,return=minimal",
    }

    endpoint = (
        supabase_url.rstrip("/")
        + "/rest/v1/signals"
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
        # Database errors must not suppress Telegram reports.
        print(
            f"Supabase persistence failed: {type(exc).__name__}: "
            f"{str(exc)[:300]}"
        )


def main():
    mode = os.environ.get("MODE", "scan").strip().lower()
    token = os.environ.get("TELEGRAM_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    twelve_key = os.environ.get("TWELVEDATA_KEY", "").strip()
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    supabase_url = os.environ.get("SUPABASE_URL", "").strip()
    service_key = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()

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
            "⏳ اسکن XAUUSD H1 شروع شد.\n"
            "در حال دریافت داده، اجرای موتورهای ستاپ و ارزیابی LightGBM هستم.",
        )
    except Exception as exc:
        print(
            f"Scan-start Telegram notification failed: "
            f"{type(exc).__name__}: {str(exc)[:250]}"
        )

    try:
        if not twelve_key:
            raise RuntimeError("TWELVEDATA_KEY is missing")

        meta_path = os.path.join(MD, "meta.json")
        model_path = os.path.join(MD, "h1_model.txt")

        with open(meta_path, encoding="utf-8") as file:
            meta = json.load(file)

        threshold = float(meta["threshold"])
        booster = lgb.Booster(model_file=model_path)

        raw = fetch_bars(twelve_key)

        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        closed = raw[
            raw.index + pd.Timedelta(minutes=MINS) <= now
        ]

        if closed.empty:
            raise RuntimeError(
                "No fully closed H1 candles were returned"
            )

        last_close = closed.index[-1] + pd.Timedelta(minutes=MINS)
        age_hours = (now - last_close).total_seconds() / 3600

        print(
            f"Last closed bar ends {last_close} UTC; "
            f"age {age_hours:.2f}h; bars {len(closed)}"
        )

        if age_hours > 2.5:
            telegram(
                token,
                chat_id,
                "⚠️ گزارش اسکن XAUUSD\n"
                f"آخرین کندل بسته‌شده {age_hours:.1f} ساعت قدمت دارد.\n"
                "برای جلوگیری از تحلیل داده کهنه، ستاپ تازه پیشنهاد نمی‌شود.",
            )
            return

        df, orders = compute_orders(closed)

        if orders.empty:
            print("No rule-engine candidates generated")

            report = (
                "🔎 گزارش نهایی اسکن XAUUSD H1\n"
                f"کندل‌های معتبر: {len(closed)}\n"
                f"آخرین کندل بسته‌شده: {last_close} UTC\n"
                "نتیجه: موتورهای ICT/SMC، پرایس اکشن، RTM و ترند "
                "هیچ نامزدی تولید نکردند.\n"
                "این به معنی نبود قطعی فرصت در بازار نیست؛ "
                "فقط یعنی این اسکن نامزد معتبری پیدا نکرد."
            )

            try:
                review = gemini_review(
                    gemini_key,
                    recent_candles(closed),
                    [],
                    threshold,
                    str(last_close),
                )
                report += "\n\n🤖 بررسی Gemini:\n" + review
            except Exception as exc:
                report += (
                    "\n\n🤖 تحلیل Gemini ناموفق بود: "
                    f"{str(exc)[:300]}\n"
                    "گزارش اسکنر بدون تحلیل Gemini ارسال می‌شود."
                )

            telegram(token, chat_id, report)
            return

        # Normal scan checks the latest 3 closed bars.
        # Test mode inspects a wider 72-bar window for diagnostics.
        lookback = 72 if mode == "test" else 3
        last_bar_index = max(0, len(df) - lookback)

        scored = orders.copy()
        features = make_X(scored, meta)

        scored["score"] = booster.predict(
            features[meta["features"]]
        )

        scored = scored[
            (scored["bar"] >= last_bar_index)
            & scored["score"].notna()
        ].sort_values("score", ascending=False)

        if scored.empty:
            print("No scored candidates in the requested window")

            telegram(
                token,
                chat_id,
                "🔎 گزارش نهایی اسکن XAUUSD H1\n"
                f"نامزدهای قوانین در داده‌ها: {len(orders)}\n"
                "در پنجره زمانی انتخاب‌شده نامزد قابل امتیازدهی پیدا نشد.\n"
                "هیچ سیگنال معاملاتی پیشنهاد نمی‌شود.",
            )
            return

        qualified = scored[
            scored["score"] >= threshold
        ].copy()

        print(
            f"Orders in window: {len(scored)}; "
            f"above threshold: {len(qualified)}; "
            f"threshold: {threshold:.4f}"
        )

        # Gemini receives actual rule-engine candidates and their model scores.
        # It must not invent a candidate when none passes the threshold.
        candidates_for_review = scored.head(8)
        candidate_data = [
            candidate_payload(row, index + 1, threshold)
            for index, (_, row) in enumerate(
                candidates_for_review.iterrows()
            )
        ]

        try:
            ai_review = gemini_review(
                gemini_key,
                recent_candles(closed),
                candidate_data,
                threshold,
                str(last_close),
            )
            gemini_status = "موفق"
        except Exception as exc:
            ai_review = (
                "تحلیل Gemini انجام نشد: "
                f"{str(exc)[:300]}\n"
                "نتیجه مستقل اسکنر در ادامه آمده است."
            )
            gemini_status = "ناموفق"
            print(
                f"Gemini review failed: {type(exc).__name__}: "
                f"{str(exc)[:400]}"
            )

        if not qualified.empty:
            persist_candidates(
                supabase_url,
                service_key,
                qualified,
                meta,
            )

        report = (
            "🔎 گزارش نهایی اسکن XAUUSD H1\n"
            f"کندل‌های دریافت‌شده: {len(closed)}\n"
            f"آخرین کندل بسته‌شده: {last_close} UTC\n"
            f"نامزدهای قوانین در پنجره: {len(scored)}\n"
            f"نامزدهای بالاتر از آستانه LightGBM "
            f"({threshold:.3f}): {len(qualified)}\n"
        )

        if qualified.empty:
            report += (
                "\n❌ هیچ ستاپی از آستانه مدل عبور نکرد.\n"
                "بنابراین ستاپ قابل‌معامله‌ای معرفی نمی‌شود.\n"
            )

            report += "\nبهترین نامزدهای زیر آستانه (صرفاً اطلاعاتی):\n"

            for _, row in scored.head(3).iterrows():
                family = FAM_FA.get(
                    str(row["family"]), str(row["family"])
                )
                side = "BUY" if int(row["dir"]) == 1 else "SELL"

                report += (
                    f"• {family} / {row['setup']} — "
                    f"{side} — امتیاز {float(row['score']):.3f}\n"
                )
        else:
            report += "\n✅ ستاپ‌های واجد شرایط LightGBM:\n"

            for _, row in qualified.head(3).iterrows():
                report += "\n" + format_candidate(
                    row, threshold
                ) + "\n"

            if len(qualified) > 3:
                report += (
                    f"\nتعداد {len(qualified) - 3} نامزد واجد شرایط دیگر "
                    "وجود دارد که در این پیام نمایش داده نشده‌اند.\n"
                )

        report += (
            f"\n\n🤖 بررسی Gemini ({gemini_status}):\n"
            f"{ai_review}\n\n"
            "⚠️ این گزارش ابزار تحلیلی است، نه تضمین سود. "
            "پیش از معامله، قیمت، ورود، حد ضرر و ریسک را دستی بررسی کن."
        )

        telegram(token, chat_id, report)

        # Tracker is optional. A tracking error must not hide scan results.
        if mode == "scan" and supabase_url and service_key:
            try:
                tracked = tracker.update(
                    supabase_url, service_key, closed
                )

                for item in tracked[:5]:
                    telegram(
                        token,
                        chat_id,
                        tracker.close_message(item),
                    )

                if now.weekday() == 4 and now.hour == 20:
                    telegram(
                        token,
                        chat_id,
                        tracker.weekly_report(
                            supabase_url, service_key
                        ),
                    )

            except Exception as exc:
                print(
                    f"Tracker error (non-fatal): {type(exc).__name__}: "
                    f"{str(exc)[:350]}"
                )

    except Exception as exc:
        print(
            f"SCAN FAILED: {type(exc).__name__}: {str(exc)[:600]}"
        )

        try:
            telegram(
                token,
                chat_id,
                "❌ اجرای اسکن XAUUSD کامل نشد.\n"
                f"نوع خطا: {type(exc).__name__}\n"
                f"شرح: {str(exc)[:500]}\n"
                "جزئیات اجرا را در GitHub Actions بررسی کن.",
            )
        except Exception as notify_exc:
            print(
                "Failure notification also failed: "
                f"{type(notify_exc).__name__}: "
                f"{str(notify_exc)[:200]}"
            )

        raise


if __name__ == "__main__":
    main()
