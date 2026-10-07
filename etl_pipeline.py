"""Nightly address ETL pipeline for the geospatial capstone project."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from shapely.geometry import Point
from sqlalchemy import create_engine, text


LOGGER = logging.getLogger("address_pipeline")

# Accepted coordinate envelope. Deliberately limited to the contiguous United
# States, so Alaska, Hawaii, and territories are rejected by design.
LAT_MIN, LAT_MAX = 25.0, 50.0
LON_MIN, LON_MAX = -130.0, -65.0

# Columns written to current_addresses, in bind-parameter order.
LOAD_COLUMNS = [
    "address_id", "street_number", "street_name", "city", "state", "zip",
    "latitude", "longitude", "geometry",
]

STATE_NAMES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "district of columbia": "DC", "florida": "FL", "georgia": "GA", "hawaii": "HI",
    "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME",
    "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE",
    "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH",
    "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI",
    "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
    "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}

# Full names plus the 2-letter codes themselves, so either form normalizes.
STATE_MAP = {**STATE_NAMES, **{code.lower(): code for code in STATE_NAMES.values()}}

# Screening categories used only for a lightweight operational ESG summary.
# They are not property-level climate-risk determinations.
CLIMATE_SCREEN = {
    "CA": "heat, drought, and wildfire screening",
    "TX": "heat and drought screening",
    "FL": "flood and hurricane screening",
    "NY": "coastal storm and heat screening",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_records(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return valid and rejected records without stopping the full run."""
    required = {"address_id", "latitude", "longitude"}
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Input is missing required columns: {sorted(missing_columns)}")

    working = df.copy()
    working["latitude"] = pd.to_numeric(working["latitude"], errors="coerce")
    working["longitude"] = pd.to_numeric(working["longitude"], errors="coerce")
    # Coerced here so a non-numeric address_id becomes a rejected row rather
    # than an exception that aborts the whole run during transform.
    numeric_ids = pd.to_numeric(working["address_id"], errors="coerce")
    duplicate_mask = numeric_ids.notna() & numeric_ids.duplicated(keep="first")

    def rejection_reason(index: Any, row: pd.Series) -> str:
        reasons: list[str] = []
        if pd.isna(row["address_id"]) or str(row["address_id"]).strip() == "":
            reasons.append("missing address_id")
        elif pd.isna(numeric_ids.loc[index]):
            reasons.append("address_id is not numeric")
        elif duplicate_mask.loc[index]:
            reasons.append("duplicate address_id")
        if pd.isna(row["latitude"]):
            reasons.append("missing latitude")
        elif not LAT_MIN <= row["latitude"] <= LAT_MAX:
            reasons.append(f"latitude outside {LAT_MIN:g} to {LAT_MAX:g}")
        if pd.isna(row["longitude"]):
            reasons.append("missing longitude")
        elif not LON_MIN <= row["longitude"] <= LON_MAX:
            reasons.append(f"longitude outside {LON_MIN:g} to {LON_MAX:g}")
        return "; ".join(reasons)

    working["rejection_reason"] = [
        rejection_reason(index, row) for index, row in working.iterrows()
    ]
    rejected = working[working["rejection_reason"] != ""].copy()
    valid = working[working["rejection_reason"] == ""].drop(
        columns=["rejection_reason"]
    ).copy()
    return valid, rejected


def normalize_state(value: Any) -> str | None:
    """Map a state name to its 2-letter code, or None when not resolvable."""
    if pd.isna(value):
        return None
    cleaned = str(value).strip()
    code = STATE_MAP.get(cleaned.lower())
    if code is None:
        LOGGER.warning("Unrecognized state value %r stored as NULL", cleaned)
    return code


def transform_records(df: pd.DataFrame) -> pd.DataFrame:
    """Trim text, standardize state values, and create EPSG:4326 points."""
    transformed = df.copy()
    for column in transformed.select_dtypes(include=["object", "str"]).columns:
        transformed[column] = transformed[column].str.strip()

    if "state" not in transformed.columns:
        transformed["state"] = None
    transformed["state"] = transformed["state"].apply(normalize_state)
    transformed["address_id"] = pd.to_numeric(
        transformed["address_id"], errors="coerce"
    ).astype("int64")
    transformed["geometry"] = transformed.apply(
        lambda row: Point(row["longitude"], row["latitude"]).wkt, axis=1
    )
    transformed["crs"] = "EPSG:4326"
    return transformed


def build_load_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Reduce the frame to bind parameters with real NULLs instead of NaN/NA."""
    payload = df.reindex(columns=LOAD_COLUMNS)
    payload = payload.astype(object).where(payload.notna(), None)
    records = payload.to_dict(orient="records")
    for record in records:
        record["address_id"] = int(record["address_id"])
    return records


def load_to_postgis(df: pd.DataFrame, connection_string: str) -> dict[str, int]:
    """Create current_addresses and upsert records by address_id."""
    engine = create_engine(connection_string, future=True)
    create_sql = text("""
        CREATE EXTENSION IF NOT EXISTS postgis;
        CREATE TABLE IF NOT EXISTS current_addresses (
            address_id BIGINT PRIMARY KEY,
            street_number TEXT,
            street_name TEXT,
            city TEXT,
            state VARCHAR(2),
            zip TEXT,
            latitude DOUBLE PRECISION,
            longitude DOUBLE PRECISION,
            geometry geometry(Point, 4326),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_current_addresses_geometry
            ON current_addresses USING GIST (geometry);
    """)
    # Rows land in a staging table first so the upsert is a single statement
    # and RETURNING can report exact insert/update counts without a race.
    stage_sql = text("""
        CREATE TEMP TABLE staging_addresses (
            address_id BIGINT,
            street_number TEXT,
            street_name TEXT,
            city TEXT,
            state TEXT,
            zip TEXT,
            latitude DOUBLE PRECISION,
            longitude DOUBLE PRECISION,
            geometry TEXT
        ) ON COMMIT DROP;
    """)
    stage_insert_sql = text("""
        INSERT INTO staging_addresses (
            address_id, street_number, street_name, city, state, zip,
            latitude, longitude, geometry
        ) VALUES (
            :address_id, :street_number, :street_name, :city, :state, :zip,
            :latitude, :longitude, :geometry
        );
    """)
    upsert_sql = text("""
        INSERT INTO current_addresses (
            address_id, street_number, street_name, city, state, zip,
            latitude, longitude, geometry, updated_at
        )
        SELECT DISTINCT ON (address_id)
            address_id, street_number, street_name, city, state, zip,
            latitude, longitude,
            ST_SetSRID(ST_GeomFromText(geometry), 4326), NOW()
        FROM staging_addresses
        ORDER BY address_id
        ON CONFLICT (address_id) DO UPDATE SET
            street_number = EXCLUDED.street_number,
            street_name = EXCLUDED.street_name,
            city = EXCLUDED.city,
            state = EXCLUDED.state,
            zip = EXCLUDED.zip,
            latitude = EXCLUDED.latitude,
            longitude = EXCLUDED.longitude,
            geometry = EXCLUDED.geometry,
            updated_at = NOW()
        RETURNING (xmax = 0) AS was_inserted;
    """)

    records = build_load_records(df)
    if not records:
        LOGGER.warning("No valid records to load; skipping database write")
        return {"inserted": 0, "updated": 0}

    with engine.begin() as connection:
        connection.execute(create_sql)
        connection.execute(stage_sql)
        connection.execute(stage_insert_sql, records)
        flags = list(connection.execute(upsert_sql).scalars())

    inserted = sum(1 for flag in flags if flag)
    return {"inserted": inserted, "updated": len(flags) - inserted}


def make_alert_event(alert_type: str, reason: str, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_type": "pipeline_alert",
        "alert_type": alert_type,
        "timestamp_utc": utc_now(),
        "pipeline": "address_update_pipeline",
        "reason": reason,
        "details": details,
    }


def publish_event(event: dict[str, Any], output_path: str | Path) -> None:
    """Simulate publishing to a queue by appending a JSON event record."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, default=str) + "\n")
    LOGGER.warning("Published alert event: %s", event)


def emit_alert_event(
    summary_dict: dict[str, Any],
    output_path: str | Path,
    rejection_threshold: float = 0.20,
) -> list[dict[str, Any]]:
    """Publish explicit alert events for operational threshold breaches."""
    events: list[dict[str, Any]] = []
    processed = int(summary_dict.get("processed", 0))
    rejected = int(summary_dict.get("rejected", 0))
    rejection_rate = rejected / processed if processed else 0.0

    if rejection_rate > rejection_threshold:
        events.append(make_alert_event(
            "high_rejection_rate",
            f"Rejection rate {rejection_rate:.1%} exceeded {rejection_threshold:.1%}",
            {"processed": processed, "rejected": rejected, "rejection_rate": rejection_rate},
        ))

    for event in events:
        publish_event(event, output_path)
    return events


def record_missing_file_day(
    input_path: str | Path,
    state_path: str | Path,
    alert_path: str | Path,
) -> bool:
    """Track consecutive missing-file days and alert on the second day."""
    input_file = Path(input_path)
    state_file = Path(state_path)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state = {"consecutive_missing_days": 0, "last_check_date": None}
    if state_file.exists():
        state.update(json.loads(state_file.read_text(encoding="utf-8")))

    today = datetime.now(timezone.utc).date().isoformat()
    if input_file.exists():
        state = {"consecutive_missing_days": 0, "last_check_date": today}
        exists = True
    else:
        if state.get("last_check_date") != today:
            state["consecutive_missing_days"] = int(state.get("consecutive_missing_days", 0)) + 1
        state["last_check_date"] = today
        exists = False
        if state["consecutive_missing_days"] >= 2:
            publish_event(make_alert_event(
                "missing_file_two_days",
                "Input file was missing for 2 consecutive daily checks",
                {"input_path": str(input_file), **state},
            ), alert_path)

    state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return exists


def write_climate_esg_summary(df: pd.DataFrame, output_path: str | Path) -> dict[str, Any]:
    """Write a lightweight, evidence-qualified climate screening summary."""
    screened = df.copy()
    screened["screening_category"] = screened["state"].map(CLIMATE_SCREEN).fillna(
        "no configured screening category"
    )
    counts = screened["screening_category"].value_counts().to_dict()
    flagged = int((screened["screening_category"] != "no configured screening category").sum())
    total = len(screened)

    lines = [
        "Climate / ESG Operational Screening Summary",
        f"Generated: {utc_now()}",
        f"Valid addresses assessed: {total}",
        f"Addresses in configured screening categories: {flagged}",
        "Category counts:",
    ]
    lines.extend(f"- {name}: {count}" for name, count in counts.items())
    lines.extend([
        "",
        "This screening identifies addresses that may merit additional service-continuity review based on broad state-level categories.",
        "It can help prioritize follow-up analysis and resilience planning, but it is not a property-level risk assessment or a prediction of future events.",
    ])
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"valid_addresses": total, "flagged_addresses": flagged, "category_counts": counts}


def write_log_summary(
    summary_dict: dict[str, Any],
    output_path: str | Path,
    status: str = "success",
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"timestamp_utc={utc_now()}",
        "pipeline=address_update_pipeline",
        f"status={status}",
    ]
    lines.extend(f"{key}={value}" for key, value in summary_dict.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_data(
    input_path: str | Path,
    work_dir: str | Path,
    rejected_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Read, validate, transform, and persist intermediate files."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    # address_id is read as text so a blank cell cannot coerce the column to
    # float and render valid ids as "1012.0" in the rejected report.
    raw = pd.read_csv(input_path, dtype={"zip": "string", "address_id": "string"})
    valid, rejected = validate_records(raw)
    transformed = transform_records(valid)
    valid_path = work / "valid_addresses.csv"
    rejected_folder = Path(rejected_dir) if rejected_dir else work
    rejected_folder.mkdir(parents=True, exist_ok=True)
    rejected_path = rejected_folder / "rejected_addresses.csv"
    transformed.to_csv(valid_path, index=False)
    rejected.to_csv(rejected_path, index=False)
    return {
        "processed": len(raw), "valid": len(transformed), "rejected": len(rejected),
        "valid_path": str(valid_path), "rejected_path": str(rejected_path),
    }


def run_pipeline(
    input_path: str,
    output_dir: str,
    connection_string: str,
    rejected_dir: str | None = None,
    alerts_dir: str | None = None,
    logs_dir: str | None = None,
) -> dict[str, Any]:
    """Run all ETL stages from the command line."""
    output = Path(output_dir)
    rejected_folder = Path(rejected_dir) if rejected_dir else output
    alerts_folder = Path(alerts_dir) if alerts_dir else output
    logs_folder = Path(logs_dir) if logs_dir else output
    if not record_missing_file_day(
        input_path,
        logs_folder / "missing_file_state.json",
        alerts_folder / "alert_events.jsonl",
    ):
        raise FileNotFoundError(f"Expected input file not found: {input_path}")

    summary: dict[str, Any] = {}
    try:
        summary = prepare_data(input_path, output, rejected_folder)
        valid_df = pd.read_csv(summary["valid_path"], dtype={"zip": "string"})
        summary.update(load_to_postgis(valid_df, connection_string))
        emit_alert_event(summary, alerts_folder / "alert_events.jsonl")
        write_climate_esg_summary(valid_df, output / "climate_esg_summary.txt")
    except Exception as error:
        write_log_summary(
            {**summary, "error": repr(error)},
            logs_folder / "pipeline_logs.txt",
            status="failed",
        )
        raise
    write_log_summary(summary, logs_folder / "pipeline_logs.txt")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--rejected-dir", default="rejected")
    parser.add_argument("--alerts-dir", default="alerts")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    args = parser.parse_args()
    if not args.database_url:
        raise ValueError("Set DATABASE_URL or pass --database-url")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = run_pipeline(
        args.input,
        args.output_dir,
        args.database_url,
        args.rejected_dir,
        args.alerts_dir,
        args.logs_dir,
    )
    LOGGER.info("ETL pipeline run complete: %s", summary)


if __name__ == "__main__":
    main()
