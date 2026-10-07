"""Airflow DAG for the nightly address update pipeline."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator


PROJECT_DIR = Path(os.getenv("CAPSTONE_PROJECT_DIR", str(Path.home() / "geospatial-capstone")))
sys.path.insert(0, str(PROJECT_DIR))

from etl_pipeline import (  # noqa: E402
    emit_alert_event,
    load_to_postgis,
    make_alert_event,
    prepare_data,
    publish_event,
    record_missing_file_day,
    write_climate_esg_summary,
    write_log_summary,
)

INPUT_FILE = Path(os.getenv("CAPSTONE_INPUT_FILE", str(PROJECT_DIR / "input" / "addresses_sample.csv")))
OUTPUT_DIR = Path(os.getenv("CAPSTONE_OUTPUT_DIR", str(PROJECT_DIR / "output")))
REJECTED_DIR = Path(os.getenv("CAPSTONE_REJECTED_DIR", str(PROJECT_DIR / "rejected")))
ALERTS_DIR = Path(os.getenv("CAPSTONE_ALERTS_DIR", str(PROJECT_DIR / "alerts")))
LOGS_DIR = Path(os.getenv("CAPSTONE_LOGS_DIR", str(PROJECT_DIR / "logs")))
# Credentials are never hardcoded. Supply the connection string through the
# DATABASE_URL environment variable, an Airflow Connection, or a secrets
# manager. Expected format:
#   postgresql+psycopg2://<user>:<password>@<host>:<port>/<database>
DATABASE_URL = os.getenv("DATABASE_URL")


def final_failure_callback(context):
    """Publish a failure event only after the task exhausts its retries."""
    task_instance = context["task_instance"]
    try_number = getattr(task_instance, "try_number", 1) or 1
    max_tries = getattr(task_instance, "max_tries", 0) or 0
    if try_number > max_tries:
        event = make_alert_event(
            "failure_after_retries",
            f"Task {task_instance.task_id} failed after all retry attempts",
            {"dag_id": task_instance.dag_id, "task_id": task_instance.task_id,
             "run_id": task_instance.run_id},
        )
        publish_event(event, ALERTS_DIR / "alert_events.jsonl")


default_args = {
    "owner": "airflow",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": final_failure_callback,
}


def check_for_file(**context):
    exists = record_missing_file_day(
        INPUT_FILE,
        LOGS_DIR / "missing_file_state.json",
        ALERTS_DIR / "alert_events.jsonl",
    )
    if not exists:
        raise FileNotFoundError(f"Expected input file not found: {INPUT_FILE}")
    return str(INPUT_FILE)


def validate_and_transform(**context):
    input_path = context["ti"].xcom_pull(task_ids="check_for_file")
    return prepare_data(input_path, OUTPUT_DIR, REJECTED_DIR)


def load_to_database(**context):
    if not DATABASE_URL:
        raise ValueError(
            "DATABASE_URL is not set. Export it in the scheduler environment or "
            "configure it as an Airflow Connection before running this DAG."
        )
    summary = context["ti"].xcom_pull(task_ids="validate_and_transform")
    valid_df = pd.read_csv(summary["valid_path"], dtype={"zip": "string"})
    return load_to_postgis(valid_df, DATABASE_URL)


def emit_alert_status(**context):
    summary = dict(context["ti"].xcom_pull(task_ids="validate_and_transform"))
    summary.update(context["ti"].xcom_pull(task_ids="load_to_postgis"))
    events = emit_alert_event(summary, ALERTS_DIR / "alert_events.jsonl")
    return {"alert_count": len(events), **summary}


def generate_climate_summary(**context):
    summary = context["ti"].xcom_pull(task_ids="validate_and_transform")
    valid_df = pd.read_csv(summary["valid_path"], dtype={"zip": "string"})
    return write_climate_esg_summary(valid_df, OUTPUT_DIR / "climate_esg_summary.txt")


def log_summary(**context):
    summary = context["ti"].xcom_pull(task_ids="emit_alert_status")
    write_log_summary(summary, LOGS_DIR / "pipeline_logs.txt")
    return summary


with DAG(
    dag_id="address_update_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule="0 2 * * *",
    catchup=False,
    default_args=default_args,
    tags=["geospatial", "capstone"],
) as dag:
    check_task = PythonOperator(task_id="check_for_file", python_callable=check_for_file)
    transform_task = PythonOperator(task_id="validate_and_transform", python_callable=validate_and_transform)
    load_task = PythonOperator(task_id="load_to_postgis", python_callable=load_to_database)
    alert_task = PythonOperator(task_id="emit_alert_status", python_callable=emit_alert_status)
    climate_task = PythonOperator(task_id="generate_climate_summary", python_callable=generate_climate_summary)
    log_task = PythonOperator(task_id="log_summary", python_callable=log_summary)

    check_task >> transform_task >> load_task >> alert_task >> climate_task >> log_task
