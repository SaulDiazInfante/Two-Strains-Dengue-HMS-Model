"""Streamlit app for interactive plotting of timestamp-indexed time series."""

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


INDEX_SENTINEL = "<Use dataframe index>"
MAX_DEFAULT_COLUMNS = 5


def parse_app_args() -> argparse.Namespace:
    """Parse app arguments passed after ``streamlit run ... --``."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--index-column", type=str, default=None)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


@st.cache_data(show_spinner=False)
def load_csv_frame(path: Path) -> pd.DataFrame:
    """Load a CSV file into a dataframe."""
    return pd.read_csv(path)


def infer_timestamp_column(frame: pd.DataFrame) -> str | None:
    """Return a likely timestamp column when one can be inferred."""
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            continue
        parsed = pd.to_datetime(frame[column], errors="coerce")
        valid_ratio = parsed.notna().mean()
        if valid_ratio >= 0.8:
            return column
    return None


def build_timestamp_indexed_frame(
    frame: pd.DataFrame,
    index_column: str | None = None,
) -> pd.DataFrame:
    """Normalize a dataframe so it has a datetime index for plotting."""
    normalized = frame.copy()
    chosen_index_column = index_column

    if chosen_index_column and chosen_index_column != INDEX_SENTINEL:
        if chosen_index_column not in normalized.columns:
            raise ValueError(f"Column '{chosen_index_column}' is not present in the CSV.")
        timestamp_index = pd.to_datetime(normalized[chosen_index_column], errors="coerce")
        normalized = normalized.drop(columns=[chosen_index_column])
    else:
        if isinstance(normalized.index, pd.DatetimeIndex):
            timestamp_index = normalized.index
        else:
            parsed_index = pd.to_datetime(normalized.index, errors="coerce")
            valid_ratio = parsed_index.notna().mean()
            uses_non_numeric_index = not pd.api.types.is_numeric_dtype(normalized.index.dtype)
            if valid_ratio >= 0.8 and uses_non_numeric_index:
                timestamp_index = parsed_index
            else:
                inferred_column = infer_timestamp_column(normalized)
                if inferred_column is None:
                    raise ValueError(
                        "Could not infer a timestamp index. Provide a timestamp column in the sidebar."
                    )
                timestamp_index = pd.to_datetime(normalized[inferred_column], errors="coerce")
                normalized = normalized.drop(columns=[inferred_column])

    normalized.index = timestamp_index
    normalized = normalized[normalized.index.notna()]
    normalized = normalized.sort_index()

    if isinstance(normalized.index, pd.DatetimeIndex) and normalized.index.tz is not None:
        normalized.index = normalized.index.tz_convert(None)

    normalized.index.name = "timestamp"
    return normalized


def render_controls(frame: pd.DataFrame, default_index_column: str | None):
    """Render sidebar controls and return selected plotting options."""
    column_options = [INDEX_SENTINEL, *frame.columns.tolist()]
    default_option = INDEX_SENTINEL
    if default_index_column and default_index_column in frame.columns:
        default_option = default_index_column
    elif default_index_column and default_index_column == INDEX_SENTINEL:
        default_option = INDEX_SENTINEL
    elif not default_index_column:
        inferred = infer_timestamp_column(frame)
        if inferred in frame.columns:
            default_option = inferred

    default_index_position = column_options.index(default_option)
    index_column = st.sidebar.selectbox(
        "Timestamp column",
        options=column_options,
        index=default_index_position,
    )

    plot_frame = build_timestamp_indexed_frame(frame, index_column=index_column)
    numeric_columns = plot_frame.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_columns:
        raise ValueError("No numeric columns were found after setting the timestamp index.")

    default_columns = numeric_columns[:MAX_DEFAULT_COLUMNS]
    selected_columns = st.sidebar.multiselect(
        "Series to plot",
        options=numeric_columns,
        default=default_columns,
    )
    if not selected_columns:
        raise ValueError("Select at least one numeric series to plot.")

    min_date = plot_frame.index.min().date()
    max_date = plot_frame.index.max().date()
    selected_dates = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if not isinstance(selected_dates, tuple) or len(selected_dates) != 2:
        raise ValueError("Select a start and end date.")
    start_date, end_date = selected_dates
    if start_date > end_date:
        raise ValueError("Date range is invalid: start date is after end date.")

    resample_rule = st.sidebar.selectbox(
        "Resample frequency",
        options=["None", "D", "W", "M"],
        index=0,
    )
    aggregation = st.sidebar.selectbox(
        "Aggregation",
        options=["mean", "sum", "min", "max"],
        index=0,
    )
    rolling_window = st.sidebar.slider(
        "Rolling window (points)",
        min_value=1,
        max_value=60,
        value=1,
    )

    return {
        "plot_frame": plot_frame,
        "selected_columns": selected_columns,
        "start_date": start_date,
        "end_date": end_date,
        "resample_rule": resample_rule,
        "aggregation": aggregation,
        "rolling_window": rolling_window,
    }


def apply_plot_transformations(
    plot_frame: pd.DataFrame,
    columns: list[str],
    start_date: dt.date,
    end_date: dt.date,
    resample_rule: str,
    aggregation: str,
    rolling_window: int,
) -> pd.DataFrame:
    """Apply selected range and smoothing transformations before plotting."""
    start_timestamp = pd.Timestamp(start_date)
    end_timestamp = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    filtered = plot_frame.loc[start_timestamp:end_timestamp, columns]

    if filtered.empty:
        return filtered

    if resample_rule != "None":
        filtered = getattr(filtered.resample(resample_rule), aggregation)()

    if rolling_window > 1:
        filtered = filtered.rolling(window=rolling_window, min_periods=1).mean()

    return filtered.dropna(how="all")


def main() -> None:
    """Run the Streamlit interactive plotting application."""
    args = parse_app_args()
    st.set_page_config(page_title="Interactive Time-Series Plot", layout="wide")
    st.title("Interactive Time-Series Plot")
    st.sidebar.header("Controls")

    csv_value = str(args.csv) if args.csv else ""
    csv_path_text = st.sidebar.text_input("CSV path", value=csv_value)
    uploaded_file = st.sidebar.file_uploader("Or upload CSV", type=["csv"])

    if uploaded_file is not None:
        source_frame = pd.read_csv(uploaded_file)
        source_label = f"uploaded:{uploaded_file.name}"
    elif csv_path_text:
        csv_path = Path(csv_path_text).expanduser()
        if not csv_path.exists():
            st.error(f"CSV path not found: {csv_path}")
            st.stop()
        source_frame = load_csv_frame(csv_path)
        source_label = str(csv_path)
    else:
        st.info("Provide a CSV path or upload a CSV file to start plotting.")
        st.stop()

    st.caption(f"Source: {source_label}")

    try:
        options = render_controls(source_frame, default_index_column=args.index_column)
        transformed = apply_plot_transformations(
            plot_frame=options["plot_frame"],
            columns=options["selected_columns"],
            start_date=options["start_date"],
            end_date=options["end_date"],
            resample_rule=options["resample_rule"],
            aggregation=options["aggregation"],
            rolling_window=options["rolling_window"],
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    if transformed.empty:
        st.warning("No rows are available for the selected range and transformations.")
        st.stop()

    plot_data = transformed.reset_index()
    figure = px.line(
        plot_data,
        x="timestamp",
        y=options["selected_columns"],
        title="Time Series",
    )
    figure.update_layout(
        xaxis_title="Timestamp",
        yaxis_title="Value",
        legend_title_text="Series",
    )
    st.plotly_chart(figure, use_container_width=True)
    st.dataframe(plot_data.tail(200), use_container_width=True)


if __name__ == "__main__":
    main()
