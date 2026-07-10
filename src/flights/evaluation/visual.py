import csv
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from random import Random

import polars as pl
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots


LABEL_VALID = "valid"
LABEL_INVALID = "invalid"
LABEL_UNKNOWN = "unknown"
DEFAULT_LABELS_CSV = "flight_visual_labels.csv"


def _rows(df: pl.DataFrame, sort_col: str) -> list[dict]:
    if sort_col in df.columns:
        df = df.sort(sort_col)
    return df.rows(named=True)


def _filter_icao(df: pl.DataFrame, icao: str) -> pl.DataFrame:
    if "icao" not in df.columns:
        raise ValueError("Expected dataframe to have an 'icao' column")
    return df.filter(pl.col("icao") == icao)


def _unique_icaos(df: pl.DataFrame) -> list[str]:
    if "icao" not in df.columns:
        return []
    return (
        df.select("icao")
        .drop_nulls()
        .unique()
        .sort("icao")
        .get_column("icao")
        .to_list()
    )


def _read_label_rows(output_csv: Path) -> list[dict]:
    if not output_csv.exists():
        return []
    return list(csv.DictReader(output_csv.open(newline="")))


def _existing_labeled_icaos(output_csv: Path) -> set[str]:
    return {row["icao"] for row in _read_label_rows(output_csv) if row.get("icao")}


def _existing_labels(output_csv: Path) -> dict[str, str]:
    return {
        row["icao"]: row["label"]
        for row in _read_label_rows(output_csv)
        if row.get("icao") and row.get("label")
    }


def _write_label(
    output_csv: Path,
    *,
    icao: str,
    label: str,
    index: int,
    total: int,
    n_flights: int,
    n_adsb_messages: int,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "icao",
        "label",
        "labeled_at",
        "index",
        "total",
        "n_flights",
        "n_adsb_messages",
    ]
    rows = [row for row in _read_label_rows(output_csv) if row.get("icao") != icao]
    rows.append(
        {
            "icao": icao,
            "label": label,
            "labeled_at": datetime.now(UTC).isoformat(),
            "index": index,
            "total": total,
            "n_flights": n_flights,
            "n_adsb_messages": n_adsb_messages,
        }
    )
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _flight_label(idx: int, flight: dict) -> str:
    callsign = flight.get("callsign") or f"Flight {idx}"
    takeoff = flight.get("takeoff_airport_ident") or "?"
    landing = flight.get("landing_airport_ident") or "?"
    return f"F{idx}: {callsign} {takeoff}->{landing}"


def visualize_flights_adsb(
    df_flights: pl.DataFrame,
    df_adsb: pl.DataFrame,
    *,
    width: int | None = 1200,
) -> go.Figure:
    no_flight_label = "No flight"

    flights = _rows(df_flights, "takeoff_time")
    flight_windows = [
        (
            flight.get("takeoff_time"),
            flight.get("landing_time"),
            _flight_label(idx, flight),
        )
        for idx, flight in enumerate(flights, start=1)
    ]

    def message_label(message: dict) -> str:
        time = message.get("time")
        for start, end, label in flight_windows:
            if (
                time is not None
                and start is not None
                and end is not None
                and start <= time <= end
            ):
                return label
        return no_flight_label

    messages = _rows(df_adsb, "time")
    groups = OrderedDict((label, []) for _, _, label in flight_windows)
    groups[no_flight_label] = []
    for message in messages:
        groups[message_label(message)].append(message)

    fig = make_subplots(
        rows=2,
        cols=1,
        specs=[[{"type": "xy"}], [{"type": "map"}]],
        row_heights=[0.36, 0.64],
        vertical_spacing=0.07,
    )

    palette = qualitative.Plotly + qualitative.Dark24 + qualitative.Light24
    colors = {
        label: ("#808080" if label == no_flight_label else palette[idx % len(palette)])
        for idx, label in enumerate(groups)
    }

    for start, end, label in flight_windows:
        if start is not None and end is not None:
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor=colors[label],
                opacity=0.07,
                line_width=0,
                row=1,
                col=1,
            )

    for label, group_messages in groups.items():
        if not group_messages:
            continue

        color = colors[label]
        hover_text = [
            "<br>".join(
                [
                    label,
                    f"time: {message.get('time')}",
                    f"altitude: {message.get('baro_altitude_ft')} ft",
                    f"speed: {message.get('ground_speed_kt')} kt",
                    f"track: {message.get('track_deg')} deg",
                    f"lat/lon: {message.get('lat')}, {message.get('lon')}",
                ]
            )
            for message in group_messages
        ]

        fig.add_trace(
            go.Scatter(
                x=[message.get("time") for message in group_messages],
                y=[message.get("baro_altitude_ft") for message in group_messages],
                mode="markers" if label == no_flight_label else "lines+markers",
                name=label,
                legendgroup=label,
                marker={"color": color, "size": 5},
                line={"color": color, "width": 2},
                hovertext=hover_text,
                hovertemplate="%{hovertext}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        map_messages = [
            (message, hover)
            for message, hover in zip(group_messages, hover_text)
            if message.get("lat") is not None and message.get("lon") is not None
        ]
        if map_messages:
            fig.add_trace(
                go.Scattermap(
                    lat=[message.get("lat") for message, _ in map_messages],
                    lon=[message.get("lon") for message, _ in map_messages],
                    mode="markers",
                    name=label,
                    legendgroup=label,
                    showlegend=False,
                    marker={"color": color, "size": 6, "opacity": 0.78},
                    hovertext=[hover for _, hover in map_messages],
                    hovertemplate="%{hovertext}<extra></extra>",
                ),
                row=2,
                col=1,
            )

    map_points = [
        message
        for message in messages
        if message.get("lat") is not None and message.get("lon") is not None
    ]
    map_center = (
        {
            "lat": sum(message.get("lat") for message in map_points) / len(map_points),
            "lon": sum(message.get("lon") for message in map_points) / len(map_points),
        }
        if map_points
        else {"lat": 0, "lon": 0}
    )
    icao = (messages[0].get("icao") if messages else None) or (
        flights[0].get("icao") if flights else ""
    )

    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Barometric altitude (ft)", row=1, col=1)
    fig.update_layout(
        title=f"Flights and ADS-B messages for {icao}",
        height=900,
        width=width,
        autosize=width is None,
        hovermode="closest",
        legend={"groupclick": "togglegroup", "traceorder": "normal"},
        map={
            "style": "open-street-map",
            "center": map_center,
            "zoom": 5 if map_points else 1,
        },
        margin={"l": 50, "r": 25, "t": 60, "b": 35},
    )
    return fig


def visualize_flight(
    df_flights: pl.DataFrame,
    df_adsb: pl.DataFrame,
    icao: str | None = None,
    *,
    width: int | None = 1200,
    show: bool = True,
) -> go.Figure:
    if icao is not None:
        df_flights = _filter_icao(df_flights, icao)
        df_adsb = _filter_icao(df_adsb, icao)
    fig = visualize_flights_adsb(df_flights, df_adsb, width=width)
    if show:
        fig.show(config={"scrollZoom": True, "responsive": True})
    return fig


def _label_icaos(
    df_flights: pl.DataFrame,
    df_adsb: pl.DataFrame,
    *,
    n_icaos: int,
    icaos: list[str] | None,
    shuffle: bool,
    seed: int,
    output_csv: Path,
    skip_labeled: bool,
) -> list[str]:
    if icaos is None:
        icaos = _unique_icaos(df_flights)
        flight_icaos = set(icaos)
        adsb_icaos = [icao for icao in _unique_icaos(df_adsb) if icao not in flight_icaos]
        icaos = [*icaos, *adsb_icaos]
    else:
        icaos = list(icaos)

    if skip_labeled:
        labeled_icaos = _existing_labeled_icaos(output_csv)
        icaos = [icao for icao in icaos if icao not in labeled_icaos]

    if shuffle:
        Random(seed).shuffle(icaos)

    return icaos[:n_icaos]


def visualize_flights_df(
    df_flights: pl.DataFrame,
    df_adsb: pl.DataFrame,
    *,
    n_icaos: int = 100,
    output_csv: str | Path = DEFAULT_LABELS_CSV,
    icaos: list[str] | None = None,
    shuffle: bool = False,
    seed: int = 0,
    skip_labeled: bool = True,
    width: int | None = 1200,
    show: bool = True,
):
    from IPython.display import display
    import ipywidgets as widgets

    output_csv = Path(output_csv)
    label_icaos = _label_icaos(
        df_flights,
        df_adsb,
        n_icaos=n_icaos,
        icaos=icaos,
        shuffle=shuffle,
        seed=seed,
        output_csv=output_csv,
        skip_labeled=skip_labeled,
    )

    state = {"index": 0}
    labels = _existing_labels(output_csv)
    full_width = widgets.Layout(width="100%")
    status = widgets.HTML(layout=full_width)
    plot_output = widgets.Output(layout=full_width)
    button_layout = widgets.Layout(width="19%")
    previous_button = widgets.Button(description="Previous", layout=button_layout)
    next_button = widgets.Button(description="Next", layout=button_layout)
    valid_button = widgets.Button(
        description="Valid", button_style="success", layout=button_layout
    )
    invalid_button = widgets.Button(
        description="Invalid", button_style="danger", layout=button_layout
    )
    unknown_button = widgets.Button(
        description="Unknown", button_style="warning", layout=button_layout
    )
    label_buttons = [valid_button, invalid_button, unknown_button]
    nav_buttons = [previous_button, next_button]
    controls = widgets.HBox([*nav_buttons, *label_buttons], layout=full_width)
    ui = widgets.VBox([status, controls, plot_output], layout=full_width)

    def update_buttons() -> None:
        done = not label_icaos or state["index"] >= len(label_icaos)
        previous_button.disabled = done or state["index"] == 0
        next_button.disabled = done or state["index"] >= len(label_icaos) - 1
        for button in label_buttons:
            button.disabled = done

    def render_current(*, render_plot: bool = True) -> None:
        plot_output.clear_output(wait=True)
        if state["index"] >= len(label_icaos):
            status.value = f"<b>Done.</b> Wrote labels to <code>{output_csv}</code>."
            update_buttons()
            return

        icao = label_icaos[state["index"]]
        df_icao_flights = _filter_icao(df_flights, icao)
        df_icao_adsb = _filter_icao(df_adsb, icao)
        current_label = labels.get(icao, "unlabeled")
        status.value = (
            f"<b>{state['index'] + 1}/{len(label_icaos)}</b> "
            f"<code>{icao}</code> "
            f"label=<b>{current_label}</b> "
            f"flights={df_icao_flights.height:,} "
            f"adsb={df_icao_adsb.height:,} "
            f"output=<code>{output_csv}</code>"
        )
        update_buttons()
        if render_plot:
            with plot_output:
                visualize_flight(
                    df_icao_flights,
                    df_icao_adsb,
                    width=width,
                    show=True,
                )

    def record(label: str) -> None:
        if state["index"] >= len(label_icaos):
            return
        icao = label_icaos[state["index"]]
        df_icao_flights = _filter_icao(df_flights, icao)
        df_icao_adsb = _filter_icao(df_adsb, icao)
        labels[icao] = label
        _write_label(
            output_csv,
            icao=icao,
            label=label,
            index=state["index"] + 1,
            total=len(label_icaos),
            n_flights=df_icao_flights.height,
            n_adsb_messages=df_icao_adsb.height,
        )
        if state["index"] < len(label_icaos) - 1:
            state["index"] += 1
        render_current(render_plot=show)

    def previous_icao(_) -> None:
        if state["index"] > 0:
            state["index"] -= 1
            render_current(render_plot=show)

    def next_icao(_) -> None:
        if state["index"] < len(label_icaos) - 1:
            state["index"] += 1
            render_current(render_plot=show)

    previous_button.on_click(previous_icao)
    next_button.on_click(next_icao)
    valid_button.on_click(lambda _: record(LABEL_VALID))
    invalid_button.on_click(lambda _: record(LABEL_INVALID))
    unknown_button.on_click(lambda _: record(LABEL_UNKNOWN))
    render_current(render_plot=show)

    if show:
        display(ui)
        return None
    return ui
