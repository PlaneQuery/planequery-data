def visualize_flights_df(df_flights, df_adsb):
    pass
def visualize_flight():
    pass

from collections import OrderedDict

import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots


def visualize_flights_adsb(df_flights, df_adsb):
    no_flight_label = "No flight"

    def rows(df, sort_col):
        if sort_col in df.columns:
            df = df.sort(sort_col)
        return df.rows(named=True)

    def flight_label(idx, flight):
        callsign = flight.get("callsign") or f"Flight {idx}"
        takeoff = flight.get("takeoff_airport_ident") or "?"
        landing = flight.get("landing_airport_ident") or "?"
        return f"F{idx}: {callsign} {takeoff}->{landing}"

    def message_label(message):
        time = message.get("time")
        for start, end, label in flight_windows:
            if time is not None and start is not None and end is not None and start <= time <= end:
                return label
        return no_flight_label

    flights = rows(df_flights, "takeoff_time")
    flight_windows = [
        (
            flight.get("takeoff_time"),
            flight.get("landing_time"),
            flight_label(idx, flight),
        )
        for idx, flight in enumerate(flights, start=1)
    ]

    messages = rows(df_adsb, "time")
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
                    f"{label}",
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
    icao = (messages[0].get("icao") if messages else None) or (flights[0].get("icao") if flights else "")

    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Barometric altitude (ft)", row=1, col=1)
    fig.update_layout(
        title=f"Flights and ADS-B messages for {icao}",
        height=900,
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
