import argparse
from datetime import datetime, timedelta, date
from pathlib import Path
import os
import tempfile
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import json
from flights_ai_model.features import get_icaos_in_features, read_icaos_from_features
from flights_ai_model.model import FEATURE_COLS, FlightEventBiGRU, MAX_SEQ_LEN, WINDOW_STRIDE, get_device
from data_engineering.flights.flight_type import FLIGHT_POLARS_SCHEMA
from airports.airportLookup import AirportLookup
airport_lookup = AirportLookup()
OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"])
MODELS_DIR = OUTPUT_DIR / "data" / "models"


def get_latest_model_path() -> Path:
    """Return the path to the most recently saved model under MODELS_DIR.

    Directory layout:
        MODELS_DIR/
            2026-03-25/
                2026-03-25_14-30.pth
                2026-03-25_09-00.pth
            2026-03-24/
                ...
    """
    pth_files = sorted(MODELS_DIR.glob("*/*/*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No .pth model files found in {MODELS_DIR}")
    print(f"Found {len(pth_files)} model files. Latest: {pth_files[-1]}")
    return pth_files[-1]



LABEL_NAMES = {0: "OUTSIDE_FLIGHT", 1: "FIRST", 2: "MIDDLE", 3: "LAST"}
_LABEL_NAMES_ARRAY = np.array(["OUTSIDE_FLIGHT", "FIRST", "MIDDLE", "LAST"])
INFERENCE_BATCH_SIZE = 64
# Upper bound on total padded tokens (batch_size * max_seq_len_in_batch) per
# forward pass. Keeps memory bounded when some ICAOs have very long sequences.
INFERENCE_MAX_PADDED_TOKENS = 64 * 2048
# Match the training window size for very long ICAO timelines. cuDNN GRU can
# reject extremely long packed sequences (for example ~94k ADS-B rows for one
# ICAO on g5 instances), so long timelines are evaluated as overlapping windows.
INFERENCE_WINDOW_MAX_SEQ_LEN = MAX_SEQ_LEN
INFERENCE_WINDOW_STRIDE = WINDOW_STRIDE
MIN_FLIGHT_DURATION = timedelta(minutes=5)

# ── Viterbi decoding with transition constraints ──────────────────────────
# States: 0=OUTSIDE, 1=FIRST, 2=MIDDLE, 3=LAST
# Allowed transitions encode valid flight structure:
#   OUTSIDE → OUTSIDE | FIRST
#   FIRST   → MIDDLE  | LAST
#   MIDDLE  → MIDDLE  | LAST
#   LAST    → OUTSIDE | FIRST
_NEG_INF = -1e9
TRANSITION_LOG = np.full((4, 4), _NEG_INF, dtype=np.float64)
TRANSITION_LOG[0, 0] = 0.0   # OUTSIDE → OUTSIDE
TRANSITION_LOG[0, 1] = 0.0   # OUTSIDE → FIRST
TRANSITION_LOG[1, 2] = 0.0   # FIRST   → MIDDLE
TRANSITION_LOG[2, 2] = 0.0   # MIDDLE  → MIDDLE
TRANSITION_LOG[2, 3] = -1.0  # MIDDLE  → LAST  (penalty discourages very short flights)
TRANSITION_LOG[3, 0] = 0.0   # LAST    → OUTSIDE
TRANSITION_LOG[3, 1] = 0.0   # LAST    → FIRST

# Initial state: must start in OUTSIDE or FIRST
INITIAL_LOG = np.array([0.0, 0.0, _NEG_INF, _NEG_INF], dtype=np.float64)


def viterbi_decode(log_probs: np.ndarray) -> np.ndarray:
    """Run Viterbi on (T, 4) log-probability matrix. Returns (T,) int labels."""
    T, C = log_probs.shape
    V = np.full((T, C), _NEG_INF, dtype=np.float64)
    backptr = np.zeros((T, C), dtype=np.int32)

    V[0] = INITIAL_LOG + log_probs[0]

    # Vectorize inner state loop: scores[prev, s] = V[t-1, prev] + TRANSITION_LOG[prev, s]
    for t in range(1, T):
        scores = V[t - 1, :, None] + TRANSITION_LOG  # (C, C)
        best_prev = np.argmax(scores, axis=0)        # (C,)
        V[t] = scores[best_prev, np.arange(C)] + log_probs[t]
        backptr[t] = best_prev

    # Backtrack — force ending in OUTSIDE or LAST for clean termination
    terminal_mask = np.array([0.0, _NEG_INF, _NEG_INF, 0.0], dtype=np.float64)
    final_scores = V[T - 1] + terminal_mask
    if np.max(final_scores) == _NEG_INF:
        # Fallback: allow any ending state
        final_scores = V[T - 1]
    path = np.zeros(T, dtype=np.int32)
    path[T - 1] = int(np.argmax(final_scores))
    for t in range(T - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]
    return path


def load_model(model_path: Path, device: torch.device) -> FlightEventBiGRU:
    state_dict = torch.load(model_path, map_location=device)
    input_dim = state_dict["gru.weight_ih_l0"].shape[1]
    if input_dim != len(FEATURE_COLS):
        print(f"Warning: checkpoint input_dim={input_dim} differs from current FEATURE_COLS ({len(FEATURE_COLS)}). Using checkpoint dim.")
    model = FlightEventBiGRU(input_dim=input_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def _features_to_tensor(df: pl.DataFrame) -> torch.Tensor:
    return torch.from_numpy(
        np.ascontiguousarray(
            df.select(FEATURE_COLS).to_numpy().astype(np.float32, copy=False)
        )
    )

def _get_group(groups: list[pl.DataFrame | None], index: int) -> pl.DataFrame:
    df = groups[index]
    if df is None:
        raise RuntimeError(f"ICAO group at index {index} was already released")
    return df

def _is_cudnn_not_supported_error(exc: RuntimeError) -> bool:
    return "CUDNN_STATUS_NOT_SUPPORTED" in str(exc)

def _forward_batched_log_probs(
    model: FlightEventBiGRU,
    x_padded: torch.Tensor,
    lengths: list[int],
) -> np.ndarray:
    x_padded = x_padded.contiguous()
    try:
        logits = model.forward_batched(x_padded, lengths)
    except RuntimeError as exc:
        if x_padded.device.type == "cuda" and _is_cudnn_not_supported_error(exc):
            print("cuDNN GRU failed for a short batch; retrying with cuDNN disabled.")
            with torch.backends.cudnn.flags(enabled=False):
                logits = model.forward_batched(x_padded, lengths)
        else:
            raise
    log_probs = F.log_softmax(logits, dim=-1).cpu().numpy()
    del logits
    return log_probs

def _window_starts(seq_len: int) -> list[int]:
    if seq_len <= INFERENCE_WINDOW_MAX_SEQ_LEN:
        return [0]

    last_start = seq_len - INFERENCE_WINDOW_MAX_SEQ_LEN
    starts = list(range(0, last_start + 1, INFERENCE_WINDOW_STRIDE))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts

def _forward_long_sequence_log_probs(
    model: FlightEventBiGRU,
    x: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Infer a single long ICAO sequence as overlapping training-sized windows."""
    seq_len = x.shape[0]
    starts = _window_starts(seq_len)
    log_prob_sums: np.ndarray | None = None
    counts = np.zeros(seq_len, dtype=np.float32)

    for window_batch_start in range(0, len(starts), INFERENCE_BATCH_SIZE):
        batch_starts = starts[window_batch_start:window_batch_start + INFERENCE_BATCH_SIZE]
        window_tensors = [
            x[start:start + INFERENCE_WINDOW_MAX_SEQ_LEN]
            for start in batch_starts
        ]
        lengths = [t.shape[0] for t in window_tensors]
        x_windows = pad_sequence(window_tensors, batch_first=True).to(device).contiguous()
        batch_log_probs = _forward_batched_log_probs(model, x_windows, lengths)
        del x_windows, window_tensors

        if log_prob_sums is None:
            log_prob_sums = np.zeros((seq_len, batch_log_probs.shape[-1]), dtype=np.float32)
        for i, start in enumerate(batch_starts):
            end = start + lengths[i]
            log_prob_sums[start:end] += batch_log_probs[i, :lengths[i]]
            counts[start:end] += 1
        del batch_log_probs

    if log_prob_sums is None or np.any(counts == 0):
        raise RuntimeError(f"Windowed inference failed to cover sequence of length {seq_len}")
    return log_prob_sums / counts[:, None]

def _append_event_predictions(
    event_dfs: list[pl.DataFrame],
    df: pl.DataFrame,
    viterbi_ids: np.ndarray,
) -> None:
    # Keep all in-flight rows (FIRST/MIDDLE/LAST) so we can compute per-flight
    # aggregates (e.g. callsign mode) in extract_flights.
    keep_mask = viterbi_ids != 0
    if keep_mask.any():
        event_df = (
            df
            .select(["icao", "time", "lat", "lon", "callsign", "registration", "aircraft_type", "aircraft_description", "owner", "category", "pia", "ladd", "military", "interesting"])
            .filter(pl.Series(keep_mask))
            .with_columns(
                pl.Series(
                    "pred_label",
                    _LABEL_NAMES_ARRAY[viterbi_ids[keep_mask]],
                    dtype=pl.String,
                )
            )
        )
        event_dfs.append(event_df)

def compress_aircraft_information(df: pl.DataFrame):
    compressed_dict = {}
    columns = ["callsign", "registration", "aircraft_type", "aircraft_description", "owner", "category", "pia", "ladd", "military", "interesting"]
    for col in columns:
        values = df.get_column(col).drop_nulls().mode()
        compressed_dict[col] = values[0] if len(values) > 0 else None
    return compressed_dict
def extract_flights(df_preds: pl.DataFrame) -> pl.DataFrame:
    """Post-process raw predictions and pair TAKEOFF/LANDING per ICAO."""
    # Only FIRST/LAST rows are needed to delimit flights; MIDDLE rows are used
    # for per-flight aggregates like callsign mode.
    boundary_events = (
        df_preds
        .filter(pl.col("pred_label").is_in(["FIRST", "LAST"]))
        .sort(["icao", "time"])
    )

    flights = []
    for icao_key, group in boundary_events.group_by("icao", maintain_order=True):
        icao = icao_key[0] if isinstance(icao_key, (list, tuple)) else icao_key # TODO: reduce this.
        takeoff_time = None
        for row in group.iter_rows(named=True):
            if row["pred_label"] == "FIRST":
                first_lat = row["lat"] # SEE IF this works. 
                first_lon = row["lon"]
                takeoff_airport_ident = airport_lookup.getClosestAirport(first_lat, first_lon).ident
                takeoff_time = row["time"]
            elif row["pred_label"] == "LAST" and takeoff_time is not None:
                last_lat = row["lat"]
                last_lon = row["lon"]
                landing_airport_ident = airport_lookup.getClosestAirport(last_lat, last_lon).ident
                if (row["time"] - takeoff_time) >= MIN_FLIGHT_DURATION:
                    flight_rows = df_preds.filter(
                        (pl.col("icao") == icao)
                        & (pl.col("time") >= takeoff_time)
                        & (pl.col("time") <= row["time"])
                    )
                    compressed_info = compress_aircraft_information(flight_rows)
                    flights.append({
                        "icao":                   icao,
                        "callsign":               compressed_info["callsign"],
                        "registration":           compressed_info["registration"],
                        "takeoff_time":           takeoff_time,
                        "takeoff_airport_ident":  takeoff_airport_ident,
                        "landing_time":           row["time"],
                        "landing_airport_ident":  landing_airport_ident,
                        "pia":                 compressed_info["pia"],
                        "ladd":                compressed_info["ladd"],
                        "military":            compressed_info["military"],
                        "interesting":         compressed_info["interesting"],
                        "aircraft_type":      compressed_info["aircraft_type"],
                        "owner":              compressed_info["owner"],
                        "aircraft_description": compressed_info["aircraft_description"],
                        "category":           compressed_info["category"]
                    })
                takeoff_time = None
    df = pl.DataFrame(flights, schema=FLIGHT_POLARS_SCHEMA)
    return df

def inference(df_features: pl.DataFrame, model: Path | FlightEventBiGRU): # TODO: I may just want to load from adsb directly to not have extra training stuff.
    device = get_device()
    if isinstance(model, Path):
        model = load_model(model, device)

    # Partition into per-ICAO groups. We defer tensor materialization until
    # each group is pulled into its batch, so we never hold all tensors at once.
    _groups = df_features.sort(["icao", "time"]).partition_by("icao", maintain_order=True)
    del df_features

    # Sort by descending length (required for pack_padded_sequence), then allow
    # slots to be cleared to None as we consume them to release memory.
    _groups.sort(key=lambda g: g.height, reverse=True) # TODO: I should probably write this 
    groups: list[pl.DataFrame | None] = list(_groups)
    del _groups

    event_dfs: list[pl.DataFrame] = []

    with torch.no_grad():
        batch_start = 0
        n = len(groups)
        while batch_start < n:
            # Dynamic batch size: take up to INFERENCE_BATCH_SIZE items, but cap
            # total padded tokens (since all are padded to the first/longest seq).
            first_group = _get_group(groups, batch_start)
            max_len_in_batch = first_group.height

            if max_len_in_batch > INFERENCE_WINDOW_MAX_SEQ_LEN:
                df = first_group
                icao = df.get_column("icao")[0]
                x = _features_to_tensor(df)
                try:
                    log_probs = _forward_long_sequence_log_probs(model, x, device)
                except RuntimeError:
                    print("GRU inference failed for long ICAO sequence:")
                    print(f"  icao={icao} seq_len={x.shape[0]}")
                    raise
                viterbi_ids = viterbi_decode(log_probs)
                _append_event_predictions(event_dfs, df, viterbi_ids)
                groups[batch_start] = None
                del x, log_probs
                batch_start += 1

                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            max_items_by_tokens = max(1, INFERENCE_MAX_PADDED_TOKENS // max_len_in_batch)
            batch_size = min(INFERENCE_BATCH_SIZE, max_items_by_tokens, n - batch_start)
            batch_end = batch_start + batch_size
            batch_groups = [_get_group(groups, j) for j in range(batch_start, batch_end)]

            batch_tensors = [
                _features_to_tensor(group)
                for group in batch_groups
            ]
            batch_icaos = [group.get_column("icao")[0] for group in batch_groups]
            batch_lengths = [t.shape[0] for t in batch_tensors]

            x_padded = pad_sequence(batch_tensors, batch_first=True).to(device).contiguous()
            del batch_tensors
            try:
                log_probs = _forward_batched_log_probs(model, x_padded, batch_lengths)
            except RuntimeError:
                print("GRU inference failed for ICAO batch:")
                for icao, seq_len in zip(batch_icaos, batch_lengths):
                    print(f"  icao={icao} seq_len={seq_len}")
                raise
            del x_padded

            for i in range(batch_size):
                j = batch_start + i
                df = batch_groups[i]
                seq_len = batch_lengths[i]
                viterbi_ids = viterbi_decode(log_probs[i, :seq_len])
                _append_event_predictions(event_dfs, df, viterbi_ids)
                # Release group memory as soon as it's been used.
                groups[j] = None

            del log_probs, batch_groups
            batch_start = batch_end

            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not event_dfs:
        return pl.DataFrame(schema=FLIGHT_POLARS_SCHEMA)

    df_preds = pl.concat(event_dfs).sort(["icao", "time"])
    flights_df = extract_flights(df_preds)
    return flights_df

def save_inference_results(model_path: Path, target_date: date, df_flights: pl.DataFrame, df_stats: pl.DataFrame | None = None) -> None:
    base_dir = model_path.parent / f"{target_date:%Y%m%d}"
    run = 0
    while True:
        output_dir = base_dir / f"{run:02d}"
        if not output_dir.exists():
            break
        run += 1
    output_dir.mkdir(parents=True, exist_ok=True)
    if "match_status" in df_flights.columns:
        df_flights.write_parquet(output_dir / "flights_with_validation.parquet")
        print(f"Flights with validation comparison saved to {output_dir / 'flights_with_validation.parquet'}")
    else:
        df_flights.write_parquet(output_dir / "flights.parquet")
        print(f"Flights saved to {output_dir / 'flights.parquet'}")
    if df_stats is not None:
        df_stats.write_csv(output_dir / "validation_stats.csv")
        print(f"Validation stats saved to {output_dir / 'validation_stats.csv'}")
# Hard code model path for now
BATCH_SIZE = 1000
from data_engineering.utils import MODEL_PATH
from data_engineering.flights.flight_type import flights_parquet_path
def run_inference(target_date: date, model_path: Path = MODEL_PATH, icaos: list[str] | None = None, tmpdir = tempfile.TemporaryDirectory(), icao_start = 0, existing_flights_df: pl.DataFrame | None = None) -> pl.DataFrame:
    start_time = datetime.now()
    if icaos is None:
        icaos = get_icaos_in_features(target_date)
    print(f"Running inference for {len(icaos)} ICAOs on {target_date} using model {model_path}")
    print(f"using {get_device()} for inference")
    flights_output_path = flights_parquet_path(target_date)
    os.makedirs(flights_output_path.parent, exist_ok=True)
    tmpdir_path = Path(tmpdir.name)
    model = load_model(model_path, get_device())
    for batch_num, i in enumerate(range(icao_start, len(icaos), BATCH_SIZE)):
        icao_subset = icaos[i:i+BATCH_SIZE]
        df_features = read_icaos_from_features(target_date, icao_subset)
        df_flights = inference(df_features, model)
        df_flights = df_flights.filter(pl.col("takeoff_time").dt.date() == target_date)
        data = {
            "icaos_stop": min(i+BATCH_SIZE, len(icaos))
        }
        if existing_flights_df is not None and batch_num == 0:
            df_flights = pl.concat([existing_flights_df, df_flights])
        df_flights.write_parquet(tmpdir_path / f"flights_batch_{batch_num:05d}.parquet")
        with open(flights_output_path.parent / "metadata.json", "w") as f:
            json.dump(data, f)
        print(f"Processed {min(i+BATCH_SIZE, len(icaos))}/{len(icaos)} ICAOs")
    pl.scan_parquet(tmpdir_path / "flights_batch_*.parquet").sink_parquet(flights_output_path)
    
    print(f"Inference flights saved to {flights_output_path}")
    print(f"Inference completed in {_format_time(datetime.now() - start_time)} (hh:mm:ss)")
    df_flights = pl.read_parquet(flights_output_path)
    return df_flights
def _format_time(td: timedelta) -> str:
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours}:{minutes:02d}:{seconds:02d}"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with FlightEventBiGRU")
    parser.add_argument("--date",       default="2025-08-01", help="Date YYYY-MM-DD")
    parser.add_argument("--model",      type=Path, default=None, help="Path to model .pth (default: latest)")
    parser.add_argument("--all-adsb", action="store_true", help="Run inference on all ADS-B data for the day, not just ICAOs in validation set")
    args = parser.parse_args()
    target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    model_path = Path(args.model) if args.model else MODEL_PATH
    df_flights = run_inference(target_date, model_path)
    print(df_flights)
