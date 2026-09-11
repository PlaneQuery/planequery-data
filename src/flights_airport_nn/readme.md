# Archetecture
USE GRU. It worked in flight segementation and ChatGPT recommends it. 

# Features
1. Take groundtruth flights

# V0
For now Don't do any masking in ADS-B data simply see if I can make a neural network that predicts well.
Relative airport features probably very similar to LightGBM model. 

# V1

# V2 future
Pretraining?: Could use pretraining i.e let it learn what aircraft trajectorys look like before getting labelled data. 

# V0 details

Candidate-relative ADS-B features
        +
repeated candidate airport/runway features
        ↓
      GRU
        ↓
   Linear(128 → 1)
        ↓
      score+


For each (flight endpoint, candidate airport) pair, the input is a sequence:

$$ X \in \mathbb{R}^{T \times F} $$

where T is the number of ADS-B points and F is the number of features per point.

For a landing, use roughly the last 5–10 minutes of the segmented flight. For takeoff, use the first 5–10 minutes.

Per-point features I would start with

For every ADS-B point and candidate airport:
[
    # Position relative to candidate airport
    east_km,
    north_km,
    distance_to_airport_km,

    # Altitude
    altitude_above_airport_ft,

    # Aircraft motion
    ground_speed_kts,
    vertical_rate_fpm,

    # Direction relative to airport
    sin_track_minus_bearing,
    cos_track_minus_bearing,

    # Motion toward/away from airport
    radial_velocity_kts,

    # Timing / coverage
    delta_time_seconds,
    time_from_endpoint_seconds,

    # ADS-B state
    on_ground,
]

[
    east_km,
    north_km,
    distance_km,
    altitude_above_airport_ft,
    ground_speed,
    vertical_rate,
    sin_heading_error,
    cos_heading_error,
    radial_velocity,
    delta_t,
    time_from_endpoint,
    on_ground,

    # repeated for every point
    airport_elevation_ft,
    longest_runway_ft,
    airport_type,
]

# 0.00
Maybe just ? 
[
    east_km,
    north_km,
    distance_to_airport_km,
    delta_time_seconds,
]