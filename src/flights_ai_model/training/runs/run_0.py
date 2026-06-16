from datetime import date

from data_engineering.adsb.read_adsb import read_icaos_from_adsb
from data_engineering.bts.read_bts_ontime_to_flights import get_bts_flights_for_day
from data_engineering.flights.sfdps_to_flights import get_sfdps_flights_day
from evaluation.flights import evaluation
from flights.flights_match_adsb import get_matching_icaos_in_flights
from flights_ai_model.features import get_icaos_in_features, read_icaos_from_features
from flights_ai_model.inference import inference, run_inference
from flights_ai_model.train import train
from flights_ai_model.training.training_days import sequence_training_days
import polars as pl

from flights_ai_model.validation import flights_df_comparison
train_paths = sequence_training_days(bts_days=2, sfdps_days=2)
model_output_paths = train(train_paths, epochs=10)

# get the training features from a day. 
# Get all the infernerce features for a day although I could get just the training features subset since that's what I'm going to be checking anyways. 
for path in model_output_paths:
    val_date = date(2026,3,1)
    df_test = run_inference(val_date, path)
    sfdps_flights = get_sfdps_flights_day(val_date)
    bts_flights = get_bts_flights_for_day(val_date)
    icaos = list(set(sfdps_flights.get_column("icao")) | set(bts_flights.get_column("icao")))
    df_adsb = read_icaos_from_adsb(val_date, icaos)
    for df_gold in [sfdps_flights, bts_flights]:
        df_gold = get_matching_icaos_in_flights(df_gold, df_adsb)
        df_test = df_test.filter(pl.col("icao").is_in(df_gold.get_column("icao").implode()))
        flights_df_comparison(df_test, df_gold)

