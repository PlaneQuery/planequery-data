"timestamp": pl.Datetime("us"),                
"flightStatus": pl.String,
"takeoff_time": pl.Datetime("us"),
"takeoff_airport_icao": pl.String,
"landing_time": pl.Datetime("us"),
"landing_airport_icao": pl.String,
"estimated_landing_time": pl.Datetime("us"),
"estimated_takeoff_time": pl.Datetime("us"),

# I need to correspond the gufis sequences to the icao. Maybe having multipel gufi is a feature? "gufi" will be a feature.