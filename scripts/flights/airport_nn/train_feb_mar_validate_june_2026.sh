python -m flights.airport_nn.data \
--dates 2026-02-{01..28} 2026-03-{01..31} 2026-06-{01..05} \
--output data/features/flights_airport_nn/feb-mar-june-2026
--workers 4
# will use 4 workers on my mac, 3 on ubunutu machine by default.

python -m flights.airport_nn.train \
--cache data/features/flights_airport_nn/feb-mar-june-2026 \
--output models/flights_airport_nn/feb-mar-june-2026 \
--train-dates 2026-02-{01..28} 2026-03-{01..31} \
--validation-dates 2026-06-{01..05} \
--epochs 10