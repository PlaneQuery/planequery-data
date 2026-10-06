from datetime import date
from data_engineering.adsb.read_adsb import ADSBSource
from flights.evaluation.evaluation import evaluate

LEFT_ADSB_SOURCE = ADSBSource.ADSBLOL
RIGHT_ADSB_SOURCE = ADSBSource.ADSBX

# ---- desired days
SFPDS_DAYS = [date(2026,2,1), date(2026,3,1)]
BTS_DAYS =  SFPDS_DAYS + [date(2025, 6,1)]
EUROCONTROL_DAYS = [date(2025,6,1), date(2025,7,1), date(2025,8,1)]

evaluate

if __name__ == "__main__":
    main()

