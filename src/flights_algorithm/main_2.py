# 1. Get average alt_baro, 
import polars as pl

def main_2(df_adsb: pl.DataFrame):
    # 1. Calculate percentiles for df_adsb between breaks in ADS-B coverage. Use maxes from the data science.
    # I could simply look at every ADS-B gap and see what the speed would have had to have been. 
    # Edge Cases
    # The ocean
    # Edge case of plane that goes back to the same airport. 
    
    # What could mess this up? 
    
    pass