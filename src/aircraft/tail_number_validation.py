import polars as pl
def tail_number_validation(df: pl.DataFrame) -> pl.DataFrame: #TODO: This should really take a series to make it more general.
    N_NUMBER_REGEX = r"^N([1-9]\d{0,4}|[1-9]\d{0,3}[A-HJ-NP-Z]|[1-9]\d{0,3}[A-HJ-NP-Z]{2})$"

    df = df.filter(
        pl.col("tail_num").str.to_uppercase().str.contains(N_NUMBER_REGEX)
    )
    return df

"246NV"
# read some sfpds and see percetnage filtered. 
