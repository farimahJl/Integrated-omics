def fillna(df, value):
    """Fill NaN values with a single value and return the DataFrame
    
    Keyword arguments:
    df -- a DataFrame
    value -- the value to fill NaN with
    """
    data = df.copy()
    data.fillna(value)

    return data