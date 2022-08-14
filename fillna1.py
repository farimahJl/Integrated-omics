#!/usr/bin/env python

"""This module contains functions to fill missing values. It requires a DataFrame as input.

Example usage:
fillna:
	df = fillna(df_nan, value)

"""

__author__ = "Maryam Jalali"
__license__ = "MIT"

def fillna(df, value):
    """Fill NaN values with a single value and return the DataFrame

    Keyword arguments:
    df -- a DataFrame
    value -- the value to fill NaN with
    """
    data = df.copy()
    data.fillna(value)

    return data
