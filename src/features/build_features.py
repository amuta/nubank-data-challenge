import pandas as pd
import reverse_geocoder as rg


def get_geo_info(data, ignore_na=True):
    """
    latitude - list o latitude values
    longitude - list o longitude values
    data - pd.DataFrame object with 2 type float columns
    """
    assert isinstance(lat_lon, pd.DataFrame)
    
    data = data.copy()
    na_idx = data.isna().any(axis=1)

    
    search_list = list(map(tuple, lat_lon.values))
    coords_info = rg.search()


