def ExtractInstance( data, shifts, addresses, D, T, nof_decimals=2 ):
    """ Extract instance from the given shift(s)
    
    NOTE: 
        resources are not returned, as it is up to you where you want to plan this!

    Args:
        data (data frame): planning or subset
        shifts (int or enumerable): the shift(s) of interest
        addresses (data frame): the reverse geocoding information 
        D (data frame): the distance matrix
        T (data frame): the travel time matrix
        nof_decimals (int, optional): nof decimals to round order sizes to. Defaults to 2.

    Returns:
        O (data frame): the order information
        A (data frame): the address information
        d (data frame): the submatrix of D
        t (data frame): the submatrix of T
    """
    
    import numpy as np, jg
    these = data.SHIFT.isin( jg.make_iterable(shifts) )
    aux = data[ these ][['FINISH_LATITUDE','FINISH_LONGITUDE']].dropna().drop_duplicates().copy().reset_index(drop=True)
    aux['idx'] = [ addresses.loc[ (lat,lon) ].idx for (lat,lon) in aux.values ]
    orders = data[these & ((data.ACTION =='pickup') | (data.ACTION =='deliver')) ]\
        [['ACTION','ORDER','ORDER_TYPE','SET_TEMPERATURE','FINISH_LATITUDE','FINISH_LONGITUDE','EARLIEST_START_TIME','LATEST_START_TIME','LOAD_LM','LOAD_KG']].copy().reset_index(drop=True)
    orders['LOAD_LM'] = orders.LOAD_LM.fillna(0)
    orders['LOAD_KG'] = orders.LOAD_KG.fillna(0)
    orders['ORDER'] = orders.ORDER.astype(int)
    orders['LM'] = np.round( np.diff([0]+orders.LOAD_LM.to_list()).tolist(), nof_decimals )
    orders['KG'] = np.round( np.diff([0]+orders.LOAD_KG.to_list()).tolist(), nof_decimals )
    orders['idx'] = [ addresses.loc[ (lat,lon) ].idx for (lat,lon) in orders[['FINISH_LATITUDE','FINISH_LONGITUDE']].values ]
    return orders[['ACTION','ORDER','ORDER_TYPE','SET_TEMPERATURE','FINISH_LATITUDE','FINISH_LONGITUDE','idx','EARLIEST_START_TIME','LATEST_START_TIME','LM','KG']], \
        aux, D.loc[aux.idx,aux.idx], T.loc[aux.idx,aux.idx]