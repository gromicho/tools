def ExtractInstance( data, shifts, addresses, D, T, nof_decimals=2 ):
    import numpy as np, jg
    these = data.SHIFT.isin( jg.make_iterable(shifts) )
    aux = data[ these ][['LATITUDE','LONGITUDE']].dropna().drop_duplicates().reset_index(drop=True)
    aux['idx'] = [ addresses.loc[ (lat,lon) ].idx for (lat,lon) in aux.values ]
    orders = data[these & ((data.ACTION =='pickup') | (data.ACTION =='deliver')) ].fillna(0).copy().reset_index(drop=True)
    orders['ORDER'] = orders.ORDER.astype(int)
    orders['LM'] = np.round( np.diff([0]+orders.LOAD_LM.to_list()).tolist(), nof_decimals )
    orders['KG'] = np.round( np.diff([0]+orders.LOAD_KG.to_list()).tolist(), nof_decimals )
    orders['idx'] = [ addresses.loc[ (lat,lon) ].idx for (lat,lon) in orders[['LATITUDE','LONGITUDE']].values ]
    return orders[['ACTION','ORDER','LATITUDE','LONGITUDE','idx','EARLIEST_START_TIME','LATEST_START_TIME','LM','KG']], aux, D.loc[aux.idx,aux.idx], T.loc[aux.idx,aux.idx]