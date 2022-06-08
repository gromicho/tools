def ExtractInstance( data, shifts, addresses, D, T, nof_decimals=2 ):
    import numpy as np, jg
    these = data.SHIFT.isin( jg.make_iterable(shifts) )
    aux = data[ these ][['FINISH_LATITUDE','FINISH_LONGITUDE']].dropna().drop_duplicates().reset_index(drop=True)
    aux['idx'] = [ addresses.loc[ (lat,lon) ].idx for (lat,lon) in aux.values ]
    orders = data[these & ((data.ACTION =='pickup') | (data.ACTION =='deliver')) ][['ACTION','ORDER','ORDER_TYPE','SET_TEMPERATURE','FINISH_LATITUDE','FINISH_LONGITUDE','EARLIEST_START_TIME','LATEST_START_TIME','LOAD_LM','LOAD_KG']].copy().reset_index(drop=True)
    orders['LOAD_LM'] = orders.LOAD_LM.fillna(0)
    orders['LOAD_KG'] = orders.LOAD_LM.fillna(0)
    orders['ORDER'] = orders.ORDER.astype(int)
    orders['LM'] = np.round( np.diff([0]+orders.LOAD_LM.to_list()).tolist(), nof_decimals )
    orders['KG'] = np.round( np.diff([0]+orders.LOAD_KG.to_list()).tolist(), nof_decimals )
    orders['idx'] = [ addresses.loc[ (lat,lon) ].idx for (lat,lon) in orders[['FINISH_LATITUDE','FINISH_LONGITUDE']].values ]
    return orders[['ACTION','ORDER','ORDER_TYPE','SET_TEMPERATURE','FINISH_LATITUDE','FINISH_LONGITUDE','idx','EARLIEST_START_TIME','LATEST_START_TIME','LM','KG']], aux, D.loc[aux.idx,aux.idx], T.loc[aux.idx,aux.idx]