import pandas as pd 

def ReadMatrix( file_name ):
    aux = pd.read_csv( f'{file_name}.csv', index_col=0 )
    aux.columns = pd.to_numeric( aux.columns )
    return aux 

def make_iterable( thing ):
    try:
        iter(thing)
    except TypeError:
        return ( thing, )
    else:
        return thing