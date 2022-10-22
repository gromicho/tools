import pandas as pd
import inspect

def spec( f ):
    args  = inspect.getfullargspec(f)
    kargs = dict(list(zip(args.args[::-1],args.defaults[::-1]))[::-1])
    return args.args[:len(args.args)-len(args.defaults)], kargs
    
def DisplayNamedArguments( functions ):
    stringify_values = lambda d : { k : f"'{v}'" if type(v) == str else f'{v}' for k,v in d.items() }
    return pd.DataFrame.from_dict( { m.__name__ : stringify_values(spec( m )[-1]) for m in functions }, orient='columns' ).fillna('').sort_index()

def DisplayPositionalArguments( functions ):
    return pd.DataFrame.from_dict( { m.__name__ : spec( m )[0] for m in functions }, orient='index' ).fillna('').sort_index().T