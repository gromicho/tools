def PlotlyToFigure( figure, file_name, engine='orca' ):
    figure.write_file(file_name,engine=engine)
    
import plotly.express as px 
def ShowResourceGantt( shift, data, resources, colors=px.colors.qualitative.Light24 ):
    import plotly.figure_factory
    import pandas as pd
    from itertools import chain
    import jg

    shift_data = data[ data.SHIFT == shift ][['RESOURCES','ACTION','START_TIME','FINISH_TIME']].copy().rename(columns = { 'START_TIME' : 'Start', 'FINISH_TIME' : 'Finish' }  )
    per_resource = []
    for r in set( chain.from_iterable( [ jg.make_iterable(eval(r)) for r in shift_data.RESOURCES if type(r) is str ] ) ):
        aux = shift_data[ shift_data.RESOURCES.str.contains(str(r)) ]
        aux[ 'Task' ] = resources[ resources.id_resource == str(r) ].resourceName.values[0]
        per_resource.append( aux )
    aux = pd.concat( per_resource )
    n = aux.ACTION.nunique()
    return plotly.figure_factory.create_gantt( aux, colors=colors[:n], group_tasks=True, index_col='ACTION', show_colorbar=True, title=shift )

def ShowShiftGantt( data, colors=px.colors.qualitative.Light24 ):
    import plotly.figure_factory
    aux = data.copy().rename(columns = { 'SHIFT': 'Task', 'START_TIME' : 'Start', 'FINISH_TIME' : 'Finish' } )
    n = aux.ACTION.nunique()
    return plotly.figure_factory.create_gantt( aux, colors=px.colors.qualitative.Light24[:n], group_tasks = True, index_col = 'ACTION', show_colorbar = True )