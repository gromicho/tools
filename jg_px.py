def PlotlyToFigure( figure, file_name, engine='orca' ):
    figure.write_file(file_name,engine=engine)