import networkx as nx
import pandas as pd
import numpy as np

from pathlib import Path

_output_path = './work in progress/'

def SetOutputPath( path ):
    global _output_path
    _output_path = path
    if not _output_path.endswith('/'):
        _output_path += '/'
    Path(_output_path).mkdir(parents=True, exist_ok=True)

def MakeGraph( nodes, edges, type_length=int ):
    g = nx.DiGraph()

    use_colors = len( nodes.legend.dropna() ) == 0

    for node,row in nodes.iterrows():
        color = 'black' if pd.isna( row.color ) else row.color
        assert( node == row.name )
        if pd.isna( row.legend ):
            label = node if pd.isna( row.label ) else row.name+':'+str(int(row.label))
        else:
            label = row.legend
        if not use_colors:
            color = 'black'
        aux = row.to_dict()
        aux.pop('color')
        aux.pop('label')
        aux.pop('x')
        aux.pop('y')
        g.add_node(node, pos='{},{}!'.format(row.x,row.y), label=label, color=color, **aux)

    for edge,row in edges.iterrows():
        aux = row.to_dict()
        aux.pop('color')
        aux.pop('length')
        color = 'black' if pd.isna( row.color ) else row.color
        if not use_colors:
            color = 'black'
        g.add_edge(*edge, length=row.length, label=str(type_length(row.length)), color=color, **aux)    

    return g

def DrawIntoFile( g, filename ):
    ag = nx.nx_agraph.to_agraph( g )
    ag.layout()
    ag.draw(_output_path+filename)
    
def DrawAndShow( nodes, edges, filename='test', type_length=int ):
    filename += '.svg'
    DrawIntoFile( MakeGraph( nodes, edges, type_length ), filename )
    from IPython.display import SVG, display
    display( SVG(filename) )
    
def DrawIntoPDF( nodes, edges, filename='test', type_length=int ):
    filename += '.pdf'
    DrawIntoFile( MakeGraph( nodes, edges, type_length ), filename )
    
def PoorManDijkstra( nodes, edges, start, terminus, show, filename ):
    nodes['label'] = nodes['color'] = nodes['legend'] = np.NaN
    edges['color'] = np.NaN
    nodes.loc[start,'label'] = 0
    iteration = 0 
    while pd.isna(nodes.loc[terminus,'color']):
        current = nodes[nodes.color.isna()].label.idxmin()
        nodes.loc[current,'color'] = 'red'
        if current != terminus:
            for to in edges.loc[current].index:
                edges.loc[(current,to),'color'] = 'red'
                if pd.isna(nodes.loc[to,'label']):
                    nodes.loc[to,'label'] = nodes.loc[current,'label']+edges.loc[(current,to),'length']
                else:
                    nodes.loc[to,'label'] = min(nodes.loc[to,'label'],nodes.loc[current,'label']+edges.loc[(current,to),'length'])
        show( nodes, edges, filename+f'{iteration:03}' )
        edges.color = np.NaN
        iteration += 1
    path = [terminus]
    while edges.loc[start].color.dropna().empty:
        for a,b in edges.loc[pd.IndexSlice[:, path[0]],:].iterrows():
            if nodes.loc[a[0]].label + b.length == nodes.loc[path[0]].label:
                path = [a[0]] + path
        edges.loc[(path[0],path[1]),'color'] = 'green'
        show( nodes, edges, filename+f'{iteration:03}' )
        iteration += 1
    return path

def PoorManDijkstraReuse( nodes, edges, start, terminus, show, filename, ndecimals=0, type_length=int ):
    description = []
    nodes['label']  = nodes['color'] = np.NaN
    nodes['legend'] = nodes.index.astype(str) + ': -| '
    edges['color'] = np.NaN
    nodes.loc[start,'label'] = 0
    iteration = 0 
    start_label = nodes.loc[start,'label']
    nodes.loc[start,'legend'] = nodes.loc[start,'legend'] + f'{start_label:.{ndecimals}f}' + '('+str(start)+')'
    description.append( 'The start node {} receives the temporary label 0.'.format(start) )
    while pd.isna(nodes.loc[terminus,'color']):
        current = nodes[nodes.color.isna()].label.idxmin()
        description.append( f'Lowest temporary label at iteration {iteration} is {nodes[nodes.color.isna()].label.min():.{ndecimals}f} at node {current}, permanent.' )
        nodes.loc[current,'color'] = 'red'
        nodes.loc[current,'legend'] = nodes.loc[current,'legend'].replace('-|',f'{iteration}|')
        if current != terminus:
            description.append( 'We scan the successors {} of node {}.'.format(list(edges.loc[current].index),current))
            for to in edges.loc[current].index:
                current_label = nodes.loc[current,'label']
                length_to = edges.loc[(current,to),'length']
                description.append( f' - Node {to} can be reached from {current} in {current_label+length_to:.{ndecimals}f} = {current_label:.{ndecimals}f} + {length_to:.{ndecimals}f}.' )
                updated = False
                edges.loc[(current,to),'color'] = 'red'
                if pd.isna(nodes.loc[to,'label']):
                    nodes.loc[to,'label'] = nodes.loc[current,'label']+edges.loc[(current,to),'length']
                    updated = True
                else:
                    min_label = min(nodes.loc[to,'label'],nodes.loc[current,'label']+edges.loc[(current,to),'length'])
                    updated = min_label < nodes.loc[to,'label']
                    nodes.loc[to,'label'] = min_label
                if updated:
                    to_label = nodes.loc[to,'label']
                    nodes.loc[to,'legend'] += ' ' + f'{to_label:.{ndecimals}f}' + '('+str(current)+')'
                    description.append( f' - node {to} receives temporary label {to_label:.{ndecimals}f} = {current_label:.{ndecimals}f} + {length_to} from {current}' )
        else:
            description.append( 'Since {} is the terminus node, we stop.'.format(current) )
            description.append( 'We create the path, from the end.')
            description.append( 'We add node by node that led to the label of the node added.')

        edges.color = np.NaN
        iteration += 1
    path = [terminus]
    while edges.loc[start].color.dropna().empty:
        for a,b in edges.loc[pd.IndexSlice[:, path[0]],:].iterrows():
            if nodes.loc[a[0]].label + b.length == nodes.loc[path[0]].label:
                path = [a[0]] + path
        edges.loc[(path[0],path[1]),'color'] = 'green'
        iteration += 1
    show( nodes, edges, filename, type_length=type_length )
    description.append( 'The path is therefore {}'.format(path))
    return path, description

def DijkstraOnTable( nodes, edges, start, terminus ):
    result = pd.DataFrame( index=range(len(nodes.index)+1), columns=['node'] + list(nodes.index) )
    to_scan = set(nodes.index)
    iteration = 0
    result.loc[iteration,start]  = 0
    while terminus in to_scan:
        current = result.loc[iteration][to_scan].fillna(np.inf).idxmin()
        to_scan.remove( current )
        result.loc[iteration+1] = result.loc[iteration]
        iteration += 1
        result.loc[iteration,'node'] = current
        if to_scan:
            aux = result.loc[iteration].fillna(np.inf)
            for to in set(edges.loc[current].index).intersection(to_scan):
                result.loc[iteration,to] = min( result.loc[iteration,current] + edges.loc[(current,to),'length'], aux[to] )

    path = [terminus]
    while path[0] != start:
        for a,b in edges.loc[pd.IndexSlice[:, path[0]],:].iterrows():
            if result.loc[iteration,a[0]] + b.length == result.loc[iteration,path[0]]:
                path = [a[0]] + path
    
    return result, path