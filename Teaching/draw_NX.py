import networkx as nx, numpy as np, matplotlib.pyplot as plt
from collections import defaultdict

import signatures as s

from pathlib import Path

_output_path = './work in progress/'

def SetOutputPath( path ):
    """sets the module level _output_path variable and ensures that the corresponding path exists

    Args:
        path (str): intended path to use
    """    
    global _output_path
    _output_path = path
    if not _output_path.endswith('/'):
        _output_path += '/'
    Path(_output_path).mkdir(parents=True, exist_ok=True)

def PosFromXY( G, h='x',v='y' ):
    """create the pos dictionary from node attributes

    Args:
        G (graph): a given graph, directed or not
        h (str, default 'x'): the name of the node attribute for the horizontal axis
        h (str, default 'y'): the name of the node attribute for the vertical axis
    Returns:
        dict: node positions conform networkx 
    """    
    x = nx.get_node_attributes(G,h)
    y = nx.get_node_attributes(G,v)
    return { n : (x[n],y[n]) for n in G.nodes() }

def FillGraphFromFrames( g, nodes, edges ):
    """ Fills a graph with nodes and edges and the corresponding attributes from data frames. 
    The nodes frame is expected to be indexed on the node names and the edges frame on the tuples of nodes defining  the edges. 
    Args:
        g (graph): either a nx.Graph or a nx.DiGraph object
        nodes (dataframe): one node per row, the columns define attribute values 
        edges (dataframe): one edge per row, the columns define attribute values 

    Returns:
        graph: the graph g taken as input extended with the nodes and edges from the dataframes
    """    
    for node, row in nodes.iterrows():
        g.add_node(node,**row.to_dict())
    for edge, row in edges.iterrows():
        g.add_edge(*edge,**row.to_dict())   
    return g

def Nudge( pos, direction='', factor=1. ):
    """nudge the given coordinates a step of length factor in the cardinal direction given

    Args:
        pos (tuple like): the pair of (x,y) coordinates
        direction (str, optional): the cardinal directions, e.g. 'NE' for North-East. Defaults to '' for not nudging.
        factor (float, optional): The length of the step to take along the direction. Defaults to 1.

    Returns:
        tuple: the nudged pos
    """    
    pos = np.array(pos)
    zero = np.array((0,0))
    cardinal_directions = { 'E' : (1,0), 'S' : (0,-1), 'W' : (-1,0), 'N' : (0,1) }
    delta = zero
    for d in direction:
        delta = delta + np.array( cardinal_directions.get(d.upper(),zero) )    
    delta = factor * delta / np.linalg.norm(delta) if np.linalg.norm(delta) else zero
    return tuple(pos + delta)

def InitFigure( figsize, spines=False ):
    """ Creates figure and corresponding axis. 

    Args:
        figsize (tuple): the intended figure size
        spines (bool, optional): whether to have framing spines. Defaults to False.

    Returns:
        tuple: fig and ax
    """    
    fig, ax = plt.subplots(figsize=figsize)
    for spine in ['top','right','bottom','left']:    
        ax.spines[spine].set_visible(spines)
    return fig, ax

def Draw( G, pos, steps, figsize=(8,6), spines=False, file_name=None ):
    """Template method to draw a graph using a sequence of steps.

    Args:
        G (graph): nx.Graph or nx.DiGraph
        pos (dict): the node positions
        steps (list of tuples): list of tuples (function,dict) where function is a networkx drawing method and dict the arguments deviating from default
        figsize (tuple, optional): the intended figure size. Defaults to (8,6).
        spines (bool, optional): whether to have a frame of spines. Defaults to False.
        file_name (string, optional): file to save the figure into. Defaults to None.
    """    
    fig, ax = InitFigure(figsize=figsize,spines=spines)
    for (f,a) in steps:
        f( G, **( s.spec(f)[-1] | dict(pos=pos) | a | dict( ax=ax ) ) )
    if file_name:
        fig.savefig( file_name, bbox_inches='tight', pad_inches=0 )
    plt.show()