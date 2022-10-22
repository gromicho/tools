import networkx as nx, numpy as np, matplotlib.pyplot as plt
from collections import defaultdict

import signatures as s

from pathlib import Path

_output_path = './work in progress/'

def SetOutputPath( path ):
    global _output_path
    _output_path = path
    if not _output_path.endswith('/'):
        _output_path += '/'
    Path(_output_path).mkdir(parents=True, exist_ok=True)

def PosFromXY( G ):
    x = nx.get_node_attributes(G,'x')
    y = nx.get_node_attributes(G,'y')
    return { n : (x[n],y[n]) for n in G.nodes() }

def FillGraphFromFrames( g, nodes, edges ):
    for node, row in nodes.iterrows():
        g.add_node(node,**row.to_dict())
    for edge, row in edges.iterrows():
        g.add_edge(*edge,**row.to_dict())   
    return g

def Nudge( pos, direction='', factor=1 ):
    pos = np.array(pos)
    zero = np.array((0,0))
    cardinal_directions = { 'E' : (1,0), 'S' : (0,-1), 'W' : (-1,0), 'N' : (0,1) }
    delta = zero
    for d in direction:
        delta = delta + np.array( cardinal_directions.get(d.upper(),zero) )    
    delta = factor * delta / np.linalg.norm(delta) if np.linalg.norm(delta) else zero
    return tuple(pos + delta)

def InitFigure( figsize, spines=False ):
    fig, ax = plt.subplots(figsize=figsize)
    for spine in ['top','right','bottom','left']:    
        ax.spines[spine].set_visible(spines)
    return fig, ax

def Draw( G, pos, functions, figsize=(8,6), file_name=None ):
    fig, ax = InitFigure(figsize=figsize)
    for (f,a) in functions:
        f( G, **( s.spec(f)[-1] | dict(pos=pos) | a | {'ax' : ax } ) )
    if file_name:
        fig.savefig( file_name+'.pdf', bbox_inches='tight', pad_inches=0 )
    plt.show()