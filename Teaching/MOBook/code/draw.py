import sys
import matplotlib.pyplot as plt
import numpy as np
import itertools
from collections import namedtuple
import pandas as pd
import pyomo.environ as pyo
from pyomo.repn import generate_standard_repn
from anytree import Node
from anytree.exporter import DotExporter
import dot2tex
from pathlib import Path

_output_path = 'd:/joaquimg/work/MO book/results/'

def SetOutputPath( path ):
    global _output_path
    _output_path = path
    if not _output_path.endswith('/'):
        _output_path += '/'
    Path(_output_path).mkdir(parents=True, exist_ok=True)

def Interpret( model ):
    collect = lambda what : list( model.component_objects(what, active=True) )
    variables   = collect(pyo.Var)
    constraints = collect(pyo.Constraint)
    objectives  = collect(pyo.Objective)

    names       = [f.name for f in constraints + objectives]
    expressions = [f.body for f in constraints] + [f.expr for f in objectives]

    repn = [ generate_standard_repn(e) for e in expressions ]
    coefficients = { name : { v.name : c for v,c in zip(repn.linear_vars,repn.linear_coefs) } for name,repn in zip(names,repn) }

    Result = namedtuple(  'Canonical'
                        , [ 'lower_variable'
                           , 'upper_variable'
                           , 'lower_constraint'
                           , 'upper_constraint'
                           , 'formulas'
                           , 'coefficients'
                           , 'objective'
                          ]
                       )

    return Result(  { v.name : v.bounds[0] for v in variables }
                  , { v.name : v.bounds[1] for v in variables }
                  , { c.name : c.lb for c in constraints }
                  , { c.name : c.ub for c in constraints }
                  , { n : str(f) for n,f in zip(names,expressions) }
                  , coefficients
                  , { o.name : 'min' if o.is_minimizing() else 'max' for o in objectives } 
                 )
    
def GetCanonicalMatrices( rep ):
    variables   = list(rep.lower_variable.keys())
    constraints = list(rep.lower_constraint.keys())
    objective   = list(rep.objective.keys())[0]
    n = len(variables)
    A = []
    b = []
    expressions = []
    for j,v in enumerate(variables):
        if rep.lower_variable[v] is not None:
            line = np.zeros(n)
            line[j] = -1
            A.append(line)
            b.append(-rep.lower_variable[v])
            expressions.append(v+' \geq '+str(rep.lower_variable[v]) )
        if rep.upper_variable[v] is not None:
            line = np.zeros(n)
            line[j] = 1
            A.append(line)
            b.append(rep.upper_variable[v])
            expressions.append(v+' \leq '+str(rep.upper_variable[v]) )
    for c in constraints:
        line = np.array([rep.coefficients[c].get(v,0) for v in variables])
        if rep.lower_constraint[c] is not None:
            A.append(-1*line)
            b.append(-rep.lower_constraint[c])
            expressions.append(rep.formulas[c]+' \geq '+str(rep.lower_constraint[c]) )
        if rep.upper_constraint[c] is not None:
            A.append(line)
            b.append(rep.upper_constraint[c])
            expressions.append(rep.formulas[c]+' \leq '+str(rep.upper_constraint[c]) )
    factor = 1 if rep.objective[objective] == 'max' else -1
    return np.vstack(A),np.vstack(b),factor*np.array([rep.coefficients[objective].get(v,0) for v in variables]),expressions
 
def GetBasicFeasibleSolutions( A, b ):
    m,n   = A.shape
    basis = []
    for i,j in itertools.combinations(range(len(b)),n):
        try:
            basis.append( np.linalg.solve(A[[i,j],:],b[[i,j]]).T )
        except:
            continue   
    import sys
    basis = np.vstack(basis).T
    return sys.float_info.epsilon*10 + basis[:,np.all( np.dot(A, basis) <= b+sys.float_info.epsilon*1e3, axis = 0 )]
 
# https://stackoverflow.com/questions/35394564/is-there-a-context-manager-for-temporarily-changing-matplotlib-settings
def Draw( model, file_name=None, 
         trajectories=dict(), isolines=True, integer=False, 
         xlim=None, ylim=None, title=None, xticksteps=None, **kwargs
         ):
    rep       = Interpret(model)
    variables = list(rep.lower_variable.keys())
    n         = len(variables)
    assert(n==2)

    plt.grid()
    plt.xlabel(r'$'+variables[0].replace('x','x_')+'$')
    plt.ylabel(r'$'+variables[1].replace('x','x_')+'$')

    A,b,c,expressions = GetCanonicalMatrices( rep )
    basis = GetBasicFeasibleSolutions( A, b )

    x = n*[[]]
    if ( xlim is None ) or ( ylim is None ):
        for i in range(n):
            min_i = min(0,min(basis[i]))
            max_i = max(0,max(basis[i]))
            delta = max_i - min_i
            x[i]  = np.linspace( min_i-delta/8, max_i+delta/8, 1000 )
    else:
        x[0] = np.linspace( xlim[0], xlim[1], 1000 )
        x[1] = np.linspace( ylim[0], ylim[1], 1000 )

    m,_ = A.shape
    cmap2 = plt.get_cmap('tab20')
    colors = [ cmap2(i) for i in [2,4,6,10,12,16,18] ] + ['black']*m

    for j, color in zip(range(m),colors):
        label = expressions[j].replace('*','').replace('x','x_')
        label = r'$'+label+'$'
        row = A[j,:]
        if np.count_nonzero(row) == n:
            X = (b[j]-row[0]*x[0])/row[1]
            plt.plot(x[0], X, label = label, color=color, zorder=3, alpha=1)
        else:
            assert( np.count_nonzero(row) == 1 )
            if row[0] == 0:
                plt.plot(x[0], b[j]/row[1]*np.ones_like(x[1]), label = label, color=color, zorder=2, alpha=1)
            else:
                assert( row[1] == 0 )
                plt.plot(b[j]/row[0]*np.ones_like(x[0]), x[1], label = label, color=color, zorder=2, alpha=1)

    if basis.size > 0:
        opt = basis[:,np.argmax(np.dot(c,basis))]
        plt.scatter( opt[0], opt[1], label = r'$'+str(tuple(opt.round(1)))+'$', **( dict( marker='o', s=49, facecolors='cyan', edgecolors='black', linewidth=.8 ) | kwargs.get('optsol_style', dict()) ), zorder=13 )

    obj = c if list(rep.objective.values())[0] == 'max' else -1*c
    
    if isolines:
        basic = sorted(np.dot(obj,basis))
        cmap = plt.get_cmap('YlGnBu') #YlGnBu YlOrRd
        for index, value in enumerate(basic):
            color = cmap(0.3 + 0.7*(index+1)/len(basic))
            label = rep.formulas[list(rep.objective.keys())[0]].replace('x','x_').replace('*','')
            label = r'$'+label+' = '+str(round(value,1))+'$'
            if obj[0] == 0 and obj[1] != 0:
                plt.plot(x[0], value/obj[1]*np.ones_like(x[1]), '--', label = label, zorder=5, alpha=1, color=color)
            elif obj[0] != 0 and obj[1] == 0:
                plt.plot(value/obj[0]*np.ones_like(x[0]), x[1], '--', label = label, zorder=5, alpha=1, color=color)
            elif obj[0] == 0 and obj[1] == 0:
                assert( opt == 0 )
                plt.plot(np.zeros_like(x[0]), x[1], '--', label = label, zorder=5, alpha=1, color=color)
            else:
                assert( all( c != 0 ) )
                plt.plot(x[0], (value-obj[0]*x[0])/obj[1], '--', label = label, zorder=5, alpha=1, color=color)

    plt.scatter( basis[0], basis[1], **( dict( marker='o', s=49, facecolors='white', edgecolors='black', linewidth=.8 ) | kwargs.get('basicsol_style', dict()) ), zorder=11 )

    x[0],x[1] = np.meshgrid(x[0],x[1])
    borders = [ (A[j,0]*x[0]+A[j,1]*x[1] <= b[j]).astype(int) for j in range(m) ]
    image = borders[0]
    for i in range(1,len(borders)):
        image *= borders[i]
        
    plt.imshow( image
               , extent=(x[0].min(),x[0].max(),x[1].min(),x[1].max())
               , origin="lower"
               , cmap="Greys"
               , alpha = 0.2
               , zorder = 0)
    
    for (label,points),color in zip(trajectories.items(),['red','black','blue','green']+['gray']*len(trajectories)):
        plt.scatter(points[0][0],points[0][1], marker='o',zorder=9,color=color)
        line, = plt.plot(*zip(*points),'-',label=label,linewidth=2.5,zorder=9,color=color)
        for start, terminus in itertools.pairwise(points):
            plt.annotate("",
            xy=terminus,
            xytext=start,
            arrowprops=dict(arrowstyle="->", lw=1.5, color=line.get_color()),
            color='black')

    if xlim is None:
        plt.xlim( x[0].min(),x[0].max() )
    else:
        plt.xlim( xlim )

    if ylim is None:
        plt.ylim( x[1].min(),x[1].max() )
    else:
        plt.ylim( ylim )

    if xticksteps:        
        from matplotlib.ticker import MultipleLocator
        plt.gca().xaxis.set_major_locator(MultipleLocator(2))
    
    if integer:
        points = list(itertools.product( range(int(x[0].min()),int(x[0].max())+1), range(int(x[1].min()),int(x[1].max())+1) ) )
        feasible   = [ p for p in points if ( np.dot(A,p) <= b.T + sys.float_info.epsilon*10 ).all() ]
        infeasible = [ p for p in points if ( np.dot(A,p) > b.T + sys.float_info.epsilon*10 ).any() ]
        if infeasible:
            plt.plot( *zip(*infeasible), 'ro', zorder=8)
        if feasible:
            plt.plot( *zip(*feasible), 'bo', zorder=8)
            
    plt.legend( **( dict(loc='center left', bbox_to_anchor=(1, 0.5)) | kwargs.get('legend_arguments',dict()) ) )

    plt.gca().set_aspect('equal', adjustable='box')
    
    if title:
        plt.title( title )
    
    plt.draw()
    plt.tight_layout()

    if file_name is not None:
        plt.savefig( _output_path+file_name, bbox_inches='tight', pad_inches=0 )

    plt.show()      

    return pd.DataFrame( np.vstack( [ basis, np.dot(obj,basis) ] ).T.round(1), columns=variables + ['value'] ).sort_values( by=variables )

def BB( m, solver='cbc', draw_integer=True, xlim=(-.5,5.5), ylim=(-.5,7.5), graphics_format='pdf' ):
    x      = [ m.x1, m.x2 ]
    xi     = [ x._name.replace('x','x_') for x in x ]
    solver = pyo.SolverFactory(solver)

    lb  = -np.inf
    sol = None

    idx = 0

    root = None

    def _BB( m, parent, innequality ):
        nonlocal x, xi, solver, lb, sol, idx, root, xlim, ylim
        idx = idx+1

        node = Node( 'Node {}'.format(idx), parent ) 
        node.idx         = idx
        node.innequality = innequality
        node.x           = None
        node.lb          = None
        node.ub          = None
        node.termination = None
        
        if parent is None:
            root = node

        result = solver.solve(m)
        if (result.solver.status == pyo.SolverStatus.ok) and \
           (result.solver.termination_condition == pyo.TerminationCondition.optimal):
            node.x = (pyo.value(x[0]),pyo.value(x[1]))
            ub = pyo.value( m.obj )
            if ub > lb:
                s = np.array( [ pyo.value(x) for x in x ] )
                print(s)
                fs = np.floor(s)
                cs = np.ceil(s)
                j = np.argmax( np.absolute( s - ( fs + (cs-fs)/2 ) ) )
                if fs[j] == cs[j]:
                    if ub > lb:
                        lb = ub
                        sol = s
                    node.termination = 'feasible'
                    node.lb          = lb
                    node.ub          = ub
                else:
                    node.lb          = lb
                    node.ub          = ub
                    xlb,xub = x[j].bounds
                    x[j].setub( fs[j] )
                    _BB(m, node, (xi[j],r'\leq',fs[j]) )
                    x[j].setub( xub )
                    x[j].setlb( cs[j] )
                    _BB(m, node, (xi[j],r'\geq',cs[j]) )
                    x[j].setlb( xlb )
            else:
                node.termination = 'fathomed'
                node.lb          = lb
                node.ub          = ub
        else:
            node.termination = 'infeasible'
            node.lb          = lb
            node.ub          = -np.inf
            
        
        v = lambda x : str(round(x,1)) if x is not None and not np.isinf(x) else '?'
        legend_title = f'LB=${v(node.lb)}$ UB=${v(node.ub)}$'
        Draw(m,integer=draw_integer,isolines=False, xlim=xlim, ylim=ylim, file_name=f'{m.name}_{node.idx}.{graphics_format}', title=f'Node ${node.idx}$', legend_arguments=dict(title=legend_title, handlelength=0.5, handletextpad=0.5, borderaxespad=0.5), xticksteps=2 )


    _BB( m, root, None )
    return sol, root

nodecolors = { 'infeasible' : 'red', 'feasible' : 'green', 'fathomed' : 'magenta' }

def TeXnumber( x ):
    if np.isinf(x):
        return '-'
    return '{:.1f}'.format(x)

node_labels = { 
#    'infeasible' : lambda node : 'label="{{ {} | {{ {} | {} }} | {} }}" shape=record color={}'.format(node.name,TeXnumber(node.lb),TeXnumber(node.ub),node.termination,nodecolors.get(node.termination,'blue')),
    'infeasible' : lambda node : 'label="{{ {} | ({}, {}) | {{ {}|{} }} | {} }}" shape=record color={}'.format(node.name,'-','-',TeXnumber(node.lb),TeXnumber(node.ub),node.termination,nodecolors.get(node.termination,'blue')),
    'all'        : lambda node : 'label="{{ {} | ({:.1f}, {:.1f}) | {{ {}|{} }} | {} }}" shape=record color={}'.format(node.name,*node.x,TeXnumber(node.lb),TeXnumber(node.ub),node.termination,nodecolors.get(node.termination,'blue')),
    'default'    : lambda node : 'label="{{ {} | ({:.1f}, {:.1f}) | {{ {}|{} }} }}" shape=record color={}'.format(node.name,*node.x,TeXnumber(node.lb),TeXnumber(node.ub),nodecolors.get(node.termination,'blue'))
    }

def nodeattrfunc(node):
    if node.termination:
        if node.x:
            return node_labels['all'](node)        
        else:
            return node_labels['infeasible'](node)
    return node_labels['default'](node)

def edgeattrfunc(node, child):
    label = '{} {} {:.0f}'.format(*child.innequality)
    return 'label="dummy" texlbl="${}$"'.format(label)

def edgetypefunc(node, child):
    return '--'
 
def Dotter( root ):
    return DotExporter(root,nodeattrfunc=nodeattrfunc,edgeattrfunc=edgeattrfunc)

def DrawBB( root, file_name ):
    Dotter(root).to_picture(_output_path+file_name)
    
def ToTikz( root, tex_file_name, dot_file_name=None, fig_only=True ):
    
    replace = { '\n-#0000' : 'black'
            ,   '-#0000' : 'black'
            , '{0.0,0.0,1.0}' : '{0.129, 0.588, 0.953}'
            , '{1.0,0.0,1.0}' : '{1.000, 0.647, 0}'
            , '{0.0,1.0,0.0}' : '{0.298, 0.686, 0.314}'
            , '{1.0,0.0,0.0}' : '{0.957, 0.263, 0.212}'
            , r'\pgfsetlinewidth{1bp}' : r'\pgfsetlinewidth{2bp}'
    }
        
    dot = '\n'.join(list(Dotter(root)))
    if dot_file_name:
        with open(_output_path+dot_file_name,'w') as f:
            f.write(dot)
    tex = dot2tex.dot2tex(dot,crop=True,figonly=fig_only,tikzedgelabels=False)
#    tex = '\n'.join( [line for line in tex.splitlines() if not '-#' in line] )

    for k,v in replace.items():
        tex = tex.replace(k,v)
    
    with open(_output_path+tex_file_name,'w') as f:
        f.write(tex)
        
def v(x):
    return x.replace('*','').replace('x','x_').replace('[',r'{').replace(']',r'}')
#    return x.replace('*','').replace('x','x_').replace('[',r'\\text{').replace(']',r'}')

def number2str(x):
    if type(x) is str:
        return x
    try:
        return str(int(x)) if int(x) == x else str(x)
    except:
        return ''

def sign(x):
    if x:
        return '-' if float(x)<0 else '+'
    return x

def describe_pair(c,x):
    x = v(x)
    try:
        if c == 1:
            return x
        elif c == -1:
            return '-'+x
        elif c == 0:
            return '' 
        return number2str(c)+x
    except:
        return ''

def justify(terms):
    width = dict(zip(terms.columns,map(lambda x: max( len(y) for y in x ),terms.T.values)))
    return terms.apply(lambda x : [ y.rjust(width[x.name]) for y in x] )

def remove_leading_plus( list_of_signs ):
    first = np.where( list_of_signs != ' ' )[0][0]
    if list_of_signs[first] == '+':
        list_of_signs[first] = ' '
    return list_of_signs

def Summary(interpreted):
    A = pd.DataFrame.from_dict( interpreted.coefficients, orient='index' )
    U = pd.DataFrame.from_dict( interpreted.upper_constraint | interpreted.upper_variable, orient='index', columns=['U'] )
    L = pd.DataFrame.from_dict( interpreted.lower_constraint | interpreted.lower_variable, orient='index', columns=['L'] )
    return pd.concat([L,A,U],axis=1).replace([None], np.nan),L,A,U
            
def to_latex(model,astype=int):
    interpreted = Interpret(model)
   
    data,L,A,U = Summary(interpreted)
    
    vars = data.index[data.index.isin(data.columns)].values
    for v in vars:
        data.at[v,v] = 1

    A = data[vars]
    
    signs = pd.DataFrame( index=A.index, columns=A.columns, data=' ' )
    signs[ A>0 ] = '+'
    signs[ A<0 ] = '-'
    signs.columns = [ f's_{x}' for x in signs.columns]
    signs = signs.apply( lambda x : remove_leading_plus( x.values ), axis=1, result_type='expand' )
    
    Leq = pd.DataFrame( index=L.index, columns=L.columns )
    Ueq = pd.DataFrame( index=U.index, columns=U.columns )
    Leq[ (L.L<U.U) | U.U.isna() & ~L.L.isna() ] = r'\le'
    Leq[ L.L==U.U ] = r'='
    Ueq[ (L.L<U.U) | L.L.isna() & ~U.U.isna() ] = r'\le'
    Ueq[ L.L==U.U ] = r'='
    
    idx = U.U.isna() & ~L.L.isna()
    
    rr = Ueq
    rr.columns = ['rhs_rel']
    rr[ idx ] = r'\ge'
    
    rhs = U.copy()
    rhs.columns = ['rhs']
    rhs[ idx ] = L[idx]
    
    idx = ~U.U.isna() & ~L.L.isna() & (L.L < U.U )
    
    if any( idx ):
        lr = pd.DataFrame( index=L.index, columns=L.columns )
        lr.columns = ['lhs_rel']
        lr[ idx ] = Leq[idx]
        
        lhs = pd.DataFrame( index=L.index, columns=L.columns )
        lhs.columns = ['lhs']
        lhs[ idx ] = L[idx]
    else:
        lr  = pd.DataFrame( index=L.index )
        lhs = pd.DataFrame( index=L.index )
    
    terms = abs(A).fillna(0).astype(astype).apply(lambda x: [describe_pair(v,k) for k,v in x.items()], axis=1, result_type='expand')
    terms.columns = A.columns
    
    from toolz import interleave
    
    result = pd.concat( [lhs, lr, pd.concat([signs,terms], axis=1)[list(interleave([signs,terms]))], rr, rhs], axis=1 ).replace([None], np.nan).applymap(number2str).fillna('')
    
    obj = list(interpreted.objective.keys())[0]
    idx = np.where( result.index == obj )[0][0]
    idx = [ idx ] + [ i for i in range(len(result.index)) if i != idx ]
    result = result.iloc[idx]
    
    leftmost = pd.DataFrame( index=result.index, columns=['left'], data='' )
    leftmost.iloc[0,0] = rf'\{interpreted.objective[obj]}'
    leftmost.iloc[1,0] = r'\text{s.t.:}'
    result = justify( pd.concat( [leftmost, result], axis=1 ) )
    
    body = '\n'.join( [ ' & '.join( line ) + r' \\' for line in result.values ] )
    
    latex = '\n'.join( [ rf'\begin{{array}}{{{"r"*result.shape[1]}}}', body, r'\end{array}' ] )
    return latex