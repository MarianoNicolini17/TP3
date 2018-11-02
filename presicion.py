import networkx as nx
import igraph
from collections import Counter
from itertools import combinations
from itertools import product


def clusterize(nx_Graph, method="infomap"):
    """
    Calcula el agrupamiento en comunidades de un grafo.
    
    In:
        nx_Graph: grafo de networkx
        method: metodo de clustering, puede ser: "infomap", "fastgreedy", "eigenvector", "louvain", "edge_betweenness","label_prop", "walktrap", ""
        
    Out:
        labels_dict: diccionario de nodo : a label al cluster al que pertenece.
    """
    if method == "edge_betweenness":
        nx_Graph = max(nx.connected_component_subgraphs(nx_Graph), key=len)#se queda con la componente más grande.
        #print("AVISO: restringiendo a la componente connexa más grade. De otro modo falla el algoritmo de detección de comunidades edge_betweenness.")
    
    isdirected = nx.is_directed(nx_Graph)
    np_adj_list = nx.to_numpy_matrix(nx_Graph)
    g = igraph.Graph.Weighted_Adjacency(np_adj_list.tolist(),mode=igraph.ADJ_UPPER)
   
    if method=="infomap":
        labels = g.community_infomap(edge_weights="weight").membership
    if method=="label_prop":
        labels = g.community_label_propagation(weights="weight").membership
    if method=="fastgreedy":
        labels = g.community_fastgreedy(weights="weight").as_clustering().membership
    if method=="eigenvector":
        labels = g.community_leading_eigenvector(weights="weight").membership
    if method=="louvain":
        labels = g.community_multilevel(weights="weight").membership
    if method=="edge_betweenness":
        labels = g.community_edge_betweenness(weights="weight", directed=isdirected).as_clustering().membership
    if method=="walktrap":
        labels = g.community_walktrap(weights="weight").as_clustering().membership
    
    label_dict = {node: label for node,label in zip(nx_Graph.nodes(), labels)}
    return label_dict

#------------------------------------------------------------------------------

def poblacionComus(particion):
# IN: Una partición en forma de diccionario.
# OUT: Un diccionario donde las keys son las comunidades y los values son el
# número de miembros de cada comunidad.
    values = particion.values()
    return Counter(values)


# VER BIEN COMO FUNCIONA ESTA FUNCIÓN:
def pares(listas):
    pares_set = set()
    for t in combinations(listas, 2):
        for par in product(*t):
            pares_set.add(frozenset(par))
    return pares_set


def presicion(p1,p2): 
    comus1 = poblacionComus(p1)
    comus2 = poblacionComus(p2)
    comus1_list = []
    comus2_list = []
    n = len(p1)
    
    for comu in sorted(comus1.keys()):
        miembros_comu = []
        for nodo in p1.keys():
            if p1[nodo] == comu:
                miembros_comu.append(nodo)
        comus1_list.append(miembros_comu)
        
    for comu in sorted(comus2.keys()):
        miembros_comu = []
        for nodo in p2.keys():
            if p2[nodo] == comu:
                miembros_comu.append(nodo)
        comus2_list.append(miembros_comu)
        
        
    pares_comus1 = []    
    for comu in comus1_list:
        pares1 = set()
        for par in combinations(comu,2):
            pares1.add(frozenset(par))
        pares_comus1.append(pares1)
        
        
    a11 = 0
    a00 = 0
    for set_pares in pares_comus1:
        for comu in comus2_list:
            for par in combinations(comu,2):
                if set(par) in set_pares:
                    a11 += 1
                    
    np1 = pares(comus1_list)
    np2 = pares(comus2_list)
    
    for par1 in np1:
        if par1 in np2:
            a00 += 1
    
    return (a11 + a00)/(n*(n-1)/2)

#------------------------------------------------------------------------------

# Carga de la red
red_delf = nx.read_gml('./Datos/dolphins.gml')

# Comunidades
comus_infomap = clusterize(red_delf, "infomap")
comus_louvain = clusterize(red_delf, "louvain")
comus_edgeb = clusterize(red_delf, "edge_betweenness")
comus_fg = clusterize(red_delf, "fastgreedy")


