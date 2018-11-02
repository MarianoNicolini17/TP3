import networkx as nx
import igraph
#import numpy as np
from collections import Counter
import math


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

def intersect(l1, l2): 
# Intersección entre dos listas.
    l3 = [value for value in l1 if value in l2] 
    return l3

def poblacionComus(particion):
# IN: Una partición en forma de diccionario.
# OUT: Un diccionario donde las keys son las comunidades y los values son el
# número de miembros de cada comunidad.

    values = particion.values()
    return Counter(values)


def informacionMutua(p1,p2):
# IN: Dos particiones de la misma red, en forma de diccionarios.
# OUT: Información mutua normalizada.    
    
    # En estas líneas calculo las probabilidades, para cada partición, de que   
    # un nodo elegido al azar de la red pertenezca a una cierta comunidad. 
    # En las listas probs_p1 y probs_p2, están dichas probabilidades.
    probs_p1 = []
    probs_p2 = []
    prob_conj = []
    comus1 = poblacionComus(p1)
    comus2 = poblacionComus(p2)
    n = len(p1)
    
    for comu in sorted(comus1.keys()):
        prob = comus1[comu]/n
        probs_p1.append(prob)
    for comu in sorted(comus2.keys()):
        prob = comus2[comu]/n
        probs_p2.append(prob)
        
    # En estas lineas hago dos listas de listas. Cada lista, tiene adentro una
    # lista con los miembros de cada comunidad.    
    comus1_list = []
    comus2_list = []
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
    
    # En estas líneas calculo la probabilidad conjunta pij de que un dado nodo 
    # pertenezca a la comunidad i de la primera partición y a la comunidad j de
    # la segunda partición. Estas probabilidades están en la lista de listas 
    # (o también podemos llamarle Matriz de coaparición normalizada), prob_conj.
    for pobi_list in comus1_list:
        pij = []
        for pobj_list in comus2_list:
            pij.append(len(intersect(pobi_list,pobj_list))/n)
        prob_conj.append(pij)
    
    # En las líneas que quedan, hago el cálculo de la información mutua, y 
    # entropías de Shannon (me encanta decirlo) para devolver la información
    # mutua normalizada.
    I = 0
    for i in range(len(comus1.keys())):
        for j in range(len(comus2.keys())):
            if prob_conj[i][j] == 0:
                I += 0
            else:
                I += prob_conj[i][j]*math.log(prob_conj[i][j]/(probs_p1[i]*probs_p2[j]),2)
            
    H1 = 0
    for i in range(len(comus1.keys())):
        H1 += -probs_p1[i]*math.log(probs_p1[i],2)
        
    H2 = 0
    for i in range(len(comus2.keys())):
        H2 += -probs_p2[i]*math.log(probs_p2[i],2)
        
    return 2*I/(H1 + H2)
# Para más información, el paper de Community detection in graphs - Fortunato:
# https://arxiv.org/abs/0906.0612

#------------------------------------------------------------------------------

# Carga de la red
red_delf = nx.read_gml('./Datos/dolphins.gml')

# Comunidades
comus_infomap = clusterize(red_delf, "infomap")
comus_louvain = clusterize(red_delf, "louvain")
comus_edgeb = clusterize(red_delf, "edge_betweenness")
comus_fg = clusterize(red_delf, "fastgreedy")


        
    
            
            
        
    
        
            

    
        
        