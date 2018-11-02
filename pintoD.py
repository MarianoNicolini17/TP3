import numpy as np
import networkx as nx
import matplotlib.pylab as plt
from urllib.request import urlopen
from random import shuffle
from copy import deepcopy
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


red_delf = nx.read_gml('./Datos/dolphins.gml')
gen_delf = open('./Datos/dolphinsGender.txt').readlines()

# Comunidades
comus_infomap = clusterize(red_delf, "infomap")
comus_louvain = clusterize(red_delf, "louvain")
comus_edgeb = clusterize(red_delf, "edge_betweenness")
comus_fg = clusterize(red_delf, "fastgreedy")

    
sex_delf = []   
for i in range(len(gen_delf)):
    a = gen_delf[i].rstrip('\n').split('\t')
    sex_delf.append(a)
    
    
    
    
   
def poblacionAtributoComus(red, particion):
# IN: Una partición en forma de diccionario.
# OUT: Un diccionario donde las keys son las comunidades y los values son el
# número de miembros de cada comunidad.
    c = [{'m': 0, 'f': 0, 'NA': 0} for i in set(particion.values())]
    for nodo, comu in particion.items():
        c[comu]['m'] += red.nodes[nodo]['gender'] == 'm'
        c[comu]['f'] += red.nodes[nodo]['gender'] == 'f'
        c[comu]['NA'] += red.nodes[nodo]['gender'] == 'NA'
    return c




def atributoNodos(r, alist, atributo):
# Toma como argumentos una red, una lista de listas, donde cada una de ellas
# indica el atributo que se le va a asignar a cada nodo de la red, y el 
# atributo que uno quiere asignar. Devuelve la red con ese atributo ya asociado
# a cada nodo.
    for idx, nodo in enumerate(np.array(alist).transpose()[0]):
        r.nodes[nodo][atributo] = np.array(alist).transpose()[1][idx]
    return
atributoNodos(red_delf, sex_delf, 'gender')


def generoAzar(r):
# Toma una red donde sus nodos tienen el atributo "genero" y lo distribuye al
# azar. 
# Estaría bueno generalizarlo después para cualquier atributo.
    ng = contadorGenero(r)
    n = list(r.nodes)
    shuffle(n)
    ra = deepcopy(r)
    for i in range(ng[0]):
        ra.nodes[n[i]]['gender'] = 'm'
    for i in range(ng[0], ng[0]+ng[1]):
         ra.nodes[n[i]]['gender'] = 'f'
    for i in range(ng[0]+ng[1], ng[0]+ng[1]+ng[2]):
         ra.nodes[n[i]]['gender'] = "NA"
    return ra


def contadorGenero(r): #Generalizarlo para cualquier atributo
    a = list(nx.get_node_attributes(r, 'gender').values())
    return a.count('m'), a.count('f'), a.count('NA')