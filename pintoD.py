import numpy as np
import networkx as nx
import matplotlib.pylab as plt
from random import shuffle
from copy import deepcopy
import igraph
from collections import Counter
#from itertools import combinations
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


def atributoNodos(r, alist, atributo):
# Toma como argumentos una red, una lista de listas, donde cada una de ellas
# indica el atributo que se le va a asignar a cada nodo de la red, y el 
# atributo que uno quiere asignar. Devuelve la red con ese atributo ya asociado
# a cada nodo.
    for idx, nodo in enumerate(np.array(alist).transpose()[0]):
        r.nodes[nodo][atributo] = np.array(alist).transpose()[1][idx]
    return


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
    gen_red = dict()
    a = list(nx.get_node_attributes(r, 'gender').values())
    gen_red['m'] = a.count('m')
    gen_red['f'] = a.count('f')
    gen_red['NA'] = a.count('NA')
    return gen_red


def combinatorio(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

#------------------------------------------------------------------------------

def poblacionComus(particion):
# IN: Una partición en forma de diccionario.
# OUT: Un diccionario donde las keys son las comunidades y los values son el
# número de miembros de cada comunidad.
    values = particion.values()
    return Counter(values)

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


def generoAzarComus(red, particion, iters):
    lista = []
    for i in range(iters):
        rr = generoAzar(red)
        lista.append(np.array(poblacionAtributoComus(rr,particion)))
    #lista = np.swapaxes(np.array(lista),0,1)
    return lista

def datosGenComu(red, particion, iters):
    dat0 = generoAzarComus(red, particion, iters)
    dat = []
    for comu in range(len(set(particion.values()))):
        gencomu = []
        for i in range(iters):
            gencomu.append((dat0[i][comu]['m'], dat0[i][comu]['f']))
        dat.append(gencomu)
    return dat
 
# -----------------------------------------------------------------------------
    
# HISTOGRAMAS
# Con esta funcion se crean las listas de datos aleatorios que voy a necesitar
# para hacer los histogramas, una lista de listas para los machos y una para 
# las hembras. Dentro de la lista de machos (hembras) hay listas donde están 
# los números de machos (hembras) que se obtuvieron en las iteraciones, para 
# una dada comunidad. 

def listasGenComu(red, particion, iters):
    dat0 = datosGenComu(red, particion, iters)
    dat_machos = []
    dat_hembras = []
    for comu in range(len(set(particion.values()))):
        mcomu = []
        hcomu = []
        for i in range(iters):
            mcomu.append(dat0[comu][i][0])
            hcomu.append(dat0[comu][i][1])
        dat_machos.append(mcomu)
        dat_hembras.append(hcomu)
    return dat_machos, dat_hembras

#------------------------------------------------------------------------------

# TEST DE FISHER

def hipergeometrica(N, r, k, m):
    p = combinatorio(k, m)*combinatorio(N-k, r-m)/combinatorio(N, r)
    return p
    
# Esta función hace el test de Fisher para todas las comunidades de una dada 
# partición. Hay que darle el atributo dicotómico para que haga el test según
# esa variable. Por ej: Si el atributo es 'f', me va a decir, para cada 
# comunidad, qué tanta probabilidad hay de obtener el número de hembras que 
# tengo, o uno más grande, asumiendo que están distribuídos al azar (esto es el
# p-value). Si esta probabilidad es muy chica, significa que debe existir una
# correlación entre esa comunidad y la cantidad de machos que hay en ella y que
# la hipótesis de que ese número viene del azar es poco probable de que sea 
# verdadera.
def testFisherParticion(red, particion, atributo):
    poblaciones = poblacionComus(particion)
    distGenerosComus = poblacionAtributoComus(red, particion)
    N = red.number_of_nodes()
    k = contadorGenero(red)[atributo]
    pval_comus = []
    for comu in poblaciones.keys():
        suma = 0
        r = poblaciones[comu]
        m = distGenerosComus[comu][atributo]
        for i in range(m, r+1):
            suma += hipergeometrica(N, r, k, i)
            #print(suma)
            print(N-k, r-i)
        pval_comus.append(suma)
    return pval_comus
   
#------------------------------------------------------------------------------         

# Carga y tratamiento de datos.
red_delf = nx.read_gml('./Datos/dolphins.gml')
gen_delf = open('./Datos/dolphinsGender.txt').readlines()

sex_delf = []   
for i in range(len(gen_delf)):
    a = gen_delf[i].rstrip('\n').split('\t')
    sex_delf.append(a)
    
atributoNodos(red_delf, sex_delf, 'gender')


# Comunidades
comus_infomap = clusterize(red_delf, "infomap")
comus_louvain = clusterize(red_delf, "louvain")
comus_edgeb = clusterize(red_delf, "edge_betweenness")
comus_fg = clusterize(red_delf, "fastgreedy")


# Datos para los histogramas:
machosAzar_louvain = listasGenComu(red_delf, comus_louvain, 5000)[0]
machosAzar_infomap = listasGenComu(red_delf, comus_infomap, 5000)[0]
machosAzar_edgeb = listasGenComu(red_delf, comus_edgeb, 5000)[0]
machosAzar_fg = listasGenComu(red_delf, comus_fg, 5000)[0]

hembrasAzar_louvain = listasGenComu(red_delf, comus_louvain, 5000)[1]
hembrasAzar_infomap = listasGenComu(red_delf, comus_infomap, 5000)[1]
hembrasAzar_edgeb = listasGenComu(red_delf, comus_edgeb, 5000)[1]
hembrasAzar_fg = listasGenComu(red_delf, comus_fg, 5000)[1]

# Confección de histogramas:
x = machosAzar_louvain[4]
plt.hist(x, bins=12, normed=True)
#plt.title()
plt.show()

#histhomo, bin_edges = np.histogram(x, bins='scott', density=True)
#bin_edges = (bin_edges[:-1] + bin_edges[1:])/2.
#plt.plot(bin_edges, histhomo,'bo')
#plt.title("Distribución nula")
#plt.show()