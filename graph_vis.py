import argparse
import numpy as np
import pandas as pd
import networkx as nx
from pyvis.network import Network 

# Generate an HTML pyvis file that visualizes spinal chord graph from a Adjacency matrix

# EXPL: python graph_vis.py -a data/resources/fc.npy -t 0.5

def generate_pos(roi_names_slevels, spacing=50):
    dict_pos = {}
    
    all_levels = [int(roi.split('_')[0][1]) for roi in roi_names_slevels]
    uall_levels = np.unique(all_levels)
    rank = np.arange(len(uall_levels))

    for i, roi in enumerate(roi_names_slevels):
        x, y = 0, 0
        name_contraction = "".join(roi.split('_')[-2:])
        # x coordinate
        if name_contraction in ['VR','FCR']:
            x = 2
        elif name_contraction in ['IR','DR','FGR']:
            x = 3
        elif name_contraction == 'I':
            x = 4
        elif name_contraction in ['IL','DL','FGL']:
            x = 5
        elif name_contraction in ['VL','FCL']: 
            x = 6
        elif name_contraction in ['SLR','CSTR']:
            x = 1
        elif name_contraction in ['SLL','CSTL']:
            x = 7
       
        # y coordinate -- don't care about L/R 
        if name_contraction in ['VR','VL']:
            y = 2
        elif name_contraction in ['DR','DL']:
            y = 4
        elif name_contraction in ['CSTR','IR','I','IL','CSTL']:
            y = 3
        elif name_contraction in ['SLR','SLL']:
            y = 1
        elif name_contraction in ['FCR','FGR','FGL','FCL']:
            y = 5

        # according to the level adjust x+4 and y+5
        idx_lev = int(roi.split('_')[0][1]) == uall_levels
        yadd = rank[idx_lev][0]*5
        y += yadd
        
        dict_pos[i] = [spacing*x,spacing*y]
    
    return dict_pos


def plot_spinal_graph(A, rois, dict_pos, thresh=None):
    
    ## A = graph adjecency matrix 
    ## rois = names / labels of the nodes
    ## dict_pos = dictionary of the coordinates
    ## thresh = threshold value for the edges

    # create dictionary 
    dict_id2name = dict(zip(range(len(rois)),rois))
    dict_id2level = dict()
    for i in range(len(dict_id2name)):
        dict_id2level[i] = dict_id2name[i][1]

    Anosl = A-np.diag(np.diag(A))  # no self-loops
    G = nx.from_numpy_array(Anosl)
    edgeslist = list(G.edges(data=True))

    nt = Network('700px', '900px', directed=False, filter_menu=False)
    nt.show_buttons(filter_=['edges'])
    # Disable physics globally for the whole graph
    nt.barnes_hut(gravity=0, central_gravity=0, spring_length=0, spring_strength=0)

    colors = iter(['green','yellow','orange','violet','maroon','grey','black','pink','cyan'])
    c = next(colors)
    lev = dict_id2level[0]
    # add nodes (colored by level)

    for i in G.nodes:
        # color by level
        if i>0:
            levn = dict_id2level[i]
            if levn != lev:
                # change color only if new level
                c = next(colors)
            lev = levn
        
        pos = dict_pos[i]
        nt.add_node(i, label=dict_id2name[i], color=c,physics=False, x=float(pos[0]),y=float(pos[1]))
        nt.nodes[i]['group'] = dict_id2level[i]

    if thresh!=None:
        # add edges according to threshold
        for edge in edgeslist:
            w = edge[-1]['weight']
            if np.abs(w) > thresh:
                c = 'blue' if w<0 else 'red'
                nt.add_edge(edge[0],edge[1],weight=w,title=w,color=c,value=abs(w))  # arrows='none;
    else:
        for edge in edgeslist:
            w = edge[-1]['weight']
            c = 'blue' if w<0 else 'red'
            nt.add_edge(edge[0],edge[1],weight=w,title=w,color=c,value=abs(w))  # arrows='none;
    
    nt.save_graph('plots/graph.html')
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Plot Spinal Cord Graph")

    parser.add_argument('-a', '--adjacency_path', type=str, help='A matrix path', required=True)
    parser.add_argument('-l', '--labels_path', type=str, help='Regions of interest path', default="data/resources/pam50_atlas.csv")
    parser.add_argument('-t', '--threshold', type=float, help='Threshold for considering Edge', default=None)
    args = parser.parse_args()
    
    A = np.load(args.adjacency_path)
    labels = pd.read_csv(args.labels_path)["regions_of_interest"].tolist()
    dict_pos = generate_pos(labels)
    plot_spinal_graph(A, labels, dict_pos, args.threshold)

