import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn_hierarchical.classifier import HierarchicalClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt,pylab
import networkx as nx
import dill
import math
from networkx.drawing.nx_agraph import write_dot, graphviz_layout,pygraphviz_layout
from collections import Counter
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from nltk import ngrams
from sklearn.metrics.pairwise import cosine_similarity
import copy

x=float('nan')
math.isnan(x)
global ics_map_old_new
global counter
counter=0
ics_map_old_new={} # create an unique integer label for each node/class and map it back to the the ics labels

# todo: there is some incongruity when the code column does not follow the usual rules

def savemodel(model,outfile):
    with open(outfile, 'wb') as output:
        dill.dump(model, output)
    return ''

def loadmodel(infile):
    model=''
    with open(infile, 'rb') as inp:
        model = dill.load(inp)
    return model


def getID(key):
    global counter
    global ics_map_old_new
    if key not in ics_map_old_new.keys():
        counter += 1
        ics_map_old_new[key] = counter

    return ics_map_old_new[key]

# todo: fix the subgroup thing (rerun)

# g=nx.DiGraph()
# g.add_node('D',abc='def')
# g.add_edge('A','B')
# g.add_edge('B','C')
# g.add_edge('B','D')
# for n in g['B']:
#     print(n,g.node[n])
#     g.node[n]['probability']=0.0
# print(g.node['D'])
# exit()


input="the standard is based on electric cars hardware software"
threshold=0.2


# ================================================== do training =======================================================
# ======================================================================================================================
# ======================================================================================================================










# ================================================== data preparation ==================================================

iso_path='/Users/asitangm/Desktop/iso_flat.json'
ics_path='/Users/asitangm/Desktop/ics.csv'

df_ics=pd.read_csv(ics_path)

# create the class tree. Have a mapping to actual names of the categories (can get from querying the ics.csv directly).
df_ics_seperated=pd.DataFrame()


ics_dict={-1:[]} # create a taxonomy to input into the Hclassif algo
for i, row in df_ics.iterrows():

    new_row={}
    code=row['code']
    code=code.split('.')
    field,group,sub_group,new_group,new_field,standard,new_standard,new_sub_group='','','','','','','',''

    if len(code)>=2:
        field=code[1]
        new_field = getID(field)
        ics_dict[-1].append(new_field)

    if len(code)>=3:
        group=code[1]+'.'+code[2]
        new_group = getID(group)

        if new_field not in ics_dict.keys():
            ics_dict[new_field]=[]

        ics_dict[new_field].append(new_group)

    if len(code)==5:
        sub_group=code[3]
        new_sub_group = getID(sub_group)
        standard = code[4]
        new_standard = getID(standard)

    if len(code)==4:
        if 'ISO' in code[3]:
            standard=code[3]
            new_standard = getID(standard)
        else:
            sub_group=code[3]
            new_sub_group = getID(sub_group)


    new_row['field'] = field
    new_row['new_field'] = new_field
    new_row['group'] = group
    new_row['new_group'] = new_group
    new_row['subgroup'] = sub_group
    new_row['new_subgroup'] = new_sub_group
    new_row['standard'] = standard
    new_row['new_standard'] = new_standard
    new_row['code'] = row['code']
    new_row['link'] = row['link']
    new_row['title'] = row['title']




    df_ics_seperated=df_ics_seperated.append(new_row, ignore_index=True)
    print(i)

df_ics_seperated.to_csv('ics_separated.csv') # save all things into a csv so that later on lables, ics labels and names could be correlated
savemodel(ics_dict,'ics_dict')
savemodel(ics_map_old_new,'ics_map_old_new')
exit()



# ================================================== vectorization =====================================================


ics_dict=loadmodel('ics_dict')
ics_map_old_new=loadmodel('ics_map_old_new')
# print(ics_dict,ics_map_old_new)

# todo: remove duplicates; standards with multiple paths ?

df = pd.read_json(iso_path)
df.to_csv('json_to_csv.csv')

X=[]
y=[]
for i, row in df.iterrows():

    text=row.description+'. '+row.title
    for ics in row.ics:
        label='.'.join(ics.split('.')[:2])
        X.append(text)
        y.append(ics_map_old_new[label])

# remove the class that occurs only once because that leads to problems during stratification
counted_y=Counter(y)

X_=[]
y_=[]
for _X,_y in zip(X,y):
    if counted_y[_y]>=2:
        X_.append(_X)
        y_.append(_y)

X=X_
y=y_


df_ics_seperated=pd.read_csv('ics_separated.csv',index_col=0)



# vectorize the data
cv=CountVectorizer()
X=cv.fit_transform(X).todense()
y=np.array(y)

print(X.shape,y.shape)

# stratify the data (second level up here, need only to stratify at the very last level)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.9, random_state=0)
for train_index, test_index in sss.split(X, y):
    print(train_index,test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# # ================================================== fit model tree/graph ============================================

be = MLPClassifier()
clf = HierarchicalClassifier(
        base_estimator=be,
        class_hierarchy=ics_dict,
    )
print('fitting')
clf.fit(X_train, y_train)

savemodel(clf,'clf')
savemodel(cv,'cv')
exit()

# ================================================== add to the graph, the leaf nodes (actual standards) ===============

clf=loadmodel('clf')
graph=clf.graph_

ics_map_old_new=loadmodel('ics_map_old_new')
df = pd.read_json(iso_path)
df_ics_seperated=pd.read_csv('ics_separated.csv',index_col=0)
for i, row in df.iterrows():

    text=row.description+'. '+row.title
    standardid=row.id
    # print('stid',standardid)
    new_standardid=df_ics_seperated[df_ics_seperated['standard']==standardid]['new_standard']
    if len(new_standardid)>0:
        new_standardid=int(new_standardid.values[0])
        # print('new stdid!', new_standardid)
    else:
        print('not found!',standardid)
        continue
    for ics in row.ics:
        # add to tree/graph's lowest node (group here)
        label='.'.join(ics.split('.')[:2])
        if label in ics_map_old_new.keys():
            new_label=ics_map_old_new[label]
        else:
            print(label,'not found in the dict!')
            continue
        graph.add_node(new_standardid, text=text, title=row.id, probability=0.0)
        graph.add_edge(new_label, new_standardid)


# ================================================== Add 'level' attribute to the nodes ================================

def add_level(graph,node,level):
    graph.node[node]['level']=level
    for _,child in graph.out_edges(node):
        add_level(graph,child,level+1)

    return

add_level(graph,-1,0)
savemodel(clf,'clf') # saves the graph as well
exit()













# ================================================== create the prediction tree/graph ==================================
# ======================================================================================================================
# ======================================================================================================================












# ================================================== do classification and add probabilities to noes (non leaf) ========

clf=loadmodel('clf')
cv=loadmodel('cv')
graph=loadmodel('graph')
graph=copy.deepcopy(graph)
input=cv.transform([input])
clf.recursive_predict_all_paths(input,levelstop=2)
graph=clf.graph_
savemodel(graph,'graph')



# ================================================== do similarity and add probability to leaf nodes ===================

input=cv.transform([input])
selected_leaf_nodes=[]
to_remove=[]
for node in graph:
    if graph.node[node]['level']==3:
        flag=0
        for parent,_ in graph.in_edges(node): # if the parent is above threshold then we calculate probability for that node
            if 'probability' not in graph.node[parent]: # todo: why there is no prob in some!! BUG
                print('no proba found!!',node)
                continue
            if graph.node[parent]['probability']>threshold:
                flag=1
                break
        if flag==1:
            print('similarity for: ',node)
            text=graph.node[node]['text']
            text=cv.transform([text])
            prob=cosine_similarity(input,text)[0][0]
            graph.node[node]['probability']=prob
            selected_leaf_nodes.append(node)
        else:
            to_remove.append(node)

# also remove the poor leaf nodes so that the visualizer runs faster!
print('nodes in graph',len(graph.node))
graph.remove_nodes_from(to_remove)
print('nodes in graph after pruning',len(graph.node))
savemodel(graph,'graph')
savemodel(selected_leaf_nodes,'selected_leaf_nodes')
exit()












# ================================================== visualize the prediction tree/graph ===============================
# ======================================================================================================================
# ======================================================================================================================

selected_leaf_nodes=loadmodel('selected_leaf_nodes')
graph=loadmodel('graph')
print('will create graph now...')
file_name='graph.png'
pos =graphviz_layout(graph, prog='dot')

top_nodes=[]
rest_nodes=[]
top_label={}
top_edges=[]
rest_edges=[]


# go through the graph and include any edges that are part of a path that has a 'good' node (above threshold)
print('number of total selected leaf nodes',len(selected_leaf_nodes))
paths=[]
print('finding interesting paths.....')
for node in selected_leaf_nodes:
    paths.append(nx.shortest_path(graph, -1, node))

print('number of total paths',len(paths))

for count, path in enumerate(paths):
    flag=0
    count=0
    for node in path:
        if 'probability' in graph.node[node].keys() and graph.node[node]['probability']>=threshold:
            top_nodes.append(node)
            top_label[x] = str(x)
            flag=1
            count+=1

    if flag==1: # if atleast one good node found on the path
        # select edges
        if count>=2:
            top_edges.extend(list(ngrams(path, 2)))
        else:
            rest_edges.extend(list(ngrams(path, 2)))
        # add the rest of the nodes on that path
        for node in path:
            if node not in top_nodes:
                rest_nodes.append(node)




print('drawing...')
nx.draw_networkx_edges(graph, pos, edgelist=rest_edges, edge_color='y')
nx.draw_networkx_edges(graph, pos, edgelist=top_edges, edge_color='r')
nx.draw_networkx_nodes(graph, pos, nodelist=rest_nodes, node_color='g', node_size=5)
nx.draw_networkx_nodes(graph, pos, nodelist=top_nodes, node_color='b', node_size=50)
# nx.draw_networkx_labels(graph, pos,labels=top_label,font_size=5)

cut = 1.00
xmax = cut * max(xx for xx, yy in pos.values())
ymax = cut * max(yy for xx, yy in pos.values())
plt.xlim(0, xmax)
plt.ylim(0, ymax)

print('saving figure...')
plt.savefig(file_name)
pylab.close()
