import glob, sys, time
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sn
import pandas as pd

import seaborn as sns
import math

from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib as mpl

def parsing(file): #aux function to parse file correctly
    list = []
    f = open(file, "r")
    s = f.read()
    f.close()
    results_list = s.strip().split("\n")
    for res in results_list:
        elements = res.strip().split(",")[-1].split(" ")
        list.append(elements)
    return list

#Params for parsing
num_pixels = "5" #1,3,5
neural_net = "vgg16" #vgg16,NiN,allcnn
directory = "targeted_saves/" #non-targeted_saves,targeted_saves
searching_path=directory+"*_"+num_pixels+"_"+neural_net+".txt"
files_to_parse = glob.glob(searching_path)

result_list = []
for file in files_to_parse:
    result_list.append(parsing(file))

Success_dic = {}
Fail_dic = {}
classes = []
rateo = {}
if(len(result_list) == 0): #nothing to parse, exit
    print("Houston, we got a problem. There is no file to parse... try to change params in Results_Parser.py")
    raise SystemExit(0)

#actual parsing, fill dictionaries with correct data
if directory == "non-targeted_saves/":
    for entry in result_list:
        for elem in entry:
            if elem[0] == "Success":
                if elem[1] in Success_dic.keys():
                    Success_dic[elem[1]].append(elem[2])
                else:
                    classes.append(str(elem[1]))
                    Success_dic[elem[1]]= [elem[2]]
            else:
                if elem[1] in Fail_dic.keys():
                    Fail_dic[elem[1]] += 1
                else:
                    classes.append(str(elem[1]))
                    Fail_dic[elem[1]] = 1

    classes = list(set(classes)) #avoid duplicates and sort
    classes.sort()
    x1 = []
    x2 = []
    for elem in classes:
        #rateo.append(len(Success_dic[elem]),Fail_dic[elem])
        if elem not in Success_dic.keys():
            rateo[elem]=([0,Fail_dic[elem]])
        elif elem not in Fail_dic.keys():
            rateo[elem]=([len(Success_dic[elem]),0])
        else:
            rateo[elem]=([len(Success_dic[elem]),Fail_dic[elem]])
        x1.append(rateo[elem][0])
        x2.append(rateo[elem][1])

    #creating graph, populating it
    ax = plt.subplot(111)
    ax.set_title('Successes and failure of '+num_pixels+' pixels on '+neural_net)
    y_pos = np.arange(len(classes))
    p1 = plt.bar(y_pos, x1, width = 0.5)
    p2 = plt.bar(y_pos, x2, align='edge', width= 0.5)
    plt.xticks(y_pos, classes, rotation= 'horizontal')

    plt.xlabel('Classes', fontsize=18)
    plt.ylabel('Occurences', fontsize=18)

    locs, labels = plt.yticks()
    #modify yticks s.t. only integer values shown
    yint = []
    for each in locs:
        yint.append(int(each))
    plt.yticks(yint)

    #save and show graph
    plt.legend((p1[0], p2[0]), ('Successes', 'Failure'))

    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig("parsed_%d_" % time.time() +num_pixels+"_"+neural_net)

else:
    classes_dict = { 0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}
    classes = []
    for e1 in classes_dict:
        classes.append(classes_dict[e1])

    rateo = {}
    for entry in result_list:
        for elem in entry:
            if elem[0] == "Success":
                if (elem[1],elem[2]) in Success_dic.keys():
                    Success_dic[(elem[1],elem[2])].append(elem[3])
                else:
                    Success_dic[(elem[1],elem[2])] = [elem[3]]
            else:
                if (elem[1],elem[2]) in Fail_dic.keys():
                    Fail_dic[(elem[1],elem[2])] += 1
                else:
                    Fail_dic[(elem[1],elem[2])] = 1

    for elem1 in classes:
        for elem2 in classes:
            if (elem1,elem2) not in Success_dic.keys() and (elem1,elem2) not in Fail_dic.keys():
                rateo[(elem1,elem2)]=([0,0])
            elif (elem1,elem2) not in Fail_dic.keys():
                    rateo[(elem1,elem2)]=([len(Success_dic[(elem1,elem2)]),0])
            elif (elem1,elem2) not in Success_dic.keys():
                rateo[(elem1,elem2)]=([0,Fail_dic[(elem1,elem2)]])
            else:
                rateo[(elem1,elem2)]=([len(Success_dic[(elem1,elem2)]),Fail_dic[(elem1,elem2)]])
                num_tries = Fail_dic[(elem1,elem2)] + len(Success_dic[(elem1,elem2)])
    #print(rateo)

    final_list =[]
    i = 0
    while(i<=9):
        a=[]
        a.append(rateo[(classes_dict[i],classes_dict[0])][0])
        for cl in classes_dict:
            if classes_dict[cl] != classes_dict[0]:
                if classes_dict[cl] == classes_dict[i] or (classes_dict[i],classes_dict[cl]) not in rateo.keys():
                    a.append(0)
                else:
                    a.append(rateo[(classes_dict[i],classes_dict[cl])][0])

        final_list.append(a)
        i+=1

    mpl.style.use('seaborn')
    conf_arr = np.array(final_list)
    df_cm = pd.DataFrame(conf_arr,
    index = ["airplane","automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    columns = ["airplane","automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])
    fig = plt.figure()

    plt.clf()

    ax = fig.add_subplot(111)
    ax.set_aspect(1)

    cmap = sns.cubehelix_palette(light=1, as_cmap=True, reverse=True)
    res = sn.heatmap(df_cm, annot=True, vmin=0.0, vmax=num_tries, fmt='.2f', cmap=cmap)

    res.invert_yaxis()

    plt.title('Confusion Matrix of results')
    save_string = 'confusion_matrix_'+num_pixels+'_'+neural_net+'.png'
    plt.savefig(save_string, dpi=100, bbox_inches='tight' )
    plt.show()
    plt.close()

#Done, printing results
print("Successes")
print(Success_dic)
print("\nFails")
print(Fail_dic)
print("\nWE DID IT. Now about the answer to everything, life, the univers--TRASMISSION ENDED.")
