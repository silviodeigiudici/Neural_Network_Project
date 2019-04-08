import glob, sys, time
import matplotlib.pyplot as plt

import numpy as np

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
neural_net = "allcnn" #vgg16,NiN,allcnn
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
#Done, printing results
print("Successes")
print(Success_dic)
print("\nFails")
print(Fail_dic)
print("\nWE DID IT. Now about the answer to everything, life, the univers--TRASMISSION ENDED.")

 #plt.show()
