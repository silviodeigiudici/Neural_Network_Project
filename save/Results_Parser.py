import glob,sys

def parsing(file):
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
num_pixels = "3" #1,3,5
neural_net = "vgg16" #vgg16,NiN,allcnn
directory = "non-targeted_saves/" #non-targeted_saves,targeted_saves
searching_path=directory+"*_"+num_pixels+"_"+neural_net+".txt"
files_to_parse = glob.glob(searching_path)

result_list = []
for file in files_to_parse:
    result_list.append(parsing(file))

Success_dic = {}
Fail_dic = {}

if(len(result_list) == 0):
    print("Houston, we got a problem. There is no file to parse... try to change params in Results_Parser.py")
    raise SystemExit(0)

if directory == "non-targeted_saves/":
    for entry in result_list:
        for elem in entry:
            if elem[0] == "Success":
                if elem[1] in Success_dic.keys():
                    Success_dic[elem[1]].append(elem[2])
                else:
                    Success_dic[elem[1]]= [elem[2]]
            else:
                if elem[1] in Fail_dic.keys():
                    Fail_dic[elem[1]] += 1
                else:
                    Fail_dic[elem[1]] = 1
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
    
print("Successes")
print(Success_dic)
print("\nFails")
print(Fail_dic)
print("\nWE DID IT. Now about the answer to everything, life, the univers--TRASMISSION ENDED.")
