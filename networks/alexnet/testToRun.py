import sys
sys.path.append('../alexnet/')
from alexnet_imagenet import alexnet
from imagenet_mng import manager_imagenet

def main():
    images_imagenet = manager_imagenet("./our_labels.txt", "alexnet_images/", False)
    num = 0
    img = images_imagenet.getImgByNum(num)
    img_clas = images_imagenet.getClassByNum(num)
    model = alexnet()
    pred = model.predict(img)
    print("\n\n")
    print(pred)
    print("\n")
    idx = model.getIdxMaxPred(pred)
    print("Indice max: " + str(idx))
    netClass = str(model.getClassByNum(idx))
    print("List of Nums: " +str(images_imagenet.getListNums()) +'\n')
    print("Network class result: " + netClass)
    print("Network label result: "+ str(images_imagenet.getLabelByClass(netClass)))
    print("Real class: " + str(img_clas))
    print("Num given class: " +str(images_imagenet.getNumByClass(img_clas)))
    print("Real label: " + str(images_imagenet.getLabelByClass(img_clas)))

if __name__== "__main__":
    main()
