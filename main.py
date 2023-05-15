from PIL import Image
import numpy as np
import os

def get_files(dir):
    arqs = os.listdir(dir)

    return [os.path.join(dir,arq) for arq in arqs]

def read_image(file):
        if os.path.isfile(file):
            with Image.open(file, "r") as image:
                pixels = image.getdata()
                return np.array(pixels).reshape(image.size[1], image.size[0], 3)
        else:
            print("not found")

def img_intensity(img):
     return np.array(img[:,:,0]*0.299+img[:,:,1]*0.587+img[:,:,2]*0.114)

def make_new_image(img,file):
# criando um objeto Image a partir da matriz em preto e branco
    new_img = Image.fromarray(img.astype('uint8'), mode='L')
    new_img.save(file[4:]) 
 
def sobel_apply(img):
    Gh = apply_sobel_op(img,[[-1,-2,-1],[0,0,0],[1,2,1]])
    Gv = apply_sobel_op(img,[[-1,0,1],[-2,0,2],[-1,0,1]])
    G = (Gh*Gh+Gv*Gv)**0.5
    return G

def apply_sobel_op(img,m):
    result = []
    for i in range(1,img.shape[0]-1):
        line = []
        for j in range(1,img.shape[1]-1):
            soma = sum(sum(m*img[i-1:i+2,j-1:j+2]))
            line.append(soma)
        result.append(line)
    return np.array(result)


def main():
    files = get_files("img")
    for file in files:
        img = read_image(file)
        intensity = img_intensity(img)
        sobel = sobel_apply(intensity)
        make_new_image(sobel,file)
    
if __name__ == "__main__":
    main()