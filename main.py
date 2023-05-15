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
    Gh = apply_op(img,[[-1,-2,-1],[0,0,0],[1,2,1]])
    Gv = apply_op(img,[[-1,0,1],[-2,0,2],[-1,0,1]])
    G = (Gh*Gh+Gv*Gv)**0.5
    return G

def prewitt_apply(img):
    Gh = apply_op(img,[[-1,-1,-1],[0,0,0],[1,1,1]])
    Gv = apply_op(img,[[-1,0,1],[-1,0,1],[-1,0,1]])
    G = (Gh*Gh+Gv*Gv)**0.5
    return G

def apply_op(img,m,size=3):
    result = []
    size_per_2 = size//2
    for i in range(size_per_2,img.shape[0]-size_per_2):
        line = []
        for j in range(size_per_2,img.shape[1]-size_per_2):
            soma = sum(sum(m*img[i-size_per_2:i+size_per_2+1,j-size_per_2:j+size_per_2+1]))
            line.append(soma)
        result.append(line)
    return np.array(result)

def gauss_filter(size,sigma):
    gauss = np.zeros(shape=(size,size))
    for i in range(size):
        for j in range(i+1):
            gauss[i][j] = np.exp(-((i-size//2)**2 + (j-size//2)**2) / (2 * sigma**2))
    for i in range(size):
        for j in range(i+1,size):
            gauss[i][j] = gauss[j][i]
    gauss /= np.sum(gauss)
    return gauss


def main():
    files = get_files("img")
    for file in files:
        img = read_image(file)
        intensity = img_intensity(img)
        teste = gauss_filter(5,1.4)
        #sobel = sobel_apply(intensity)
        #prewitt = prewitt_apply(intensity)
        teste = apply_op(intensity,teste,5)
        make_new_image(teste,file)
    
if __name__ == "__main__":
    main()