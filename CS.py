
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
import Layer
class tools:

    def norm(im):
            im=(im+np.ones((im.shape)))
            im=im*100
            im=(im - np.min(im)) / (np.max(im) - np.min(im))
            im=im.astype(np.float16)
            return im


    def display(setting):       
        if np.float32:
            fig, ax = plt.subplots()
            setting=np.round(setting,2)
            minpix,maxpix=setting.shape    
            ax.matshow(setting, cmap=plt.cm.Blues)
            for i in range(minpix):
                for j in range(maxpix):
                    c = setting[j,i]
                    ax.text(i-0.3, j+0.093, str(c), va='bottom', ha='left')
            ax.set_xticks(np.arange(maxpix))
            ax.set_yticks(np.arange(maxpix)) 
            ax.set_axisbelow(False)
            fig = plt.gcf()
            plt.show()

    def decode(box):
        x=box.shape[0]
        y=box.shape[1]
        z=box.shape[2]
        for depth in range(1,z+1):
            sheet=box[:,:,:depth]
            sheet=sheet[:,:,depth-1:,]
            sheet=np.reshape(sheet,((x,y))).astype(np.float16)
            yield sheet
    
    def display3d(setting):
        data = setting
        x = np.indices(data.shape)[0]
        y = np.indices(data.shape)[2]
        z = np.indices(data.shape)[1]
        col = data.flatten()

        # 3D Plot
        ax=plt.subplot()
        fig = plt.figure()
        ax3D = fig.add_subplot(projection='3d')
        p3d = ax3D.scatter(x, y, z)
    def convolute(im, ker, pad=0, stri=1):
                    # Cross Correlation
                ker = np.flipud(np.fliplr(ker))

                # Gather Shapes of ker + im + pad
                xKernShape = ker.shape[0]
                yKernShape = ker.shape[1]
                xImgShape = im.shape[0]
                yImgShape = im.shape[1]

                # Shape of Output Convolution
                xOutput = int(((xImgShape - xKernShape + 2 * pad) / stri) + 1)
                yOutput = int(((yImgShape - yKernShape + 2 * pad) / stri) + 1)
                output = np.zeros((xOutput, yOutput))

                # Apply Equal pad to All Sides
                if pad != 0:
                    imPadded = np.zeros((im.shape[0] + pad*2, im.shape[1] + pad*2))
                    imPadded[int(pad):int(-1 * pad), int(pad):int(-1 * pad)] = im
                    #print(imPadded)
                else:
                    imPadded = im

                # Iterate through im
                for y in range(im.shape[1]):
                    # Exit Convolution
                    if y > im.shape[1] - yKernShape:
                        break
                    # Only Convolve if y has gone down by the specified stri
                    if y % stri == 0:
                        for x in range(im.shape[0]):
                            # Go to next row once ker is out of bounds
                            if x > im.shape[0] - xKernShape:
                                break
                            try:
                                # Only Convolve if x has moved by the specified stri
                                if x % stri == 0:
                                    output[x, y] = (ker * imPadded[x: x + xKernShape, y: y + yKernShape]).sum()                                         
                            except:
                                break
                return output        
    def box(im):
        q=round(im.shape[1]/9)
        output=np.zeros((3,3)).astype(float)
        for j in range(q):        
            for i in range(q):
                x=im[j][i]
                output[j][i]=x
        return output
    def showall():
        for y in range(16):
            circle1=shapes.all[y]
            for x in range(10):
                kern=Layer.Conv.kernels.all[x]
                k=Layer.Conv(circle1,kernel=kern)
                p=k.forward()
                tools.display(p)

class shapes():
                c1=cv2.imread("/home/cam/Downloads/CNN-20230903T210946Z-001/CNN/shapes_I_drew/circle1.png",cv2.IMREAD_GRAYSCALE).astype(np.float32)/ -255
                c2=cv2.imread("/home/cam/Downloads/CNN-20230903T210946Z-001/CNN/shapes_I_drew/circle2.png",cv2.IMREAD_GRAYSCALE).astype(np.float32)/ -255
                c3=cv2.imread("/home/cam/Downloads/CNN-20230903T210946Z-001/CNN/shapes_I_drew/circle3.png",cv2.IMREAD_GRAYSCALE).astype(np.float32)/ -255
                c4=cv2.imread("/home/cam/Downloads/CNN-20230903T210946Z-001/CNN/shapes_I_drew/circle4.png",cv2.IMREAD_GRAYSCALE).astype(np.float32)/ -255
                c5=cv2.imread("/home/cam/Downloads/CNN-20230903T210946Z-001/CNN/shapes_I_drew/circle5.png",cv2.IMREAD_GRAYSCALE).astype(np.float32)/ -255
                c6=cv2.imread("/home/cam/Downloads/CNN-20230903T210946Z-001/CNN/shapes_I_drew/circle6.png",cv2.IMREAD_GRAYSCALE).astype(np.float32)/ -255
                c7=cv2.imread("/home/cam/Downloads/CNN-20230903T210946Z-001/CNN/shapes_I_drew/circle7.png",cv2.IMREAD_GRAYSCALE).astype(np.float32)/ -255
                c8=cv2.imread("/home/cam/Downloads/CNN-20230903T210946Z-001/CNN/shapes_I_drew/circle8.png",cv2.IMREAD_GRAYSCALE).astype(np.float32)/ -255
                s1=cv2.imread("/home/cam/Downloads/CNN-20230903T210946Z-001/CNN/shapes_I_drew/square1.png",cv2.IMREAD_GRAYSCALE).astype(np.float32)/ -255
                s2=cv2.imread("/home/cam/Downloads/CNN-20230903T210946Z-001/CNN/shapes_I_drew/square2.png",cv2.IMREAD_GRAYSCALE).astype(np.float32)/ -255
                s3=cv2.imread("/home/cam/Downloads/CNN-20230903T210946Z-001/CNN/shapes_I_drew/square3.png",cv2.IMREAD_GRAYSCALE).astype(np.float32)/ -255
                s4=cv2.imread("/home/cam/Downloads/CNN-20230903T210946Z-001/CNN/shapes_I_drew/square4.png",cv2.IMREAD_GRAYSCALE).astype(np.float32)/ -255
                s5=cv2.imread("/home/cam/Downloads/CNN-20230903T210946Z-001/CNN/shapes_I_drew/square5.png",cv2.IMREAD_GRAYSCALE).astype(np.float32)/ -255
                s6=cv2.imread("/home/cam/Downloads/CNN-20230903T210946Z-001/CNN/shapes_I_drew/square6.png",cv2.IMREAD_GRAYSCALE).astype(np.float32)/ -255
                s7=cv2.imread("/home/cam/Downloads/CNN-20230903T210946Z-001/CNN/shapes_I_drew/square7.png",cv2.IMREAD_GRAYSCALE).astype(np.float32)/ -255
                s8=cv2.imread("/home/cam/Downloads/CNN-20230903T210946Z-001/CNN/shapes_I_drew/square8.png",cv2.IMREAD_GRAYSCALE).astype(np.float32)/ -255
                circles=[c1,c2,c3,c4,c5,c6,c7,c8]
                squares=[s1,s2,s3,s4,s5,s6,s7,s8]
                c=tools.norm(np.array(circles))
                s=tools.norm(np.array(squares))
                all=[c1,c2,c3,c4,c5,c6,c7,c8,s1,s2,s3,s4,s5,s6,s7,s8]

class data():
    def __init__(self,random=False):
        self.circles=np.concatenate(shapes.c)
        self.squares=np.concatenate(shapes.s)
        self.lableset=np.atleast_2d(np.empty((800)).astype(np.uint))
        self.lableset[:,:400]=(0)
        self.lableset[:,400:]=(1)
        self.MTimarray=np.arange(50,850,50)
        self.x=np.concatenate([self.circles,self.squares],axis=0)
    def random(self,amount_of_samples=1):
        randax=np.random.choice(self.MTimarray)
        i=randax-50
        j=int(self.lableset[0][i])
        y=np.zeros(2).astype(np.uint)
        y[j]=1
        X=np.array(self.x[i:randax,:50])
        return X,y
    def randset(self,amount_of_samples=1):
        X=[]
        y=[]
        for n in range(amount_of_samples):
            randax=np.random.choice(self.MTimarray)
            i=randax-50
            j=int(self.lableset[0][i])
            label=np.zeros(2).astype(np.uint)
            label[j]=1
            X.append(np.array(self.x[i:randax,:50]))
            y.append(label)
        return X,y

class kernels:
    identity=np.array([[0,0,0],
                        [0,1,0],
                        [0,0,0]])
    
    box=np.array  ([[1, 1, 1],
                    [1,-8, 1],
                    [1, 1, 1]])
    
    edge=np.array([[-1,-1,-1],
                    [-1, 8,-1],
                    [-1,-1,-1]])
    
    sharp=np.array([[ 0,-1, 0],
                    [-1, 4,-1],
                    [ 0,-1, 0]])
    
    feature1=np.array([ [-1,-1, 1],
                        [-1, 4,-1],
                        [ 1,-1,-1]])
    
    feature2=np.array([ [ 1,-1, -1],
                        [-1, 4, -1],
                        [-1,-1,  1]])
    
    feature3=np.array([ [ 2,-1,-1],
                        [-1, 2,-1],
                        [-1,-1, 2]])
    
    feature4=np.array([ [ -1,-1, 2],
                        [-1, 2, -1],
                        [ 2,-1, -1]])
    
    feature5=np.array([ [ -1,-1,-1],
                        [  2, 2, 2],
                        [ -1,-1,-1]])
    
    feature6=np.array([ [ -1, 2,-1],
                        [ -1, 2,-1],
                        [ -1, 2,-1]])                            
    all=[identity,box,edge,sharp,feature1,feature2,feature3,feature4,feature5,feature6]


