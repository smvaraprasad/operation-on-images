
import os
import math
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from os import listdir
import random
import numpy as np
class preprocess(object):
    def __init__(self, dataset_location="", batch_size=1,shuffle= False):
        self.location=dataset_location
        self.images=[]
        self.simages=[]
        self.batch_size=batch_size
        list=[]
        copycat=[]
        copycat=os.listdir(self.location).copy()
        self.sidx=list.copy()
        self.snames=copycat.copy()
        for j in os.listdir(self.location):
            num=''
            for k in j:
                if k=='_':
                    break
                num=num+k
            list.append(int(num))
        print(list)
        for i in (copycat):
            if len(self.simages) == batch_size:
                break
            self.simages.append(plt.imread(self.location+"\\"+i))
            #print(plt.imread(self.location+"\\"+i))
        for m in range(0,len(list)-1):
            #print(m)
            #print(list[m])
            for n in range(0,len(list)-m-1): #[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 20, 2, 3, 4, 5, 6, 7, 8, 9]
                if list[n]>list[n+1]:
                    copycat[n+1],copycat[n]=copycat[n],copycat[n+1]
                    list[n+1],list[n]=list[n],list[n+1]
                #    print(copycat)
                 #   print(list)
                #n=0
        #copycat[0]='11'
        print(os.listdir(self.location))
        print(list,copycat)
        self.idx = list
        self.length= len(list)
        self.names=copycat
        #self.seed = 
        self.shuffle = shuffle
        #print(list.sort())
        #print(list)
        for i in (copycat):
            if len(self.images) == batch_size:
                break
            self.images.append(plt.imread(self.location+"\\"+i))
            print(plt.imread(self.location+"\\"+i))
        #print('00000000000000000000000000000')
        ##print(self.images)
      # First argument specifies the location of the dataset,
    # batch_size indicates the number of input images on which operation has to be performed(It takes in integer as input),
    # shuffle can take boolean True/ False values. If True it picks up random indices of the input images every time and if false it will pick up indexes sequentially
    def rescale(self,s):
        
        numb=self.batch_size
        dict={}
        #print(plt.imshow(self.images[0]))
        if self.shuffle== False :
            for n in range(numb):
                sh=(self.images[n].shape)
                resh1,resh2=sh[0]*s,sh[1]*s
                rescaled=np.empty([resh1,resh2])

                for i in range(resh1-1):
                    for j in range(resh2-1):
                        xl,yl=math.floor(j*float(sh[0]-1)/(resh1-1)),math.floor(i*float(sh[1]-1)/(resh2-1))
                        xh,yh=math.ceil(j*float(sh[0]-1)/(resh1-1)),math.ceil(i*float(sh[1]-1)/(resh2-1))
                        one=self.images[n][yl,xl]
                        two=self.images[n][yl,xh]
                        three=self.images[n][yh,xl]
                        four=self.images[n][yh,xh]
                        rescaled[i][j]=one*(1-((j*(sh[0]-1)/(resh1-1))-xl))*(1-((i*(sh[1]-1)/(resh1-1))-yl)) \
                                        +two*(((j*(sh[0]-1)/(resh1-1))-xl))*(1-((i*(sh[1]-1)/(resh1-1))-yl)) \
                                            +three*(1-((j*(sh[0]-1)/(resh1-1))-xl))*(((i*(sh[1]-1)/(resh1-1))-yl)) \
                                                +four*(((j*(sh[0]-1)/(resh1-1))-xl))*(((i*(sh[1]-1)/(resh1-1))-yl))
                rescaled*=255/(np.max(rescaled))
                plt.imshow(rescaled,cmap='gray')
                dict[self.names[numb]]=rescaled
        else:
            
            for n in range(numb):
                sh=(self.simages[n].shape)
                resh1,resh2=sh[0]*s,sh[1]*s
                rescaled=np.empty([resh1,resh2])
                for i in range(resh1-1):
                    for j in range(resh2-1):
                        xl,yl=math.floor(j*float(sh[0]-1)/(resh1-1)),math.floor(i*float(sh[1]-1)/(resh2-1))
                        xh,yh=math.ceil(j*float(sh[0]-1)/(resh1-1)),math.ceil(i*float(sh[1]-1)/(resh2-1))
                        one=self.simages[n][yl,xl]
                        two=self.simages[n][yl,xh]
                        three=self.simages[n][yh,xl]
                        four=self.simages[n][yh,xh]
                        rescaled[i][j]=one*(1-((j*(sh[0]-1)/(resh1-1))-xl))*(1-((i*(sh[1]-1)/(resh1-1))-yl)) \
                                        +two*(((j*(sh[0]-1)/(resh1-1))-xl))*(1-((i*(sh[1]-1)/(resh1-1))-yl)) \
                                            +three*(1-((j*(sh[0]-1)/(resh1-1))-xl))*(((i*(sh[1]-1)/(resh1-1))-yl)) \
                                                +four*(((j*(sh[0]-1)/(resh1-1))-xl))*(((i*(sh[1]-1)/(resh1-1))-yl))
                rescaled*=255/(np.max(rescaled))
                plt.imshow(rescaled,cmap='gray')
                dict[self.snames[numb]]=rescaled
        return dict
    def resize(self,h,w):
        numb=self.batch_size
        dict={}
        l=0
        for i in range(numb):
            if self.shuffle==True:
                img=self.simages[i]
            else:
                img=self.images[i]
            r,c=(img.shape)
            rr,rc=h/r,w/c
            irr,irc=int(rr),int(rc)
            iterr=rr-irr
            resized=np.empty([h,w])            
            for j in range(h):
                for k in range(w):
                    resized[j,k]=img[int(j/rr),int(k/rc)]
                    #if irr>l:
                     #   resized[j,k]=img[int(j/irr),int(k/irc)]
                    #elif l==irr:
                     #   resized[j,k]=iterr*(img[int((j-1)/irr),int((k-1)/irc)])+(1-iterr)*(img[int(j/irr),int(k/irc)])
                      #  rr=rr-(1-iterr)
                       # l=0   '''
                        
            #print(self.names)
            #plt.imshow(resized,cmap='gray')
            if self.shuffle==True:
                dict[self.snames[i]]=resized
            else:
                dict[self.names[i]]=resized
        return dict
    def translate(self,tx,ty):
        numb=self.batch_size
        dict2={}
        for i in range(0,numb):
            if self.shuffle==True:
                img=self.simages[i]
            else:
                img = self.images[i]
            row,col=img.shape
            timg=np.empty([row,col])
            for j in range(0,row):
                for k in range(0,col):
                    if(j-ty>=0 and k+ty<=col-1):
                        timg[j-ty][k+ty]=img[j][k]
            #timg=timg[0:row,0:col]
           # print(timg.shape,img.shape)
            if self.shuffle==True:
                dict2[self.snames[i]]=timg
            else:
                dict2[self.names[i]]=timg
        return dict2
            
                    
                        
                    
                
    # Translates the batch of input images. Where tx is the translation in x direction and ty is the translation in the y direction.
    # This function has to return a dictionary of translated images where the key is the name of the image and the value is corresponding translated output in accordance to the random indexes picked up.

    def crop(self,id1,id2,id3,id4):
        numb=self.batch_size
        #print('vastunna vachestunna')
        dict={}
        for i in range(numb):
            if self.shuffle==True:
                img=self.simages[i]
            else:
                img = self.images[i]
            #print(img)
            #print(id1[0],id3[0],id1[1],id2[1])
            row,col = img.shape
            #print(row)
            cimg=img[row-id1[1]:row-id4[1],id1[0]:id2[0]]
            #print(cimg)
            if self.shuffle==True:
                dict[self.snames[i]]=cimg
            else:
                dict[self.names[i]]=cimg
        #print(plt.imshow(dict['1_ir.bmp']))
        return dict
                  
    # Resizes the batch of input images to the given dimensions. Where h is the number of rows and w is the number of columns
    # This function has to return a dictionary of resized images where the key is the name of the image and the value is corresponding resized output in accordance to the random indexes picked up.
    # The datatype of output has to be a numpy array
  
    def blur(self):
        numb=self.batch_size
        dict={}
        print('hedgtdgf')
        for i in range(numb):
            if self.shuffle==True:
                img=self.simages[i]
            else:
                img = self.images[i]
            row,col=img.shape
            bimg=img.copy()
            for j in range(1,row-1):
                for k in range(1,col-1):
                    temp=(img[j-1:j+2,k-1:k+2].flatten())
                    temp.sort()
                    bimg[j:j+2,k:k+2]=temp[4]                   
            bimg[:,-1]=img[:,-1]
            bimg[-1,:]=img[-1,:]
            print(bimg)
            if self.shuffle==True:
                dict[self.snames[i]]=bimg
            else:
                dict[self.names[i]]=bimg
        return dict
            #print(plt.imshow(img))
       # print(plt.imshow(dict['1_ir.bmp']))
        
    # Blurring the batch of input images in accordance to the filter specified.
    # This function has to return a dictionary of blurred images where the key is the name of the image and the value is corresponding blurred output in accordance to the random indexes picked up.
    def edge_detection(self):
        numb=self.batch_size
        dict={}
        gx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        gy=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        for i in range(numb):
            if self.shuffle==True:
                img=self.simages[i]
            else:
                img = self.images[i]
            row,col=img.shape
            
            eimg=np.empty([row+2,col+2])
            eimg[1:-1,1:-1]=img
            #print(eimg)
            feimg=np.empty([row,col])
            for j in range(row):
                for k in range(col):
                    GX=(gx.flatten())*(eimg[0+j:3+j,0+k:3+k].flatten())
                    GY=(gy.flatten())*(eimg[0+j:3+j,0+k:3+k].flatten())
                    GX=abs(np.sum(GX))
                    GY=abs(np.sum(GY))
                    val=(GX**2+GY**2)**0.5
                    feimg[j,k]=val
                    #print('main',img)
            mm=np.max(feimg)
            feimg[j,k]*=255/(mm)
    #    break
                #break
            if self.shuffle==True:
                dict[self.snames[i]]=feimg
            else:
                dict[self.names[i]]=feimg
            plt.imshow(feimg,cmap="gray")
        return dict
            #print(feimg)
            #print("input")
            #print(img.shape)
            
            #print('padded')
            #print(eimg.shape)
            #print("output")
            #print(feimg.shape)
           # plt.imshow(feimg,cmap='gray')
    # Extraction of edge map of a batch of input images.
    # This function has to return a dictionary of edge maps where the key is the name of the image and the value is corresponding edge map in accordance to the random indexes picked up.
    def rgb2gray(self):
        numb=self.batch_size
        dict={}
        for i in range(numb):
            # img=plt.imread(r'C:\Users\91807\Downloads\WIN_20201122_06_12_15_Pro.jpg')
            if self.shuffle==True:
                img=self.simages[i]
            else:
                img = self.images[i]
            #print(img.shape)
            #plt.imshow(img)
            if len(img.shape)!=3:
                if self.shuffle==True:
                    dict[self.simages[i]]=img
                else:
                    dict[self.names[i]]=img
            else:
                row,col,x=img.shape
                gimg=np.empty([row,col])
                print(img[:,:,0])
                gimg[:,:]=0.2989*img[:,:,0]+0.5870*img[:,:,1]+0.1140*img[:,:,2]
                if self.shuffle==True:
                    dict[self.snames[i]]=gimg
                else:
                    dict[self.names[i]]=gimg
                #print(plt.imshow(gimg,cmap='gray'))
        return dict
    # Converts a batch of rgb image to a gray image.
    # This function has to return a dictionary of gray images where the key is the name of the image and the value is corresponding gray image in accordance to the random indexes picked up.
#    def rotate(self,theta):
    def __getitem__(self):
        numb=self.batch_size
        for i in range(numb):
            if self.shuffle==True:
                dict[self.snames[i]]=self.simages[i]
            else:
                dict[self.names[i]]=self.images[i]
        return dict             
        # This function is to fetch items specified at the given location. Keeping in mind the batch_size, shuffle argument and the dataset location specified. It will pick up number of images = batch size and return the images.
# If shuffle = True it has to pick up random indices and if it is false it will pick up the images sequentially.
    # This function has to return a dictionary of input images where the key is the name of the image and the value is corresponding input image in accordance to the indexes picked up.
# Every other function will call the __getitem__ function to get the input images on which the operation has to be performed.

    def rotate(self,theta):
        numb=self.batch_size
        dict={}
        for i in range(numb):
            #img=plt.imread(r'C:\Users\91807\Downloads\WIN_20201122_06_12_15_Pro.jpg')
            if self.shuffle==True:
                img=self.simages[i]
            else:
                img = self.images[i]
            row,col=img.shape
            #edges
            nrow=round(abs(row*math.cos(np.radians(theta)))+abs(col*math.sin(np.radians(theta))))+1
            ncol=round(abs(col*math.cos(np.radians(theta)))+abs(row*math.sin(np.radians(theta))))+1
            rimg=np.zeros([int(nrow),int(ncol)])
            cen=[round(((row+1)/2) -1),round(((col+1)/2)-1)]
            ncen=[round(((nrow+1)/2) -1),round(((ncol+1)/2)-1)]
            for j in range(row):
                for k in range(col):
                    y=row-1-j-cen[0]
                    x=col-1-k-cen[1]
                    nx=math.floor(x*(math.cos(np.radians(theta)))+y*(math.sin(np.radians(theta))))
                    ny=math.floor(-x*(math.sin(np.radians(theta)))+y*(math.cos(np.radians(theta)))) 
                    #if a<row and b<col:
                        #print(rimg[int(a),int(b)])
                        #print(img[j,k])
                    if 0<=int(ncen[1]-nx)<ncol and 0<=int(ncen[0]-ny)<nrow:
                        rimg[int(ncen[0]-ny),int(ncen[1]-nx)]=img[j,k]
            if self.shuffle==True:
                dict[self.simages[i]]=rimg
            else:
                dict[self.names[i]]=rimg
        return dict



        
            #print('naadi')
            #print(rimg)
            #print(plt.imshow(rimg,cmap='gray'))
                    # Rotates the batch of input images to the given dimensions. Where theta is the angle of rotation.
        # This function has to return a dictionary of gray images where the key is the name of the image and the value is corresponding gray image in accordance to the random indexes picked up.
        
    #tates the batch of input images to the given dimensions. Where theta is the angle of rotation.
    # This function has to return a dictionary of gray images where the key is the name of the image and the value is corresponding gray image in accordance to the random indexes picked up.
#inputarg = preprocess(r"C:\Users\91807\Downloads\ir_test", 1 ,False)
#translated_output=inputarg.translate(50,50)
#cropped_output=inputarg.crop((50,300),(250,300),(250,40),(50,40))
#rescaled_output = inputarg.rescale(2)
#resized_output = inputarg.resize(1000,1000)
#blurrd_output=inputarg.blur()
#edge_detected=inputarg.edge_detection()
#gray=inputarg.rgb2gray()
#rotated_output=inputarg.rotate(180)