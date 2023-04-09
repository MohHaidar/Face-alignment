import os
import cv2

def get_xy_from_value(img, value):
    x=[]
    y=[]
    row, col = img.shape
    for i in range(row):
        for j in range(col):
            if img[i,j] > value:
                x.append(j) # get x indices
                y.append(i) # get y indices
    return x,y

def center_image(image, new_shape):
    '''
    new_shape: (width, height)
    
    '''
    
    mask = image[:,:,2]
    x,y = get_xy_from_value(mask,10)
    centered_img = image[min(y):max(y),min(x):max(x),:]
    h,w,_ = centered_img.shape
    if w > h:
        scale = new_shape[0]/w
        offset_w = 0
        offset_h = int((new_shape[1] - int(scale*h))/2)
    else:
        scale = new_shape[1]/h
        offset_w = int((new_shape[0] - int(scale*w))/2)
        offset_h = 0
    centered_img = cv2.resize(centered_img,(int(w * scale), int(h * scale)))
    output_img = cv2.copyMakeBorder(centered_img, offset_h, offset_h, offset_w, offset_w, cv2.BORDER_CONSTANT, value=0)
    output_img = cv2.resize(output_img,new_shape)
    return output_img

if __name__=='__main__':   
    
    for file in os.listdir('remove bg/'):
        print(file)
        image = cv2.imread(os.path.join('remove bg',file), cv2.IMREAD_UNCHANGED)
        new_shape = (500,500)
        output_img = center_image(image, new_shape)
        cv2.imwrite(os.path.join('results',file), output_img)