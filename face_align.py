import os
import cv2
import numpy as np
from mtcnn import MTCNN
detector = MTCNN()

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
    x_center = np.argmax(np.convolve(centered_img[:,:,3].sum(axis=0), np.ones(int(w/5)), 'same'))
    left_side = x_center 
    right_side = w - x_center
    padding_left = max(left_side,right_side) - left_side
    padding_right = max(left_side,right_side) - right_side
    centered_img = cv2.copyMakeBorder(centered_img, 0, 0, padding_left, padding_right, cv2.BORDER_CONSTANT, value=0)
    
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

class FaceAligner:
    #modify the box size with changing the values for desiredLeftEye
    def __init__(self, desiredLeftEye=(0.30, 0.30),
        desiredFaceWidth=224, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight
        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth
    def align(self, image, left_eye, right_eye):
        
        # compute the angle between the eye centroids
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        #compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
        
        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist
        
        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = (int((left_eye[0] + right_eye[0]) // 2),
                      int((left_eye[1] + right_eye[1]) // 2))
        # grab the rotation matrix for rotating and scaling the face
        #print(eyesCenter,angle,scale)
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])
        
        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_CUBIC)
        # return the aligned face
        return output
      
    
if __name__=='__main__':   
    
    for file in os.listdir('remove bg/'):
        print(file)
        image = cv2.imread(os.path.join('remove bg',file), cv2.IMREAD_UNCHANGED)
        new_shape = (500,500)
        centered_img = center_image(image, new_shape)
        fa = FaceAligner(desiredFaceWidth=200, desiredLeftEye=(0.42,0.46))
        
        faces = detector.detect_faces(centered_img[:,:,:3])
        if len(faces)>0:
            output_img = fa.align(centered_img, faces[0]['keypoints']['left_eye'], faces[0]['keypoints']['right_eye'])
        else:
            print(f'face not detected in image: {file}')
            # output_img = centered_img
            continue
        cv2.imwrite(os.path.join('results_3',file), output_img)