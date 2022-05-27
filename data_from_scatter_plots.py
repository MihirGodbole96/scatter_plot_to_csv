import os
import numpy as np
import pandas as pd
import cv2
from google.colab.patches import cv2_imshow
import glob as glob
import pytesseract
from absl import app,flags
from absl.flags import FLAGS

flags.DEFINE_string('input',None, 'path to input image')
flags.DEFINE_string('output',None, 'path to output csv')
flags.DEFINE_boolean('origin', False, 'origin provided or not')

def horizontal_vertical_rem(image):
  # Remove horizontal and vertical line i.e the grid
  ret3,thresh = cv2.threshold(image,230,255,cv2.THRESH_BINARY_INV)
  horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,1))
  detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
  kernel = np.ones((5,5), np.uint8)  
  vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,10))
  detected_lines_V = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
  kernel = np.ones((5,5), np.uint8)  

  d = detected_lines
  flag = 0
  for i in reversed(range(0,len(d))):
    if sum(d[i]) != 0:
      if flag == 0:
        y = i
        flag = 1
      elif int(sum(d[i])/255)<0.8*d.shape[1]:
        d[i][d[i]==255] = 0
        
  flag = 0  
  x_len = int(sum(d[y])/255)

  dv = cv2.rotate(detected_lines_V, cv2.ROTATE_90_COUNTERCLOCKWISE)
  x_ratio = detected_lines_V.shape[1]/dv.shape[0]
  for i in reversed(range(0,len(dv))):
    if sum(dv[i]) != 0:
      if flag == 0:       
        x = int(i*x_ratio)
        flag = 1
      elif int(sum(dv[i])/255)<0.6*dv.shape[1]:
        dv[i][dv[i]==255] = 0

  y_len = int(sum(dv[x])/255)
  x  = detected_lines_V.shape[1] - x
  first_y = y - y_len
  end_x = x + x_len
  detected_lines = d
  detected_lines_V = cv2.rotate(dv, cv2.ROTATE_90_CLOCKWISE)
  detected_lines = cv2.dilate(detected_lines,kernel,iterations=1)
  detected_lines_V = cv2.dilate(detected_lines_V,kernel,iterations=1)

  new_img  = cv2.inpaint(cv2.inpaint(image,detected_lines_V,3,cv2.INPAINT_TELEA) ,detected_lines,3,cv2.INPAINT_TELEA)
 
  return new_img, y, first_y , x, end_x, x_len, y_len

def get_scale(img, x, y):
  #Read 
  data_x = pytesseract.image_to_string(img[y:img.shape[0]], lang='eng', config='--psm 6')
  data_y = pytesseract.image_to_string(cv2.flip(cv2.rotate(img.transpose()[0:x], cv2.ROTATE_90_CLOCKWISE),1), lang='eng', config='--psm 6')

  x_arr = []
  num = 0
  dig = 0
  for i in reversed(range(0,len(data_x)-2)):
    if data_x[i] == " ":
      x_arr.append(num)
      num = 0
      dig = 0
    elif data_x[i].isnumeric():
      num += int(data_x[i])*(10**dig)
      dig += 1
  x_arr.append(num)

  y_arr = []
  num = 0
  dig = 0
  for j in reversed(range(-1,len(data_y)-2)):
    if data_y[j] == "\n":
      y_arr.append(num)
      num = 0
      dig = 0
    elif  data_y[j].isnumeric():
      num += int(data_y[j])*(10**dig)
      dig += 1
  y_arr.append(num)
  scale_x = x_arr[0]
  scale_y = y_arr[-1]

  return scale_x, scale_y

def get_contours(image_new, x, y, x_len, y_len):
  image = image_new[y-y_len:y, x:x+x_len].copy()
  ret, thresh_edge = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  edge_image = cv2.GaussianBlur(thresh_edge, (3, 3), 1)
  edge_image = cv2.Canny(edge_image, 100, 200)
  contours, h = cv2.findContours(edge_image.copy(),cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
  contours = sorted(contours, key=cv2.contourArea)
  return contours

def generate_centroids_csv(path, scale_x, scale_y, contours, x_len, y_len):
  data = pd.DataFrame(columns=['X','Y'])
  conv_x = scale_x/(x_len)
  conv_y = scale_y/(y_len)

  for i in contours:
    M = cv2.moments(i)
    if M['m00'] != 0:
      cx = int(M['m10']/M['m00'])
      cy = int(M['m01']/M['m00'])
      data.loc[len(data)] = [cx*conv_x,(y_len-cy)*conv_y]

  data['Dist'] = (data['X']**2+data['Y']**2)**(1/2)
  data = data.sort_values(by = 'Dist')
  data = data.reset_index()
  data = data.drop(columns = ['index','Dist'])
  del_index = []
  for ind in range(1,len(data.index)):  
    if ((data['X'][ind] - data['X'][ind-1])**2+(data['Y'][ind] - data['Y'][ind-1])**2)**(1/2)<0.5:
      del_index.append(ind-1)
  data = data.drop(del_index)
  data = data.reset_index()
  data = data.drop(columns = 'index')
  data = data.round()
  data.to_csv(path)
  print("CSV generated!")

def main(_argv):
  if FLAGS.origin:
    origin_x = int(input('Enter x co-ordinate of origin'))
    origin_y = int(input('Enter y co-ordinate of origin '))
    img = cv2.imread(FLAGS.input,0)
    print(1)
    image_new, y, first_y, x, end_x, x_len, y_len = horizontal_vertical_rem(img)
    print(1)
    x = origin_x
    y = origin_y
    scale_x, scale_y = get_scale(img, x, y)
    print(1)
    contours = get_contours(image_new, x, y, x_len, y_len)
    print(1)
    generate_centroids_csv(FLAGS.output, scale_x, scale_y, contours, x_len, y_len)
  else:
    img = cv2.imread(FLAGS.input,0)
    print(1)
    image_new, y, first_y, x, end_x, x_len, y_len = horizontal_vertical_rem(img)
    print(1)
    scale_x, scale_y = get_scale(img, x, y)
    print(1)
    contours = get_contours(image_new, x, y, x_len, y_len)
    print(1)
    generate_centroids_csv(FLAGS.output, scale_x, scale_y, contours, x_len, y_len)
    
if __name__ == '__main__':
  try:
    app.run(main)
  except SystemExit:
    pass
