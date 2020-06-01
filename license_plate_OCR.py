import cv2
import pytesseract
from imutils import contours
import pandas as pd

data = pd.read_csv(r"D:\IDFY ASSIGN\trainVal.csv")
datanew = data['image_path_new'].tolist()


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
savedata = []


for i in datanew:
    image = cv2.imread(i)
    height, width, _ = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    conts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    conts = conts[0] if len(conts) == 2 else conts[1]
    conts, _ = contours.sort_contours(conts, method="left-to-right")
    
    license_plate = ""
    for c in conts:
        area = cv2.contourArea(c)
        x,y,w,h = cv2.boundingRect(c)
        center_y = y + h/2
        if area > 3000 and (w > h) and center_y > height/2:
            ROI = image[y:y+h, x:x+w]
            data = pytesseract.image_to_string(ROI, lang='eng', config='--psm 6')
            license_plate += data
    
    print(i, license_plate)
    newout = i,license_plate
    savedata.append((newout))
    df = pd.DataFrame(savedata)
    df.to_csv('./Desktop/license_license_plates_OCR_report.csv', index=False, sep=',')

    