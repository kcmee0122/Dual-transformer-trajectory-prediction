import cv2
import numpy as np

# 이미지 불러오기
OpenData_dir = './raw'
Data_num = 18  

IMG = cv2.imread(f'./{OpenData_dir}/{Data_num:02d}_background.png')
IMG_with_lines = IMG.copy()

# 이미지 크기
height, width, _ = IMG.shape

# Data_num에 따른 M2P 값 설정
if int(Data_num) >= 0 and int(Data_num) <= 6:
    M2P = 6.55  # 00~06
elif int(Data_num) >= 7 and int(Data_num) <= 17:
    M2P = 10.2  # 07~17
elif int(Data_num) >= 18 and int(Data_num) <= 29:
    M2P = 10.2  # 18~29
elif int(Data_num) >= 30 and int(Data_num) <= 32:
    M2P = 10.2  # 30~32

# 직선 정보: (a, b) (기울기 a, 절편 b)
line_params = [
    (0.2690, -36.8954), 
    (0.1590, -41.1995),  
    (-3.1755, 91.9147),
    (-4.4015 , 224.7228)        
]

# 각 직선을 이미지에 그리기
for a, b in line_params:
    # x 값 (이미지 좌표계 기준)
    x1, x2 = 0, width - 1
    
    # M2P를 반영하여 x와 y의 값을 변환
    # 실제 물리적 거리에서 픽셀로 변환 (x는 이미 픽셀 단위로 가정)
    y1 = - (a * (x1/M2P) + b)  # 첫 번째 점의 y 값
    y2 = - (a * (x2/M2P) + b) # 두 번째 점의 y 값

    # M2P 반영하여 y값을 실제 픽셀로 변환
    y1 = int(y1 * M2P)
    y2 = int(y2 * M2P)

    # 이미지 경계 내에 있을 경우만 그리기
    pt1 = (x1, y1)
    pt2 = (x2, y2)
    cv2.line(IMG_with_lines, pt1, pt2, (0, 0, 255), 2)  # 빨간 선

# 결과 출력
cv2.imshow('Lines by a, b with M2P', IMG_with_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()