import pytest
import cv2 as cv

def key_input(param):
    if param==27:
        return -1
    elif param==32:
        return 1
    else:
        print(chr(param))
        return 0
    
def result_print(param):
    print("Result is "+str(param))
    return 0

def show_camera():
    cap=cv.VideoCapture(0)
    ref, img=cap.read()
    cap.release()
    #cv.imshow("Figure_1", img)
    cv.waitKey(0)
    return 0

def test():
    assert key_input(27)==0
    
    assert key_input(32)==0
    
    for i in range(26):
        assert key_input(65+i)==1
        
    assert result_print("c")==1
    
    assert show_camera()==1
