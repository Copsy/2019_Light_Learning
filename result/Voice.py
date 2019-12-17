# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:52:51 2019

@author: Alero
"""

import pyttsx3 as ps

def speak(word):
    
    engine = ps.init()
    
    # 말하는 속도
    engine.setProperty('rate', 140)
    rate = engine.getProperty('rate')
    # 소리 크기
    engine.setProperty('volume', 0.5) # 0~1 
    volume = engine.getProperty('volume')
    # 목소리
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id) # Female
    #engine.setProperty('voice', voices[0].id) # Make
    
    # 말하기
    engine.say(word) 
    engine.runAndWait() # 말 다할때까지 대기
    engine.stop()
