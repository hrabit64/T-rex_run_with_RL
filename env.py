from selenium import webdriver
from PIL import Image
from io import BytesIO
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common import action_chains
from selenium.webdriver.support.ui import WebDriverWait
import numpy as np
import cv2
import time
import threading
import json
import sys
import decimal
import pyautogui
import io
import base64

class Env:

    def __init__(self):

        self.driver = webdriver.Chrome()
        self.driver.maximize_window()
        self.url = 'file:///C:/Users/hzser/Desktop/t-rex/env/index.html'

        self.url2 = 'https://ifn8fzu2qqqqztdjdzmhba-on.drv.tw/www.t-rex_env.com/'
        self.action_size = 3
        self.action_space = ['stay','jump','duck']
        self.state_shape = (66, 166, 4)
        self.score = 0
        self.game_thread = threading.Thread(self.run_thread())
        self.tot_score = 0
        self.pause_switch = False
        self.key = 0
        self.checker = 0

    def run(self):
        # self.game_thread.run()
        self.driver.get(self.url2)

    def run_thread(self):

        self.driver.get(self.url2)

    def exit(self):

        self.driver.close()

    def play(self):

        _ = self.driver.find_element(By.TAG_NAME, "html").send_keys(Keys.SPACE)

    def reset(self):
        try:
            self.exit()
            self.driver = webdriver.Chrome()
            self.driver.maximize_window()
        except:
            pass
        self.run()
        time.sleep(1)
        check = self.driver.execute_script('return resetDone()')

        if(check == None):
            raise Exception("Warring! function resetDone doesn't work!")

        self.tot_score = 0

    def stay(self):

        pyautogui.keyUp('w')
        pyautogui.keyUp('s')

    def step(self,action):

        if (action[0] == 1):

            pyautogui.keyUp('w')
            pyautogui.keyUp('s')
            print(">> stay")

        elif (action[1] == 1):

            pyautogui.keyDown('w')
            pyautogui.keyUp('s')
            print(">> jump")

        elif (action[2] == 1):

            pyautogui.keyUp('w')
            pyautogui.keyDown('s')
            print(">> duck")

    def setepi(self,epi):
        check = self.driver.execute_script('return setEpi(' + str(epi)+')')

        if (check == None):
            raise Exception("Warring! function setEpi doesn't work!")

    def pause(self):

        self.pause_switch = True
        wait = WebDriverWait(self.driver, 10)
        _ = wait.until(lambda x : self.pause_switch)

    def get_img(self):
        stack = []
        for i in range(4):
            img_base64 = self.driver.execute_script('return document.getElementsByClassName("runner-canvas")[0].toDataURL().substring(21);')
            decode = io.BytesIO(base64.b64decode(img_base64))
            img = Image.open(decode)

            gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
            ret, dst = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
            canny = cv2.Canny(dst,200,300)
            canny = cv2.resize(canny,(166,66))
            stack.append(canny)

        stack = np.stack((i for i in stack),axis=-1)
        stack = stack.astype('float32') / 255.0

        return stack

    def get_state(self,epi):
        self.score = 0
        state = self.get_img()
        done = self.driver.execute_script('return returnDone()')

        if done:

            self.score = -1
            print(f"epi {epi} >> end, score = {self.tot_score-1}")
        else:

            self.score = 0.01

        self.tot_score += decimal.Decimal(str(self.score))

        return state, self.score,done

