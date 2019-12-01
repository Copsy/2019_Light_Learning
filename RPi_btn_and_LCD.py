import RPi.GPIO as gp
import time
import RPi_I2C_driver

gp.setmode(gp.BOARD) # board number
btn_1 = 7
btn_2 = 11
btn_3 = 13
btn_4 = 15
gp.setup(btn_1, gp.IN)
gp.setup(btn_2, gp.IN)
gp.setup(btn_3, gp.IN)
gp.setup(btn_4, gp.IN)

lcd = RPi_I2C_driver.lcd(0x27)
#lcd.cursor()

def button(n, btn):
  print ('Button number:{0}'.format(n))
  lcd.print('Button number:{0}'.format(n))
  time.sleep(1)
  lcd.clear()
  gp.setup(btn, gp.OUT)
  gp.setup(btn, gp.IN)

try :
  while True:
    if gp.input(btn_1) == 1:
      button(1, btn_1)
      
    elif gp.input(btn_2) == 1:
      button(2, btn_2)
      
    elif gp.input(btn_3) == 1:
      button(3, btn_3)
      
    elif gp.input(btn_4) == 1:
      button(4, btn_4)

except KeyboardInterrupt:
  print("\nKeyboardInterrupt!\n")
      
finally:
  gp.cleanup()
  lcd.clear()
