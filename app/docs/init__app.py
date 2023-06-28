import keyboard
import time

end_time = time.time() + 7200

while time.time() < end_time:
    keyboard.press('ctrl')
    keyboard.release('ctrl')
    time.sleep(10)
