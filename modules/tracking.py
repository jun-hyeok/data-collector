import msvcrt
import threading

from pynput import keyboard, mouse
from .__init__ import STATE_PC

# Thread to detect mouse cursor movement
class MouseTracker(threading.Thread):
    '''
    Detect whether mouse cursor is moving or not
    '''
    def __init__(self):
        super().__init__()

    def run(self):
        # The event listener will be running in this block
        global STATE_PC
        while(True):
            with mouse.Events() as events:
                event = events.get(1.0) # Block at most one second
                if event is None:
                    print('You did not interact with the mouse within one second')
                    STATE_PC = 0
                else:
                    print('Received event Mouse {}'.format(event))
                    STATE_PC = 1

                if msvcrt.kbhit() and msvcrt.getch() == chr(27).encode():
                    break

# Thread to detect keyboard input
class KeboardTracker(threading.Thread):
    '''
    Detect whether keyboard is being pressed or not
    '''
    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        # The event listener will be running in this block
        global STATE_PC
        while(True):
            with keyboard.Events() as events:
                event = events.get(1.0) # Block at most one second
                if event is None:
                    print('You did not interact with the keyboard within one second')
                    STATE_PC = 0 
                elif str(event) == "Press(key=Key.esc)": # Press ESC key to EXIT
                    self.path.close()
                    STATE_PC = 3
                    print("escaped")
                    break
                else:
                    print(event)
                    print('Received event keyboard {}'.format(event))
                    STATE_PC  = 1

