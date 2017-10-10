# This code is the combination of multiple blog posts.
# It uses ctypes to get the processID for PS4 Remote Play
# and then feed keyboard commands to it.
#
# The blogs can be found at:
# https://sjohannes.wordpress.com/2012/03/23/win32-python-getting-all-window-titles/
# http://pixomania.net/programming/python-getting-the-title-of-windows-getting-their-processes-and-their-commandlines-using-ctypes-and-win32/
# https://stackoverflow.com/questions/13564851/how-to-generate-keyboard-events-in-python
#

import ctypes
from ctypes import wintypes, c_int, byref
import time, random, keyboardCombos

user32 = ctypes.WinDLL('user32', use_last_error=True)

INPUT_MOUSE    = 0
INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP       = 0x0002
KEYEVENTF_UNICODE     = 0x0004
KEYEVENTF_SCANCODE    = 0x0008

MAPVK_VK_TO_VSC = 0

# msdn.microsoft.com/en-us/library/dd375731
VK_TAB      = 0x09 # Tab key
VK_LMENU    = 0x12 # Left Alt key
VK_LCONTROL = 0xA2 # Left Ctrl key
VK_RIGHT    = 0x27 # Right arrow key
LK_LWIN     = 0x5B # Left Windows key
VK_6        = 0x36 # 6 key
DPAD_UP     = 0x57 # W key
DPAD_DOWN   = 0x53 # S key
DPAD_LEFT   = 0x41 # A key
DPAD_RIGHT  = 0x44 # D key
CROSS       = 0x45 # E key
SQUARE      = 0x52 # R key
CIRCLE      = 0x54 # T key
TRIANGLE    = 0x59 # Y key
R1          = 0x51 # Q key

wintypes.ULONG_PTR = wintypes.WPARAM

class MOUSEINPUT(ctypes.Structure):
    _fields_ = (("dx",          wintypes.LONG),
                ("dy",          wintypes.LONG),
                ("mouseData",   wintypes.DWORD),
                ("dwFlags",     wintypes.DWORD),
                ("time",        wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR))

class KEYBDINPUT(ctypes.Structure):
    _fields_ = (("wVk",         wintypes.WORD),
                ("wScan",       wintypes.WORD),
                ("dwFlags",     wintypes.DWORD),
                ("time",        wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR))

    def __init__(self, *args, **kwds):
        super(KEYBDINPUT, self).__init__(*args, **kwds)
        # some programs use the scan code even if KEYEVENTF_SCANCODE
        # isn't set in dwFflags, so attempt to map the correct code.
        if not self.dwFlags & KEYEVENTF_UNICODE:
            self.wScan = user32.MapVirtualKeyExW(self.wVk,
                                                 MAPVK_VK_TO_VSC, 0)
class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (("uMsg",    wintypes.DWORD),
                ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD))

class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = (("ki", KEYBDINPUT),
                    ("mi", MOUSEINPUT),
                    ("hi", HARDWAREINPUT))
    _anonymous_ = ("_input",)
    _fields_ = (("type",   wintypes.DWORD),
                ("_input", _INPUT))

LPINPUT = ctypes.POINTER(INPUT)

def _check_count(result, func, args):
    if result == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    return args

user32.SendInput.errcheck = _check_count
user32.SendInput.argtypes = (wintypes.UINT, # nInputs
                             LPINPUT,       # pInputs
                             ctypes.c_int)  # cbSize

# Functions
global_PS4RemotePlayPID = 0
global_PS4RemotePlayHWND = 0

def GetRemotePlayPID():
    # register winapi functions
    EnumWindows = ctypes.windll.user32.EnumWindows
    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
    GetWindowText = ctypes.windll.user32.GetWindowTextW
    GetWindowTextLength = ctypes.windll.user32.GetWindowTextLengthW
    IsWindowVisible = ctypes.windll.user32.IsWindowVisible
    GetWindowThreadProcessId = ctypes.windll.user32.GetWindowThreadProcessId

    def foreach_window(hwnd, lParam):
        # window must be visible
        if IsWindowVisible(hwnd):
            length = GetWindowTextLength(hwnd)
            buff = ctypes.create_unicode_buffer(length + 1)
            GetWindowText(hwnd, buff, length + 1)
            try:
                windowtitle = buff.value
                if "PS4" in windowtitle:
                    # get the processid from the hwnd
                    # declaring this as global means refer to the global version
                    global global_PS4RemotePlayPID
                    global global_PS4RemotePlayHWND
                    processID = c_int()
                    threadID = GetWindowThreadProcessId(hwnd, byref(processID))
                    # found the process ID
                    global_PS4RemotePlayPID = processID
                    global_PS4RemotePlayHWND = hwnd
                    return True
            except:
                print("Unexpected error:"+sys.exc_info()[0])
                pass;
        return True
    EnumWindows(EnumWindowsProc(foreach_window), 0)


def PressKey(hexKeyCode):
    x = INPUT(type=INPUT_KEYBOARD,
              ki=KEYBDINPUT(wVk=hexKeyCode))
    user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    x = INPUT(type=INPUT_KEYBOARD,
              ki=KEYBDINPUT(wVk=hexKeyCode,
                            dwFlags=KEYEVENTF_KEYUP))
    user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))

def FocusWindow(hwnd):
    ctypes.windll.user32.SetForegroundWindow(hwnd)
    ControlAltForREMAP()

def ControlAltForREMAP():
    time.sleep(0.5)
    PressKey(VK_LCONTROL)
    PressKey(VK_LMENU)
    time.sleep(0.01)
    ReleaseKey(VK_LCONTROL)
    ReleaseKey(VK_LMENU)
    time.sleep(0.01)

def MoveRight():
    PressKey(LK_LWIN)
    PressKey(VK_6)
    time.sleep(0.3)
    ReleaseKey(LK_LWIN)
    ReleaseKey(VK_6)
    time.sleep(1)
    for i in range(0,10):
        PressKey(VK_RIGHT)
        time.sleep(0.1)
        ReleaseKey(VK_RIGHT)
def HoldDelay():
    time.sleep(random.uniform(0.1, 0.2))

def QuickPressDelay():
    time.sleep(random.uniform(0.05, 0.1))

def strafe():
    PressKey(DPAD_UP)
    HoldDelay()
    ReleaseKey(DPAD_UP)
    HoldDelay()
    PressKey(DPAD_UP)
    HoldDelay()
    ReleaseKey(DPAD_UP)
    HoldDelay()

def ExecuteActions(actions):
    # start from 1 because first index is the delay time
    for i in range(1,len(actions)-1):
        PressKey(actions[i])

    time.sleep(actions[0][0])
    # start from 1 because first index is the delay time
    for i in range(1,len(actions)-1):
        ReleaseKey(actions[i])
    time.sleep(random.uniform(0.01, 0.5))

def ChooseRandomCommands():
    x = 100
    while x > 0:
        amount = random.randint(0,7)
        actions = keyboardCombos.GetActions(amount=amount)
        ExecuteActions(actions=actions)
        x = x-1

def AltTab():
    """Press Alt+Tab and hold Alt key for 2 seconds
    in order to see the overlay.
    """
    PressKey(VK_MENU)   # Alt
    PressKey(VK_TAB)    # Tab
    ReleaseKey(VK_TAB)  # Tab~
    time.sleep(2)
    ReleaseKey(VK_MENU) # Alt~

if __name__ == "__main__":
    #AltTab()
    GetRemotePlayPID()
    FocusWindow(global_PS4RemotePlayHWND)
    time.sleep(1)
    ChooseRandomCommands()
    time.sleep(1)
    ControlAltForREMAP()

    #MoveRight()
#hexKeyCode is t
