#
# This is the main file for handling keyboard inputs
#

import time, random
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

# delay to hold the keypresses for
# :random.uniform(0.1, 0.2) for hold
# :random.uniform(0.05, 0.1) for tap
delay = ['hold','tap']
# available inputs by type
direction = [0, DPAD_UP, DPAD_DOWN, DPAD_LEFT, DPAD_RIGHT]
attack = [0, TRIANGLE, CIRCLE, CROSS, SQUARE]
rage = [R1]

def GetActions(amount):
    actions = []
    actions.append([])
    action = 0
    for i in range(0,amount):
        temp = random.randint(0,1)
        if temp == 0:
            # select something random from the direction arroy
            action = direction[random.randint(0,len(direction)-1)]
        else:
            # select something random from the attack array
            action = attack[random.randint(0,len(direction)-1)]
        # Only add the input if it is not 0. 0 Is the same as nothing.
        if action != 0:
            actions.append(action)
    # Get the delay time for pressing these keys
    delayVal = delay[random.randint(0,1)]
    if delayVal == 'hold':
        # can't use i as the index because I am only adding non 0 inputs
        actions[0].append(random.uniform(0.1, 0.2))
    else:
        # can't use i as the index because I am only adding non 0 inputs
        actions[0].append(random.uniform(0.05, 0.1))
    return actions
