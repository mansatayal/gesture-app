import cv2          # access webcam, draw shapes and text, handle mouse events
import mediapipe as mp   # for hand and face detection
import pygame        # for sound playback
import time
import numpy as np
import pyautogui      # for sending keypresses to control presentations
pyautogui.FAILSAFE = False


# GLOBALS 
MODE = "COUNT"      # default mode: DRAW, FILTER, COUNT
frame_w = 640       # webcam frame width 
frame_h = 480       # webcam frame height 


# GLOBALS — DRAW MODE ============================================================

canvas = None              # persistent drawing surface for DRAW mode
eraser_size = 15           # current eraser radius in px (resizable)

draw_color = (255, 255, 255)   # active brush color (BGR)

COLORS = [
    (255, 255, 255),  # white
    (0, 0, 255),      # red
    (0, 255, 0),      # green
    (255, 0, 0),      # blue
    (0, 255, 255),    # yellow
    (255, 255, 0),    # cyan
    (255, 0, 255),    # magenta
]

STRIP_X = 40                 # right edge of color strip (left side of screen)
SWATCH_H = 50                 # height of each color swatch
STRIP_START_Y = 90             # top of strip, below menu + MODE text

color_select_start = 0          # timestamp when current swatch hover began
last_color_swatch = -1           # last swatch index being hovered (-1 = none)
COLOR_HOLD_TIME = 1              # seconds to hold before color changes

# ============================================================



# GLOBALS — FILTER MODE ============================================================

filter_state = "CAPTURE"     # CAPTURE -> TEMPLATE -> APPLY
filter_template = None       # fixed face skeleton background (never modified)
filter_canvas = None         # user's drawing on top of the skeleton
face_points = {}             # raw captured landmark positions (idx -> pixel pos)
scaled_points = {}            # face_points scaled + centered to fill canvas
strokes = []                 # processed drawing: (landmark_idx, offset_x, offset_y, color)
apply_processed = False       # whether APPLY click has been processed this cycle

FACE_OUTLINE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
LEFT_EYE  = [33, 160, 158, 133, 153, 144, 33]
RIGHT_EYE = [362, 385, 387, 263, 373, 380, 362]
NOSE      = [168, 197, 195, 5, 4]
MOUTH     = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
             375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
 
# ===========================================================


# GLOBALS — COUNT MODE ============================================================
played = False
last_gesture = None
gesture = ""
gesture_start_time = 0


pygame.mixer.init()
prev_sound = pygame.mixer.Sound("prev.wav")
next_sound = pygame.mixer.Sound("next.wav")
pause_sound = pygame.mixer.Sound("pause.wav")
start_sound = pygame.mixer.Sound("starts.wav")
# ============================================================


# UI FUNCTIONS ============================================================
def draw_menu(frame):
    h, w, c = frame.shape       #height     width     channels
    cv2.rectangle(frame, (0, 0), (w, 60), (50, 50, 50), -1)  #image, startpoint, endpoint, color,thickness
    cv2.putText(frame, "DRAW", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)   #1- size, white, 2- thickness
    cv2.putText(frame, "FILTER", (250, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "COUNT", (480, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


def draw_color_strip(frame, draw_color):
# NOTE: the color strip is drawn over the frame (not on canvas) so it's a UI overlay, never part of the drawing

    # loop through each color in COLORS list with its index         enumerate- gives both idx and value 
    for i, color in enumerate(COLORS):

        # each swatch is SWATCH_H pixels tall, stacked below STRIP_START_Y
        y1 = STRIP_START_Y + i * SWATCH_H   #top edge       90, 140, 190 etc
        y2 = y1 + SWATCH_H                  #bottom edge    always 50px below top 

        # draw filled color rect for every color
        cv2.rectangle(frame, (0, y1), (STRIP_X, y2), color, -1)

        # thick border for the color in use     else thin border
        border = 3 if color == draw_color else 1

        # draw gray border 
        cv2.rectangle(frame, (0, y1), (STRIP_X, y2), (200, 200, 200), border)



def handle_color_selection(cx, cy, fingers):
    # Checks if index finger is hovering over the color strip and handles the hold-to-select logic. Returns True if finger was over the strip (caller should skip normal drawing), False otherwise.
    global draw_color, last_color_swatch, color_select_start

    if fingers == [0, 1, 0, 0, 0]:
        # cx < STRIP_X — fingertip is to the left of 40px, meaning it's over the strip
        # STRIP_START_Y < cy < STRIP_START_Y + len(COLORS) * SWATCH_H — fingertip is between y=90 and y=440, meaning it's within the strip's vertical bounds, not above or below it
        if cx < STRIP_X and STRIP_START_Y < cy < STRIP_START_Y + len(COLORS) * SWATCH_H:
            swatch_idx = (cy - STRIP_START_Y) // SWATCH_H

            # which swatch you're hovering over  So cy=100 → (100-90)//50 = 0 (first swatch). cy=150 → (150-90)//50 = 1 (second swatch).
            if 0 <= swatch_idx < len(COLORS):           #is the index valid?
                if swatch_idx != last_color_swatch:     #hovering over different swatch?
                    last_color_swatch = swatch_idx
                    color_select_start = time.time()
                if time.time() - color_select_start > COLOR_HOLD_TIME:  #hold for 1sec to change the color
                    draw_color = COLORS[swatch_idx]
            return True
    return False


def handle_drawing(target_canvas, restore_source, cx, cy, x4, y4, x12, y12, fingers):
    # restore_source: None for DRAW mode (erase = gray fill) filter_template for FILTER mode (erase = restore skeleton).global eraser_size, draw_color

    if fingers == [1, 1, 0, 0, 0]:               #thumb + index — erase
        # FILTER mode: can't just paint gray, that would also erase the face skeleton underneath. Instead, restore the ORIGINAL pixels from restore_source (filter_template) inside the eraser circle, using a mask as a stencil:
        global eraser_size, draw_color
        if restore_source is not None:
            # 1. blank black grid, same h x w as canvas, single channel (no BGR needed)
            mask = np.zeros(target_canvas.shape[:2], dtype=np.uint8)

            # 2. draw a solid white (255) circle on the mask at the eraser position this marks "which pixels to restore"
            cv2.circle(mask, (cx, cy), eraser_size, 255, -1)

            # 3. restore wherever it's white
            target_canvas[mask == 255] = restore_source[mask == 255]
        else:
            # erase grey
            cv2.circle(target_canvas, (cx, cy), eraser_size, (30, 30, 30), -1)

    elif fingers == [1, 1, 1, 0, 0]:      # thumb + index + middle — resize eraser
        spread = int(((x4 - x12) ** 2 + (y4 - y12) ** 2) ** 0.5)    #distance between thumb and middle 
        eraser_size = max(15, min(80, int(15 + ((spread - 30) / 120) * 65)))    #b/w 15 to 80 px

    elif fingers == [0, 1, 0, 0, 0]:      # index only — color select or thin draw
        if not handle_color_selection(cx, cy, fingers):
            cv2.circle(target_canvas, (cx, cy), 3, draw_color, -1)

    elif fingers == [0, 1, 1, 0, 0]:      # peace — thick draw
        cv2.circle(target_canvas, (cx, cy), 10, draw_color, -1)


def make_face_template(face_points, w, h):
    # Builds the face skeleton template once, using captured face_points

    # create a blank grey array of the size as frame
    template = np.zeros((h, w, 3), dtype=np.uint8)
    template[:] = (30, 30, 30)

    # create a list of all x & y coordinates from the 468 landmark points
    xs = [face_points[i][0] for i in range(468)]
    ys = [face_points[i][1] for i in range(468)]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # actual width and height of the face
    face_w = max_x - min_x
    face_h = max_y - min_y

    # calculate how much to scale the face so it fills 80% of the canvas
    # use min() so the face fits both dimensions without distortion
    # 0.8 leaves a 10% margin on each side so the face isn't clipped
    scale = min((w * 0.8) / face_w, (h * 0.8) / face_h)

    # center of the face
    cx_face = (min_x + max_x) / 2
    cy_face = (min_y + max_y) / 2

    # rescale and recenter every landmark onto the template canvas
    # (px - cx_face) shifts each point relative to face center
    # * scale enlarges/shrinks it proportionally
    # w/2 and h/2 recenters it onto the middle of the canvas
    scaled = {}
    for idx, (px, py) in face_points.items():
        new_x = int(w / 2 + (px - cx_face) * scale)
        new_y = int(h / 2 + (py - cy_face) * scale + 40)  # +40 clears the menu bar
        scaled[idx] = (new_x, new_y)

    # draw full mesh tesselation (468 landmark connections) in dim teal
    # gives the detailed grid-like face structure
    for connection in mp.solutions.face_mesh.FACEMESH_TESSELATION:
        # FACEMESH_TESSELATION is a set of pairs — each pair defines two landmark indices that should be connected by a line. So connection is one such pair, like (33, 7) meaning "draw a line between landmark 33 and landmark 7.
        a, b = connection
        cv2.line(template, scaled[a], scaled[b], (100, 120, 80), 1)

    # overdraw key features (outline, eyes, nose, mouth) in brighter gray so they stand out clearly above the full mesh
    for group in [FACE_OUTLINE, LEFT_EYE, RIGHT_EYE, NOSE, MOUTH]:
        for i in range(len(group) - 1):
            a, b = group[i], group[i + 1]
            cv2.line(template, scaled[a], scaled[b], (180, 180, 180), 1)

    #template for drawing on, scaled_points for landmark-to-pixel mapping in APPLY state
    return template, scaled     

# ============================================================


# MOUSE CLICK HANDLER ============================================================
def mouse_click(event, x, y, flags, param):
    # event — type of mouse action (left click, right click, scroll, move etc)
    # x, y  — pixel coordinates of where the mouse event happened
    # flags, param — additional info passed by OpenCV (not used here)

    # global needed because we're reassigning these variables inside the function
    # without global, Python would treat them as new local variables that disappear
    # when the function ends, and the module-level values would never update
    global MODE, filter_state, apply_processed

    if event == cv2.EVENT_LBUTTONDOWN:  #react only on left click ignore the rest 
        if 0 < y < 60:  #navbar is in y = 0 to 60 if the click is in between switch mode
            if 0 < x < 200:
                MODE = "DRAW"
            elif 200 < x < 400:
                MODE = "FILTER"
            elif 400 < x < 640:
                MODE = "COUNT"

        # FILTER mode: APPLY/EDIT button (bottom-right corner)
        # clicking it moves from drawing state to applying state
        # apply
        if MODE == "FILTER" and filter_state == "TEMPLATE":
            if x > frame_w - 120 and y > frame_h - 50:
                filter_state = "APPLY"
        
        # edit
        elif MODE == "FILTER" and filter_state == "APPLY":
            if x > frame_w - 120 and y > frame_h - 50:
                filter_state = "TEMPLATE"
                apply_processed = False

# ============================================================


# SETUP ============================================================
cap = cv2.VideoCapture(0)                       #use the first available camera device (built-in webcam)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)      #capture in 640 X 480 instead of default
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)     #default might be higher resolution = more data = poor performance
cap.set(cv2.CAP_PROP_FPS, 30)                   #request 30 fps    why 30 fps- it's smooth enough frame rate most webcams , videocalls  typically runs at 30 fps  less processing power needed           

cv2.namedWindow("Gesture App")                  #create the actual window that'll display user's video
cv2.setMouseCallback("Gesture App", mouse_click)    #run the mouse function for the window 

mp_draw = mp.solutions.drawing_utils       #draws the skeleton
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,        #atmost 2 hands per frame 
    model_complexity=0,     #use the lighter & faster model (1 is more accurate but slower)
    min_detection_confidence=0.7,   #detect hand only if confidence is above 70%
    min_tracking_confidence=0.5     #track hand only if confidence is above 50%
)

mp_face_mesh = mp.solutions.face_mesh   
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ============================================================


# MAIN LOOP ============================================================
while True:     #run every frame 
    pointer_pos = None

    ret, frame = cap.read()     #ret - did the camera capture successfully(t/f)  frame- image captured 
    frame = cv2.flip(frame, 1)  #flip the mirrored image (0 - vertically, 1 - horizontally, -1 - both)

    if MODE == "DRAW" and canvas is None:
        canvas = np.zeros_like(frame)   #creates a blank array with exact same shape and size as frame (black)
        canvas[:] = (30, 30, 30)    #overwrite every pixel with dark grey 
        #this block runs only once when switched to draw mode cause of canvas is none check     after that the canvas contains the real array(the drawing pixels) so it never gets executed again to retain what's drawn


    draw_menu(frame)    #draw menu/ navbar on frame 
    cv2.putText(frame, f"MODE: {MODE}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)    

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      #open cv uses bgr mediapipe expects rgb
    results = hands.process(rgb_frame)      #sends frame to mediapipe model if hand is detected it returns 21 landmark points

    # ---------------- HAND LOOP ----------------
    if results.multi_hand_landmarks:    #any hands detected?
        for hand_landmarks, hand_label in zip(results.multi_hand_landmarks, results.multi_handedness):  # 21points, right/left
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers = []
            label = hand_label.classification[0].label

            # concept:
            # Index finger landmarks: 5, 6, 7, 8
            # 5 = MCP (knuckle, base of finger)
            # 6 = PIP (middle joint)
            # 7 = DIP (joint closer to tip)
            # 8 = TIP (fingertip)
            # tip:
            # Thumb  → 4    Index  → 8    Middle → 12   Ring   → 16     Pinky  → 20
            # Each finger also has a middle joint (PIP):
            # Thumb  → 3    Index  → 6    Middle → 10   Ring   → 14     Pinky  → 18
            
            # we're comapring tip vs pip to see if the finger is open or close 
            
            # thumb moves sideways not vertically so we compare x values
            # right and left seperately cause of difference in direction of thumb in both hands

            if label == "Right":
                fingers.append(1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0)
            else:
                fingers.append(1 if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x else 0)

            # other fingers: compare y (tip above pip = open)
            for tip in [8, 12, 16, 20]:
                fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y else 0)     # tip - 2 to calculate pip

            # ---------------- COUNT MODE ----------------
            if MODE == "COUNT":
                count = fingers.count(1)
                cv2.putText(frame, f"{label}: {count}", (10, 150 if label == "Left" else 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if fingers == [0, 1, 0, 0, 0]:          #index
                    gesture = "NEXT"
                elif fingers == [1, 0, 0, 0, 0]:        #thumb
                    gesture = "PREV"
                elif fingers == [0, 0, 0, 0, 0]:        #fist
                    gesture = "PAUSE"
                elif fingers == [1, 1, 1, 1, 1]:        #all fives
                    gesture = "START"

                if gesture != last_gesture:
                    last_gesture = gesture
                    gesture_start_time = time.time()
                    played = False      #makes sure the sound triggers once per gesture

                if gesture == "NEXT" and not played and time.time() - gesture_start_time > 1:
                    next_sound.play()
                    pyautogui.press('right')
                    played = True
                elif gesture == "PREV" and not played and time.time() - gesture_start_time > 1:
                    prev_sound.play()
                    pyautogui.press('left')
                    played = True
                elif gesture == "START" and not played and time.time() - gesture_start_time > 1:
                    start_sound.play()
                    pyautogui.press('f5')
                    played = True
                elif gesture == "PAUSE" and not played and time.time() - gesture_start_time > 1:
                    pause_sound.play()
                    pyautogui.press('esc')
                    played = True

            # ---------------- DRAW MODE ----------------
            if MODE == "DRAW":
                h, w, _ = frame.shape
                cx = int(hand_landmarks.landmark[8].x * w)      #index
                cy = int(hand_landmarks.landmark[8].y * h)
                x4 = int(hand_landmarks.landmark[4].x * w)      #thumb
                y4 = int(hand_landmarks.landmark[4].y * h)
                x12 = int(hand_landmarks.landmark[12].x * w)    #middle 
                y12 = int(hand_landmarks.landmark[12].y * h)

                handle_drawing(canvas, None, cx, cy, x4, y4, x12, y12, fingers)

                if fingers == [0, 1, 1, 0, 1]:          # clear canvas
                    canvas = np.zeros_like(frame)
                    canvas[:] = (30, 30, 30)

                pointer_pos = (cx, cy)

            # ---------------- FILTER MODE: TEMPLATE drawing ----------------
            if MODE == "FILTER" and filter_state == "TEMPLATE":
                h, w, _ = frame.shape
                cx = int(hand_landmarks.landmark[8].x * w)
                cy = int(hand_landmarks.landmark[8].y * h)
                x4 = int(hand_landmarks.landmark[4].x * w)
                y4 = int(hand_landmarks.landmark[4].y * h)
                x12 = int(hand_landmarks.landmark[12].x * w)
                y12 = int(hand_landmarks.landmark[12].y * h)

                handle_drawing(filter_canvas, filter_template, cx, cy, x4, y4, x12, y12, fingers)

                pointer_pos = (cx, cy)

    # ---------------- FILTER MODE(outside hand loop) ----------------
    if MODE == "FILTER":

        if filter_state == "CAPTURE":
            # Show "Show your face..." text every frame while in CAPTURE state
            # Simultaneously run face detection every frame
            # The moment a face IS detected, if results_face.multi_face_landmarks: becomes True — capture the landmarks, build the template, switch to TEMPLATE state
            # CAPTURE state never runs again, so the text disappears and the template canvas appears instead
            cv2.putText(frame, "Show your face...", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            results_face = face_mesh.process(rgb_frame)     #run face detection on current frame 

            if results_face.multi_face_landmarks:   #any face detected?
                # [0] because multi_face_landmarks is a list — max_num_faces=1 so only index 0 exists 
                face_lm = results_face.multi_face_landmarks[0]  
                h, w, _ = frame.shape

                # convert all 468 normalized landmark coordinates (0-1) to actual pixel positions
                # lm.x * w and lm.y * h scales them to the frame's pixel dimensions
                # stored as face_points dict: {landmark_index: (pixel_x, pixel_y)}
                # For example, a landmark at the center of the frame would have .x = 0.5, .y = 0.5 regardless of what resolution your webcam is.
                # To convert to actual pixel positions you multiply by the frame dimensions:
                # lm.x * w → 0.5 * 640 = 320 pixels from left
                # lm.y * h → 0.5 * 480 = 240 pixels from top
                for idx, lm in enumerate(face_lm.landmark):
                    face_points[idx] = (int(lm.x * w), int(lm.y * h))
                
                # build the face skeleton template using the captured landmark positions
                filter_template, scaled_points = make_face_template(face_points, w, h)

                # copy is used so the original template list remains intact so it can be restored when erasing and many more
                filter_canvas = filter_template.copy() 
                     
                filter_state = "TEMPLATE"   # face captured and template built switch to drawing state

        elif filter_state == "TEMPLATE":
            # canvas composting means building the final visible frame 
            frame = filter_canvas.copy()    #replace webcam feed with canvas
            draw_menu(frame)                #draw menu
            cv2.putText(frame, "MODE: FILTER", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            draw_color_strip(frame, draw_color)     #navbar
            cv2.rectangle(frame, (frame_w-120, frame_h-50), (frame_w-10, frame_h-10), (70,70,70), -1)   
            cv2.putText(frame, "APPLY", (frame_w-110, frame_h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)     # apply button 

            # pointer_pos is none by default it changes only when a hand is detected so if a hand isn't detected it skips the cv2.circle statement
            # cv2.circle needs coordinates (320,240) to actually draw at that coordinate in case of no coordinates (none value) cv2.circle won't know where to draw hence crash hence we used a if statement to avoid the null statement completely 
            if pointer_pos:
                cv2.circle(frame, pointer_pos, 10, draw_color, -1)

        elif filter_state == "APPLY":
            # find all drawn pixels , calculate nearest landmark
            if not apply_processed:
                strokes = []
                # gives a grid with 0 and non-0 values (0 - no change, any non zero value - something changed (drawing stroke in this case))
                diff = cv2.absdiff(filter_canvas, filter_template)
                diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                ys, xs = np.where(diff_gray > 0)

                # iterate through every point and calculate the distance where we need to draw(on face)
                for y, x in zip(ys, xs):
                    best_idx = 0
                    best_dist = ((scaled_points[0][0] - x) ** 2 + (scaled_points[0][1] - y) ** 2) ** 0.5
                    for idx, (px, py) in scaled_points.items():
                        dist = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
                        if dist < best_dist:
                            best_idx = idx
                            best_dist = dist

                    lm_x, lm_y = scaled_points[best_idx]

                    # let say the lanmark is (100,50) and drawn pixel is at (108, 59) offset tells you the 8px to the right and 50 px down
                    offset_x = x - lm_x
                    offset_y = y - lm_y

                    # create a tuple stroing all the color in bgr codes present in filter_canvas foe every pixel
                    color = tuple(int(c) for c in filter_canvas[y, x])

                    # if not gray add the pixel point to the strokes list
                    if color != (30, 30, 30):
                        strokes.append((best_idx, offset_x, offset_y, color))

                apply_processed = True      #apply processed mark it so it doesn't run again and again

            results_face = face_mesh.process(rgb_frame)

            #draw strokes on face if any face is detected
            if results_face.multi_face_landmarks:  
                face_lm = results_face.multi_face_landmarks[0]
                h, w, _ = frame.shape

                # create a list of all the live points 468 landmark points
                live_points = {}
                for idx, lm in enumerate(face_lm.landmark):
                    live_points[idx] = (int(lm.x * w), int(lm.y * h))

                # draw at the live points following thr index offsets and color (on face)
                for (landmark_idx, offset_x, offset_y, color) in strokes:
                    lx, ly = live_points[landmark_idx]
                    cv2.circle(frame, (lx + offset_x, ly + offset_y), 2, color, -1)

            # edit button
            cv2.rectangle(frame, (frame_w - 120, frame_h - 50), (frame_w - 10, frame_h - 10), (70, 70, 70), -1)
            cv2.putText(frame, "EDIT", (frame_w - 100, frame_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # canvas composting again as the feed changed from canvas to webcam
            draw_menu(frame)
            cv2.putText(frame, "MODE: FILTER", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # ---------------- DRAW MODE: canvas compositing ----------------
    if MODE == "DRAW" and canvas is not None:
        frame = canvas.copy()   #webcam to canvas
        draw_menu(frame)    #redraw menu
        cv2.putText(frame, f"MODE: {MODE}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        draw_color_strip(frame, draw_color)
        if pointer_pos:     #redraw pointer too only if the hand is visible
            cv2.circle(frame, pointer_pos, 10, draw_color, -1)

    # ---------------- shared: eraser preview + color selection feedback ----------------
    in_drawing_state = (MODE == "DRAW") or (MODE == "FILTER" and filter_state == "TEMPLATE")
    if in_drawing_state and pointer_pos:
        if fingers == [1, 1, 0, 0, 0]:
            # hollow grey circle showing the size of the eraser 
            cv2.circle(frame, pointer_pos, eraser_size, (100, 100, 100), 1)
            # shows current eraser's size radius
            cv2.putText(frame, f"eraser: {eraser_size}px", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # if index finger is over the strip
        elif (cx < STRIP_X and STRIP_START_Y < cy < STRIP_START_Y + len(COLORS) * SWATCH_H and fingers == [0, 1, 0, 0, 0]):
            # calculate time for each swatch
            held = time.time() - color_select_start
            # if 1sec hits change color
            cv2.putText(frame, f"selecting: {min(held, COLOR_HOLD_TIME):.1f}s", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # part 1 but it needs to be at the bottom to follow the pipeline
    cv2.imshow("Gesture App", frame)  # displays a image in a window called gesture app   

    if cv2.waitKey(1) & 0xFF == 27:   # 27 is esc key (it exits closes the program)
        break

# outside while loop
cap.release()   #turn off camera
cv2.destroyAllWindows()     #close window


# ============================================================
# NOTES
# ============================================================
# limitations:
# - finger detection assumes palm faces camera; back-of-hand reduces accuracy
# - finger detection assumes hand is upright; tilting reduces accuracy
# - two consecutive identical COUNT gestures won't re-trigger sound

# future scope:
# - index finger as mouse
# - save canvas / filter drawing to file
# - undo/redo

# step 1: menu bar (DRAW / FILTER / COUNT), webcam setup, mouse click mode switch, ESC to exit

# step 2: hand detection — 21 landmarks + connection lines

# step 3: COUNT mode — index = next slide
#                      thumb = prev slide
#                      all-five = start (F5)
#                      fist = stop (esc)
#         all gestures require 1 sec hold + audio confirmation, uses pyautogui

# step 4: DRAW mode — gray canvas
#                     index = thin line
#                     peace = thick line,
#                     index middle pinky = clear canvas
#                     thumb index = eraser
#                     thumb index middle = resize eraser
#                     color strip (1 sec hold to select)

# step 5: FILTER mode — CAPTURE (detect face, build skeleton template) ->
#         TEMPLATE (user draws using same gestures as DRAW) ->
#         APPLY (strokes mapped to nearest face landmark + offset, redrawn live on face)
#         EDIT button returns to TEMPLATE for further changes