import cv2      #access webcam ,draw shapes and text        handles mouse events
import mediapipe as mp      #for hand detection
import pygame
import time
import numpy as np
import pyautogui
pyautogui.FAILSAFE = False

frame_w = 640
frame_h = 480
scaled_points = {}

# step 1 ---------------------------------------------------------------------------------------------
MODE = "COUNT"      #default

def draw_menu(frame):
    h, w, c = frame.shape       #height     width       channels
    cv2.rectangle(frame,(0,0),(w,60), (50,50,50),-1)      #image, start_point, end_point, color, thickness
    cv2.putText(frame,"DRAW",(50,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)    # 1 size, white color, 2 thickness
    cv2.putText(frame,"FILTER",(250,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(frame,"COUNT",(480,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

cap = cv2.VideoCapture(0)       #opens the webcam -- 0: use laptop's camera

def mouse_click(event,x,y,flags,param):     #function calls automatically when any mouse event happens
    global MODE, filter_state, apply_processed  

    if event == cv2.EVENT_LBUTTONDOWN:      #reacts to left clicks and ignore any other event like right click, mouse scroll, mouse move 

        # print(x,y)        #prints the position of the mouse click -- for debugging
        if 0 < y < 60:      #vertical position (0 - top of screen   60 - bottom of menu)
            if 0 < x < 200:
                MODE = "DRAW"
            elif 200 < x < 400:
                MODE = "FILTER"
            elif 400 < x < 640:
                MODE = "COUNT"
        
        # for FILTER MODE ---------------------------------------
        if MODE == "FILTER" and filter_state == "TEMPLATE":
            if x > frame_w-120 and y > frame_h-50:
                filter_state = "APPLY"

        elif MODE == "FILTER" and filter_state == "APPLY":
            if x > frame_w-120 and y > frame_h-50:
                filter_state = "TEMPLATE"
                apply_processed = False
        # -------------------------------------------------------

cv2.namedWindow("Gesture App")      #creates window named gesture app
cv2.setMouseCallback("Gesture App", mouse_click)       #detects mouse click in the window (left click)

# ---------------------------------------------------------------------------------------------


# for part 2 and 3 -------------------------------------------------------------------------------

mp_draw = mp.solutions.drawing_utils    #draws the skeleton

mp_hands = mp.solutions.hands       #hand detection module
hands = mp_hands.Hands(max_num_hands = 2)   #detect upto 2 hands in the frame

pygame.mixer.init()
prev_sound = pygame.mixer.Sound("prev.wav")
next_sound = pygame.mixer.Sound("next.wav")
pause_sound = pygame.mixer.Sound("pause.wav")
start_sound = pygame.mixer.Sound("starts.wav")
played = False

last_gesture = None
gesture = ""
gesture_start_time = 0



# --------------------------------------------------------------------------------------------------------



# part 4----------------------------------------

canvas = None

eraser_size = 15

draw_color = (255, 255, 255)    #active brush color, whatever color is selected it's stored here 

COLORS = [
    (255, 255, 255),  # white
    (0, 0, 255),      # red (BGR)
    (0, 255, 0),      # green
    (255, 0, 0),      # blue
    (0, 255, 255),    # yellow
    (255, 255, 0),    # cyan
    (255, 0, 255),    # magenta
]

STRIP_X = 40        #sets width of the strip (anything with cx < strip_x means fingertip is over strip), also sets teh width for each swatch

SWATCH_H = 50       #Height of each color swatch in pixels. 7 colors × 50px = 350px total strip height.

STRIP_START_Y = 90      #Where the strip starts vertically — below the menu (60px) and MODE text (80px). Without this offset, the first swatch would be hidden behind the menu bar.

color_select_start = 0      #Timestamp of when you started hovering over a swatch. (change color in 1 sec)

last_color_swatch = -1      #track which swatch you were last hovering over 


def draw_color_strip(frame, draw_color):
    for i, color in enumerate(COLORS):      #enumerate gives index and the color at the same time ex.(if i = 0, color = (0,0,0), i = 1 color = (0,0,1) and so on...)
        
        # top edge:
        y1 = STRIP_START_Y + i * SWATCH_H   #For i=0: y1=90. For i=1: y1=140. For i=2: y1=190. Each swatch starts exactly where the previous one ended.

        # bottom edge:
        y2 = y1 + SWATCH_H      #y1 = 90   → top of swatch 1    y2 = 140  → bottom of swatch 1  (90 + 50)

        # draw the filled color swatch:
        cv2.rectangle(frame, (0, y1), (STRIP_X, y2), color, -1)     #(0, y1) is top-left corner, (STRIP_X, y2) is bottom-right corner. -1 means filled.

        # if this swatch's color matches the currently selected draw_color, give it a thick border (3px) to show it's selected. Otherwise thin border (1px).
        border = 3 if color == draw_color else 1
        
        # Draws the border on top of the filled swatch. Same coordinates, gray color (200,200,200), thickness is either 1 or 3 depending on whether it's selected.
        cv2.rectangle(frame, (0, y1), (STRIP_X, y2), (200,200,200), border)


        # So two rectangles are drawn per swatch — first the filled color, then the border on top of it.

# -----------------------------------------

# filter state ------------------------------------------------------------------
filter_state = "CAPTURE"
filter_template = None
filter_canvas = None
face_points = {}
apply_processed = False
strokes = []



mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

FACE_OUTLINE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
LEFT_EYE  = [33, 160, 158, 133, 153, 144, 33]
RIGHT_EYE = [362, 385, 387, 263, 373, 380, 362]
NOSE      = [168, 197, 195, 5, 4]
MOUTH     = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
# -----------------------------------------------------------------------------

def handle_color_selection(cx, cy, fingers):
    global draw_color, last_color_swatch, color_select_start
    
    if fingers == [0, 1, 0, 0, 0]:
        # print(f"checking color strip: cx={cx}, cy={cy}, STRIP_X={STRIP_X}")
        if cx < STRIP_X and STRIP_START_Y < cy < STRIP_START_Y + len(COLORS) * SWATCH_H:
            # print("over strip!")
            swatch_idx = (cy - STRIP_START_Y) // SWATCH_H
            if 0 <= swatch_idx < len(COLORS):
                if swatch_idx != last_color_swatch:
                    last_color_swatch = swatch_idx
                    color_select_start = time.time()
                if time.time() - color_select_start > 1:
                    draw_color = COLORS[swatch_idx]
            return True  # was over strip
    return False  # not over strip

def handle_drawing(target_canvas, restore_source, cx, cy, x4, y4, x12, y12, fingers):
    global eraser_size, draw_color
    
    if fingers == [1, 1, 0, 0, 0]:
        if restore_source is not None:
            mask = np.zeros(target_canvas.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (cx, cy), eraser_size, 255, -1)
            target_canvas[mask == 255] = restore_source[mask == 255]
        else:
            cv2.circle(target_canvas, (cx, cy), eraser_size, (30, 30, 30), -1)

    elif fingers == [1, 1, 1, 0, 0]:
        spread = int(((x4 - x12)**2 + (y4 - y12)**2) ** 0.5)
        eraser_size = max(15, min(80, int(15 + ((spread - 30) / 120) * 65)))

    elif fingers == [0, 1, 0, 0, 0]:
        if not handle_color_selection(cx, cy, fingers):
            cv2.circle(target_canvas, (cx, cy), 3, draw_color, -1)

    elif fingers == [0, 1, 1, 0, 0]:
        cv2.circle(target_canvas, (cx, cy), 10, draw_color, -1)

while True:

    
    pointer_pos = None      #step 4
    fingers = []
    cx, cy = 0, 0
    

    # step 1 ---------------------------------------------------------------------------------------------
    # in while loop (run every frame)
    ret, frame = cap.read()     #captures frames from the webcam    
    # ret - Did the camera capture successfully? (T/F)   frame - The actual image captured from the camera
    
    frame = cv2.flip(frame,1)   #so it looks like a mirror image 

    # part step 4----------------
    if MODE == "DRAW" and canvas is None:
        canvas = np.zeros_like(frame)   #create an empty array containing 0's (black) of the same size as frame 
        canvas[:] = (30, 30, 30)  # dark gray background -- overwriting the 0 (black)    
    # ---------------------------
    
    draw_menu(frame)

    # cv2.putText(image, text, position, font, size, color, thickness)  10 pixels from left 100 pixels down from top
    cv2.putText(frame,f"MODE: {MODE}",(10,80),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)

    #----------------------------------------------------------------------------------------------------

    #  STEP 2 ------------------------------------------------------------------------------------------
    # hand logic in while loop ---------
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      #open cv uses bgr mediapipe expects rgb

    results = hands.process(rgb_frame)      #sends frame to mediapipe model if hand is detected it returns 21 landmark points


    # hand loop
    if results.multi_hand_landmarks:        #any hands detected??
        for hand_landmarks, hand_label in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label.classification[0].label

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)    #frame - the img to draw on     handlandmark- 21 points     hand connection - which points to connect

            fingers = []
            label =  hand_label.classification[0].label

            # tip:
            # Thumb  → 4    Index  → 8    Middle → 12   Ring   → 16     Pinky  → 20
            # Each finger also has a middle joint (PIP):
            # Thumb  → 3    Index  → 6    Middle → 10   Ring   → 14     Pinky  → 18
            
            # we're comapring tip vs pip to see if the finger is open or close 
            
            # thumb moves sideways not vertically so we compare x values
            # right and left seperately cause of difference in direction of thumb in both hands
            if label == "Right":
                if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
                    fingers.append(1)
                else:
                    fingers.append(0)

            else:  # Left hand
                if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # other fingers we compare y values
            tips = [8,12,16,20]

            for tip in tips:
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip- 2].y:        # tip - 2 to calculate pip
                    fingers.append(1)
                else:
                    fingers.append(0)
            
            # step 2-------------------------------------------------------------------------------------
            if MODE == "COUNT":
                count = fingers.count(1)

                cv2.putText(frame, f"{label}: {count}",(10, 150 if label == "Left" else 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

            # step 3----------------------------------------------------------------------------------
            if MODE == "COUNT":
                if fingers == [0,1,0,0,0]:          #index
                    gesture = "NEXT"
                if fingers == [1,0,0,0,0]:          #thumb
                    gesture = "PREV"
                if fingers == [0,0,0,0,0]:          #fist
                    gesture = "PAUSE" 
                if fingers == [1,1,1,1,1]:          #all fives
                    gesture = "START"


                # if the gesture just changed reset the timer 
                if gesture != last_gesture:
                    last_gesture = gesture
                    gesture_start_time = time.time()
                    played = False

                # play the sound only if the tester hold it for atleast 1 sec 
                
                if gesture == "NEXT" and not played:
                    if time.time() - gesture_start_time > 1:
                        next_sound.play()
                        pyautogui.press('right')
                        played = True

                if gesture == "PREV" and not played:
                    if time.time() - gesture_start_time > 1:
                        prev_sound.play()
                        pyautogui.press('left')
                        played = True

                if gesture == "START" and not played:
                    if time.time() - gesture_start_time > 1:
                        start_sound.play()
                        pyautogui.press('f5')
                        played = True

                if gesture == "PAUSE" and not played:
                    if time.time() - gesture_start_time > 1:
                        pause_sound.play()
                        pyautogui.press('esc')
                        played = True
            # ---------------------------------------------------------------------------------

                

            # step 4 -------------------------------------------------------------------------------------
            
            if MODE == "DRAW":
                h, w, _ = frame.shape
                cx = int(hand_landmarks.landmark[8].x * w)
                cy = int(hand_landmarks.landmark[8].y * h)
                x4 = int(hand_landmarks.landmark[4].x * w)
                y4 = int(hand_landmarks.landmark[4].y * h)
                x12 = int(hand_landmarks.landmark[12].x * w)
                y12 = int(hand_landmarks.landmark[12].y * h)

                handle_drawing(canvas, None, cx, cy, x4, y4, x12, y12, fingers)

                if fingers == [0,1,1,0,1]:
                    canvas = np.zeros_like(frame)
                    canvas[:] = (30, 30, 30)

                pointer_pos = (cx, cy)



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

    # ------- --------------------------------------------------------------
    if MODE == "FILTER":
        if filter_state == "CAPTURE":
            cv2.putText(frame, "Show your face...", (150, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            
            results_face = face_mesh.process(rgb_frame)
            
            if results_face.multi_face_landmarks:
                face_lm = results_face.multi_face_landmarks[0]
                h, w, _ = frame.shape
                
                for idx, lm in enumerate(face_lm.landmark):
                    face_points[idx] = (int(lm.x * w), int(lm.y * h))
                
                filter_state = "TEMPLATE"
                print(f"face_points captured: {len(face_points)} landmarks")
                print(f"sample — landmark 33: {face_points[33]}")


                # build template once using captured face_points
                template = np.zeros_like(frame)
                template[:] = (30, 30, 30)  # gray background

                # DRAWING FACE MESH *******
                # Step 1 — find bounding box of all landmarks:
                xs = [face_points[i][0] for i in range(468)]
                ys = [face_points[i][1] for i in range(468)]

                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)

                # Step 2 — calculate scale factor:
                face_w = max_x - min_x
                face_h = max_y - min_y

                scale = min((w * 0.8) / face_w, (h * 0.8) / face_h)

                # Step 3 — apply scale and center each point:
                cx_face = (min_x + max_x) / 2  # center of face
                cy_face = (min_y + max_y) / 2  # center of face

                scaled_points = {}
                for idx, (px, py) in face_points.items():
                    new_x = int(w/2 + (px - cx_face) * scale)
                    new_y = int(h/2 + (py - cy_face) * scale + 40)  #shoft down 40px (it was overlapping with the nav bar)
                    scaled_points[idx] = (new_x, new_y)

                # step 4 - draw full mesh 
                for connection in mp.solutions.face_mesh.FACEMESH_TESSELATION:
                    a, b = connection
                    cv2.line(template, scaled_points[a], scaled_points[b], (80, 120, 100), 1)

                # draw key features on top with brighter color
                for group in [FACE_OUTLINE, LEFT_EYE, RIGHT_EYE, NOSE, MOUTH]:
                    for i in range(len(group) - 1):
                        a = group[i]
                        b = group[i + 1]
                        cv2.line(template, scaled_points[a], scaled_points[b], (180,180,180), 1)
                # ***********


                # apply button 
                cv2.rectangle(template, (w-120, h-50), (w-10, h-10), (70,70,70), -1)
                cv2.putText(template, "APPLY", (w-110, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                
                filter_template = template.copy()   # fixed background — never changes
                filter_canvas = template.copy()     # user draws on this
        
        elif filter_state == "APPLY":
            if not apply_processed:
                # run processing once
                strokes = []
                diff = cv2.absdiff(filter_canvas, filter_template)
                diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                ys, xs = np.where(diff_gray > 0)
                
                for y, x in zip(ys, xs):
                    # find nearest landmark
                    best_idx = 0
                    best_dist = ((scaled_points[0][0] - x)**2 + (scaled_points[0][1] - y)**2) ** 0.5
                    for idx, (px, py) in scaled_points.items():
                        dist = ((px - x)**2 + (py - y)**2) ** 0.5
                        if dist < best_dist:
                            best_idx = idx
                            best_dist = dist
                    
                    lm_x, lm_y = scaled_points[best_idx]
                    offset_x = x - lm_x
                    offset_y = y - lm_y
                    color = tuple(int(c) for c in filter_canvas[y, x])
                    if color != (30, 30, 30):  # skip background pixels
                        strokes.append((best_idx, offset_x, offset_y, color))
                
                apply_processed = True

            # get fresh face landmarks from live webcam
            results_face = face_mesh.process(rgb_frame)

            if results_face.multi_face_landmarks:
                
                face_lm = results_face.multi_face_landmarks[0]
                h, w, _ = frame.shape
                
                # get current landmark positions
                live_points = {}
                for idx, lm in enumerate(face_lm.landmark):
                    live_points[idx] = (int(lm.x * w), int(lm.y * h))
                
                # redraw each stored stroke at live position + offset
                for (landmark_idx, offset_x, offset_y, color) in strokes:
                    lx, ly = live_points[landmark_idx]
                    draw_x = lx + offset_x
                    draw_y = ly + offset_y
                    cv2.circle(frame, (draw_x, draw_y), 2, color, -1)

            # draw EDIT button
            cv2.rectangle(frame, (frame_w-120, frame_h-50), (frame_w-10, frame_h-10), (70,70,70), -1)
            cv2.putText(frame, "EDIT", (frame_w-100, frame_h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            draw_menu(frame)
            cv2.putText(frame, "MODE: FILTER", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            
        elif filter_state == "TEMPLATE":      
            frame = filter_canvas.copy()      # show canvas instead of webcam
            draw_menu(frame)
            cv2.putText(frame, "MODE: FILTER", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)  
            draw_color_strip(frame, draw_color) #color strip
            if pointer_pos:
                cv2.circle(frame, pointer_pos, 10, draw_color, -1)
                
        elif fingers == [0, 1, 0, 0, 0]:
            if not handle_color_selection(cx, cy, fingers):
                cv2.circle(filter_canvas, (cx, cy), 3, draw_color, -1)
    
    # OUTSIDE hand loop — canvas compositing
    if MODE == "DRAW" and canvas is not None:
        # Replaces the webcam frame with the canvas. So instead of seeing your camera feed, you see the drawing. canvas.copy() is used instead of canvas directly so that anything drawn on frame afterwards (menu, pointer, text) doesn't permanently mark the canvas itself.
        frame = canvas.copy()  

        # redraw menu canvas and below this redraw mode text and color strip too [canvas.copy wiped it out]
        draw_menu(frame)         
        cv2.putText(frame, f"MODE: {MODE}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        draw_color_strip(frame, draw_color) 
        # draw the pointer too only if the hand is visible 
        if pointer_pos:
            cv2.circle(frame, pointer_pos, 10, draw_color, -1)      #color of the pointer = draw_color


    # OUTSIDE hand loop — eraser preview + color selection feedback
    if MODE == "DRAW"  or (MODE == "FILTER" and filter_state == "TEMPLATE") and pointer_pos:
        if fingers == [1, 1, 0, 0, 0]:
            # hollow gray circle showing eraser size
            cv2.circle(frame, pointer_pos, eraser_size, (100,100,100), 1)

            # # shows current eraser size as text so user knows the radius
            cv2.putText(frame, f"eraser: {eraser_size}px", (10,110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)

        # index finger hovering over the color strip
        elif cx < STRIP_X and STRIP_START_Y < cy < STRIP_START_Y + len(COLORS) * SWATCH_H and fingers == [0, 1, 0, 0, 0]:

            # calculate how long the finger has been held over this swatch
            held = time.time() - color_select_start

            # show hold progress — caps at 3.0s, color changes when it reaches 3
            cv2.putText(frame, f"selecting: {min(held, 1):.1f}s", (10,110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
 
    # -----------------------------------------------------------------------------------------------
    

    


    # part 1 but it needs to be at bottom of the while loop to follow the pipeline of the project---------
    cv2.imshow("Gesture App",frame)     # displays a image in a window called gesture app   

    if cv2.waitKey(1) & 0xFF == 27:     # 27 is esc key (it exits closes the program)
        break
    # ---------------------------------------------------------------------------------------------------



# outside while loop 
cap.release()       #turn off camera 
cv2.destroyAllWindows()     #close window








# it was lagging when we were using playsound 



# limitation 
# Finger detection assumes the palm faces the camera. Detection accuracy may decrease when the back of the hand is shown.
# finger detection assumes your hand is straight up tilting your hand decreses the accuracy
# two consecutive peace won't generate sound twice 


# future scope
# add animation
# index finger as mouse
# save canvas
# add canvas 




# step 1 menu bar - draw filter count mode 
#                 - setup web cam 
#                 - detect mouse pointer's coordinates and change mode
#                 - exits when the user presses esc key
# 
# step 2 hand detection - detect hand 21 points and connection lines 

# step 3 gestures       - index finger → next slide
#                       - thumb → previous slide  
#                       - all five → presentation starts
#                       - fist → presentation stops 
#                       - all with 1 sec hold + audio confirmation

# step 4 paint in air   - gray canvas if not selected anything 
#                       - index - thin line
#                       - index + middle - thick line
#                       - index + middle + pinky - clear canvas
#                       - thumb + index - eraser
#                       - thumb + index + middle - to resize eraser 
#                       -color strip hold 1 sec with index to change '
#                       - ctrl + s - save your drawing

# step 4 filter mode    - adds filter to your face
#                       - everything in the draw mode
#                       - edit filter save filter 