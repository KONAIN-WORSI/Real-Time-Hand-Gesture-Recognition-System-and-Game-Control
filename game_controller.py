import cv2
import mediapipe as mp
import pydirectinput

# Disable PyDirectInput failsafe to prevent game input interruptions
pydirectinput.FAILSAFE = False

# 1. Initialize MediaPipe Hands
mp_hands = mp.solutions.hands 

hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.4
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
current_keys = set()
window_name = "Hand-Tracking Game Controller"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 940, 720)

def handle_game_movement(new_keys):
    """
    Holds the selected direction keys and releases keys that are no longer active.
    """
    global current_keys

    new_keys = set(new_keys or ())
    if new_keys == current_keys:
        return

    for key in current_keys - new_keys:
        pydirectinput.keyUp(key)

    for key in new_keys - current_keys:
        pydirectinput.keyDown(key)

    current_keys = new_keys

print("\n🚀 Direct Hand-Tracking Game Controller ACTIVE (No ML Model Needed!)")
print("Instructions:")
print("1. Open your game in the background and click on it to focus.")
print("2. Move your palm through the virtual 3x3 grid to move your character.")
print("3. Press 'q' in the camera window to safely exit.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror flip frame for intuitive control
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False

    results = hands.process(rgb_frame)
    action_text = "NEUTRAL (IDLE)"
    target_keys = set()
    grid_size = 3

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Use only the palm center. Finger shape and individual gestures
            # do not affect movement direction.
            palm_x = sum(hand_landmarks.landmark[index].x for index in (0, 5, 9, 13, 17)) / 5
            palm_y = sum(hand_landmarks.landmark[index].y for index in (0, 5, 9, 13, 17)) / 5
            grid_column = min(grid_size - 1, max(0, int(palm_x * grid_size)))
            grid_row = min(grid_size - 1, max(0, int(palm_y * grid_size)))

            # The center cell is neutral. Row and column offsets provide
            # eight-way movement across the 3x3 control grid.
            center_cell = grid_size // 2
            if grid_column < center_cell:
                target_keys.add('a')
            elif grid_column > center_cell:
                target_keys.add('d')

            if grid_row < center_cell:
                target_keys.add('w')
            elif grid_row > center_cell:
                target_keys.add('s')

            if target_keys:
                direction_names = {
                    frozenset({'a'}): "LEFT",
                    frozenset({'d'}): "RIGHT",
                    frozenset({'w'}): "UP",
                    frozenset({'s'}): "DOWN",
                    frozenset({'a', 'w'}): "UP-LEFT",
                    frozenset({'d', 'w'}): "UP-RIGHT",
                    frozenset({'a', 's'}): "DOWN-LEFT",
                    frozenset({'d', 's'}): "DOWN-RIGHT",
                }
                action_text = f"{direction_names[frozenset(target_keys)]}"
            else:
                action_text = "CENTER (IDLE)"

    # Send key events directly to OS background
    handle_game_movement(target_keys)

    # Show the 3x3 control grid and current action for calibration.
    for grid_line in range(1, grid_size):
        x = int(w * grid_line / grid_size)
        y = int(h * grid_line / grid_size)
        cv2.line(frame, (x, 0), (x, h), (90, 90, 90), 1)
        cv2.line(frame, (0, y), (w, y), (90, 90, 90), 1)

    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.putText(frame, f"Character Action: {action_text}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up keyboard states on exit
handle_game_movement(None)
cap.release()
cv2.destroyAllWindows()