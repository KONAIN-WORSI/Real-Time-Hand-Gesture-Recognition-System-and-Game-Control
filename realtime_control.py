import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import time
import pyautogui
import screen_brightness_control as sbc
import warnings
warnings.filterwarnings('ignore')

# 1. Define Model Architecture (Must match 6 output classes)
class GestureANN(nn.Module):
    def __init__(self, input_dim=42, num_classes=6):
        super(GestureANN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# 2. Load Weights and Scaler
checkpoint = torch.load("gesture_ann_model.pth", weights_only=False)
num_classes = checkpoint['num_classes']
scaler = checkpoint['scaler']

model = GestureANN(input_dim=42, num_classes=num_classes)
model.load_state_dict(checkpoint['model_state'])
print(model.eval())

# 3. Master Gesture Action Mapping
GESTURE_MAP = {
    0: {"name": "Fist", "type": "key", "action": "volumemute"},
    1: {"name": "Open Palm", "type": "key", "action": "space"},
    2: {"name": "Thumbs Up", "type": "key", "action": "volumeup"},
    3: {"name": "Peace Sign", "type": "key", "action": "volumedown"},
    4: {"name": "Brightness Down Sign", "type": "brightness", "action": "-10"},
    5: {"name": "Brightness Up Sign", "type": "brightness", "action": "+10"}
}

# 4. MediaPipe Setup
mp_hands = mp.solutions.hands 
mp_draw = mp.solutions.drawing_utils 

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
last_action_time = time.time()
COOLDOWN_SEC = 0.8  # Prevent rapid-fire actions

print("\n🚀 All-in-One Gesture Controller ACTIVE!")
print("Controlling: Volume, Media Playback & Screen Brightness.")
print("Press 'q' to quit.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_status = "Scanning for gestures..."

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Wrist Normalization (Feature Engineering)
            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y
            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x - wrist_x, lm.y - wrist_y])

            # Preprocessing & Inference
            input_scaled = scaler.transform([landmarks])
            tensor_input = torch.FloatTensor(input_scaled)

            with torch.no_grad():
                outputs = model(tensor_input)
                probabilities = torch.softmax(outputs, dim=1)
                prob, pred_idx = torch.max(probabilities, 1)

                pred_id = pred_idx.item()
                confidence = prob.item() * 100

                # Check Confidence & Cooldown
                if confidence > 80.0 and pred_id in GESTURE_MAP:
                    gesture_info = GESTURE_MAP[pred_id]
                    current_status = f"{gesture_info['name']} ({confidence:.0f}%)"

                    curr_time = time.time()
                    if curr_time - last_action_time > COOLDOWN_SEC:
                        # Handle Keyboard Actions
                        if gesture_info['type'] == "key":
                            pyautogui.press(gesture_info['action'])
                            print(f"⚡ Triggered Keypress: {gesture_info['action']} via {gesture_info['name']}")

                        # Handle Brightness Actions
                        elif gesture_info['type'] == "brightness": 
                            if gesture_info['action'] == "+10":
                                sbc.set_brightness('+10')
                                print(f"☀️ Brightness Increased | Level: {sbc.get_brightness()[0]}%")
                            elif gesture_info['action'] == "-10":
                                sbc.set_brightness('-10')
                                print(f"🌙 Brightness Decreased | Level: {sbc.get_brightness()[0]}%")

                        last_action_time = curr_time

    # Display Status Overlay
    cv2.rectangle(frame, (0, 0), (550, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Gesture: {current_status}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Master Hand Gesture Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()