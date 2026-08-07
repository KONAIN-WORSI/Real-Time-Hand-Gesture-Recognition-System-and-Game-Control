import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import pyautogui
import time

# 1. Define Model Architecture (Must match Step 2 exactly)
class GestureANN(nn.Module):
    def __init__(self, input_dim=42, num_classes=4):
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

# 2. Load Saved Checkpoint & Scaler
checkpoint = torch.load("gesture_ann_model.pth", weights_only=False)
num_classes = checkpoint['num_classes']
scaler = checkpoint['scaler']

model = GestureANN(input_dim=42, num_classes=num_classes)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# Map gesture IDs to display names & OS shortcuts
GESTURE_MAP = {
    0: {"name": "Fist (Mute)", "action": "volumemute"},
    1: {"name": "Open Palm (Play/Pause)", "action": "space"},
    2: {"name": "Thumbs Up (Vol Up)", "action": "volumeup"},
    3: {"name": "Peace Sign (Vol Down)", "action": "volumedown"}
}

# 3. Setup MediaPipe
mp_hands = mp.solutions.hands 
mp_draw = mp.solutions.drawing_utils 

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# 4. Webcam Loop
cap = cv2.VideoCapture(0)

last_action_time = time.time()
COOLDOWN_SEC = 1.0  # Time delay between OS key triggers

print("\n🚀 System Controller Active! Perform gestures in front of the camera.")
print("Press 'q' in the window to exit.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_gesture = "Scanning..."
    confidence = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Feature Engineering: Wrist Normalization
            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y
            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x - wrist_x, lm.y - wrist_y])

            # Model Inference
            input_scaled = scaler.transform([landmarks])
            tensor_input = torch.FloatTensor(input_scaled)

            with torch.no_grad():
                outputs = model(tensor_input)
                probabilities = torch.softmax(outputs, dim=1)
                prob, pred_idx = torch.max(probabilities, 1)

                pred_id = pred_idx.item()
                confidence = prob.item() * 100

                # Set threshold: Only react if model confidence > 80%
                if confidence > 80.0 and pred_id in GESTURE_MAP:
                    gesture_info = GESTURE_MAP[pred_id]
                    current_gesture = f"{gesture_info['name']} ({confidence:.0f}%)"

                    # Execute OS command with Cooldown
                    curr_time = time.time()
                    if curr_time - last_action_time > COOLDOWN_SEC:
                        pyautogui.press(gesture_info['action'])
                        print(f"⚡ Triggered Action: {gesture_info['action']} via {gesture_info['name']}")
                        last_action_time = curr_time

    # Render Visual Feedback Frame
    cv2.rectangle(frame, (0, 0), (450, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Gesture: {current_gesture}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture System Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()