import cv2
import numpy as np
import time
from picamera2 import Picamera2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import mediapipe as mp
import RPi.GPIO as GPIO

# -------------------------------
# 1. Configuration
# -------------------------------
TFLITE_MODEL_PATH       = "final_model_dynamic_quantized.tflite"
CAM_RESOLUTION         = (980, 1020)
MODEL_IMG_SIZE         = (224, 224)
MIN_DETECTION_CONFIDENCE = 0.4
DEBOUNCE_INTERVAL      = 1.3   # seconds the same gesture must persist

CLASS_NAMES = [
    "all_fingers_down",
    "all_fingers_up",
    "peace_sign",
    "thumb_down",
    "thumb_index_down",
    "thumb_index_middle_down",
    "thumb_index_middle_ring_down",
    "thumb_index_middle_ring_up",
    "thumb_index_middle_up",
    "thumb_index_up",
    "thumb_pinky_up",
    "thumb_up"
]

# -------------------------------
# 2. GPIO / LED Setup
# -------------------------------
LED_PINS = [17, 27, 22, 23, 24]  # BCM

GESTURE_ACTION = {
    "all_fingers_down":            ("all_off",),
    "all_fingers_up":              ("all_on",),
    "thumb_up":                    ("on",  0),
    "thumb_index_up":              ("on",  1),
    "thumb_index_middle_up":       ("on",  2),
    "thumb_index_middle_ring_up":  ("on",  3),
    "thumb_pinky_up":              ("on",  4),
    "thumb_down":                  ("off", 0),
    "thumb_index_down":            ("off", 1),
    "thumb_index_middle_down":     ("off", 2),
    "thumb_index_middle_ring_down":("off", 3),
    "peace_sign":                  ("off", 4),
}

GPIO.setmode(GPIO.BCM)
for p in LED_PINS:
    GPIO.setup(p, GPIO.OUT, initial=GPIO.LOW)
led_state = {i: False for i in range(len(LED_PINS))}

def apply_gesture(gesture):
    act = GESTURE_ACTION.get(gesture)
    if not act: return
    cmd = act[0]
    if cmd == "all_off":
        for i,p in enumerate(LED_PINS):
            if led_state[i]:
                GPIO.output(p, GPIO.LOW); led_state[i]=False
    elif cmd == "all_on":
        for i,p in enumerate(LED_PINS):
            if not led_state[i]:
                GPIO.output(p, GPIO.HIGH); led_state[i]=True
    else:
        idx = act[1]
        want = (cmd=="on")
        if led_state[idx] != want:
            GPIO.output(LED_PINS[idx], GPIO.HIGH if want else GPIO.LOW)
            led_state[idx] = want

# -------------------------------
# 3. MediaPipe Hands
# -------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_DETECTION_CONFIDENCE
)

# -------------------------------
# 4. TFLite Interpreter
# -------------------------------
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
inp_det  = interpreter.get_input_details()[0]
out_det  = interpreter.get_output_details()[0]
assert inp_det["dtype"] == np.float32

# -------------------------------
# 5. Picamera2 Setup
# -------------------------------
picam2 = Picamera2()
cfg = picam2.create_preview_configuration(main={"size": CAM_RESOLUTION, "format": "RGB888"})
picam2.configure(cfg)
picam2.start()
time.sleep(2)  # warm up

# -------------------------------
# 6. Helpers
# -------------------------------
def detect_hand(frame_rgb):
    return bool(hands.process(frame_rgb).multi_hand_landmarks)

def otsu_pipeline(frame_bgr):
    # Apply grayscale convert → contrast stretching → median filter → Otsu → morphology
    
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    mn, mx = gray.min(), gray.max()
    if mx > mn:
        stretched = ((gray - mn) / (mx - mn) * 255).astype(np.uint8)
    else:
        stretched = gray.copy()
        
    blurred = cv2.medianBlur(stretched, 5)
    
    _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    closed = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, kernel, iterations=4)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=8)
    return opened

def prepare_input(mask):
    # Resize → gray→ RGB → preprocess_input → expand dims
    
    resized = cv2.resize(mask, MODEL_IMG_SIZE)
    rgb_mask = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    pre = preprocess_input(rgb_mask.astype(np.float32))
    return np.expand_dims(pre, axis=0).astype(np.float32)

def run_inference(inp):
    interpreter.set_tensor(inp_det["index"], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(out_det["index"])[0]
    idx = int(np.argmax(out))
    return idx, float(out[idx])

def overlay_label(frame_rgb, txt):
    cv2.putText(frame_rgb, txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0),2)

# -------------------------------
# 7. Main Loop with Debounce
# -------------------------------
stable_pred       = None
stable_start_time = 0.0
last_label        = ""

try:
    while True:
        frame_rgb = picam2.capture_array()
        if frame_rgb is None: 
            continue

        # check for hand presence
        has_hand = detect_hand(frame_rgb)

        if has_hand:
            frame_bgr  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            mask = otsu_pipeline(frame_bgr)
            inp  = prepare_input(mask)
            pred_idx, pred_confidence = run_inference(inp)
            gesture   = CLASS_NAMES[pred_idx]
            last_label= f"{gesture}: {pred_confidence:.2f}"
            cv2.imshow("Otsu Mask", mask)

            now = time.time()
            if gesture == stable_pred:
                # same, check if persisted
                if now - stable_start_time >= DEBOUNCE_INTERVAL:
                    apply_gesture(gesture)
            else:
                # new hand gesture, reset debounce
                stable_pred       = gesture
                stable_start_time = now

        else:
            # no hand, clear mask display, and all settings reset
            blank = np.zeros((MODEL_IMG_SIZE[1], MODEL_IMG_SIZE[0]), np.uint8)
            cv2.imshow("Otsu Mask", blank)
            stable_pred       = None
            stable_start_time = 0.0
            last_label        = ""

        if last_label:
            overlay_label(frame_rgb, last_label)
        cv2.imshow("Gesture Demo", frame_rgb)

        if cv2.waitKey(1)&0xFF == ord('q'):
            break

finally:
    picam2.stop()
    hands.close()
    GPIO.cleanup()
    cv2.destroyAllWindows()
