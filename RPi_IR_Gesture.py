import cv2
import mediapipe as mp
import math
import time
import RPi.GPIO as GPIO
import pigpio

IR = 17
LED_1 = 20
LED_2 = 21
LED_3 = 27

GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_1, GPIO.OUT)
GPIO.setup(LED_2, GPIO.OUT)
GPIO.setup(LED_3, GPIO.OUT)

GPIO.output(LED_1,GPIO.HIGH)
GPIO.output(LED_2,GPIO.HIGH)
GPIO.output(LED_3,GPIO.HIGH)
time.sleep(2)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pi = pigpio.pi()
if not pi.connected:
    print("Could not connect to pigpio daemon!")
    exit()
def reverse_bytes(code):
    reversed_bits = []
    for i in range(4):  
        byte = (code >> (i * 8)) & 0xFF
        reversed_byte = int(f"{byte:08b}"[::-1], 2)
        reversed_bits.append(reversed_byte)
    
    wire_code = 0
    for i in range(4):
        wire_code |= reversed_bits[i] << ((3 - i) * 8)  

    return wire_code


def carrier_wave(gpio, frequency, micros):
    """Generate a carrier wave burst"""
    period = int(1e6 / frequency)
    on_time = period // 2
    off_time = period - on_time
    pulses = []
    for _ in range(int(micros / period)):
        pulses.append(pigpio.pulse(1 << gpio, 0, on_time))
        pulses.append(pigpio.pulse(0, 1 << gpio, off_time))
    return pulses

def send_nec(code, pin=IR):
    """Send NEC code using pigpio"""
    frequency = 38000
    pi.set_mode(pin, pigpio.OUTPUT)
    pi.wave_clear()
    pi.wave_add_new()
    pulses = []

    # Leader: 9ms on, 4.5ms off
    pulses += carrier_wave(pin, frequency, 9000)
    pulses.append(pigpio.pulse(0, 0, 4500))

    # 32 bits of data, LSB first
    for i in range(32):
        pulses += carrier_wave(pin, frequency, 560)
        if (code >> i) & 1:
            pulses.append(pigpio.pulse(0, 0, 1690))
        else:
            pulses.append(pigpio.pulse(0, 0, 560))

    # Final burst
    pulses += carrier_wave(pin, frequency, 560)

    # Send the wave
    pi.wave_add_generic(pulses)
    wave_id = pi.wave_create()
    pi.wave_send_once(wave_id)

    while pi.wave_tx_busy():
        time.sleep(0.01)

    pi.wave_delete(wave_id)

def distance(point1, point2):
    return math.hypot(point1[0] - point2[0], point1[1] - point2[1])

def detect_gesture(landmarks):
    tips = [4, 8, 12, 16, 20] 
    coords = [(lm.x, lm.y) for lm in landmarks]

    fingers_up = []
    for i in range(1, 5):  
        fingers_up.append(coords[tips[i]][1] < coords[tips[i] - 2][1])  

    thumb_above = coords[4][1] < coords[3][1] < coords[2][1]
    thumb_extended = distance(coords[4], coords[0]) > 0.15
    thumb_up = thumb_above and thumb_extended

    if fingers_up == [True, False, False, False]:  
        original_code = 0xCB04F       
        nec_code = reverse_bytes(original_code)
        send_nec(nec_code)
        return "Arrow_Left"
    if fingers_up == [False, False, False, True]: 
        original_code = 0xC708F          
        nec_code = reverse_bytes(original_code)
        send_nec(nec_code)
        return "Arrow_Right"
    if fingers_up == [True, True, True, True]:     
        original_code = 0xCD02F        
        nec_code = reverse_bytes(original_code)
        send_nec(nec_code)
        return "Arrow_Up"
    if fingers_up == [True, False, False, True]:   
        original_code = 0xC30CF           
        nec_code = reverse_bytes(original_code)
        send_nec(nec_code)
        return "Arrow_Down"

    if sum([distance(coords[4], coords[tip]) < 0.06 for tip in [8, 12, 16, 20]]) >= 3:
        original_code = 0xC40BF         
        nec_code = reverse_bytes(original_code)
        send_nec(nec_code)
        return "Circle"
    if fingers_up == [True, True, False, False]:
        original_code = 0xC18E7           
        nec_code = reverse_bytes(original_code)
        send_nec(nec_code)
        return "Victory"
    if thumb_up and all(f is False for f in fingers_up):
        original_code = 0xC9867           
        nec_code = reverse_bytes(original_code)
        send_nec(nec_code)
        return "Thumbs"

    return "Unknown"

cap = cv2.VideoCapture(0)
last_gesture = None
last_time = 0
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = detect_gesture(hand_landmarks.landmark)

                current_time = time.time()
                if gesture != last_gesture or current_time - last_time > 0.4:
                    print(f"Gesture: {gesture}")
                    last_gesture = gesture
                    last_time = current_time

        cv2.imshow("Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:  
            GPIO.output(LED_1,GPIO.LOW)
            GPIO.output(LED_2,GPIO.LOW)
            GPIO.output(LED_3,GPIO.LOW)
            GPIO.cleanup()
            break

cap.release()
cv2.destroyAllWindows()

    