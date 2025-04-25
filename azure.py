import RPi.GPIO as GPIO 
import time
import requests
import os
import json
from picamera2 import Picamera2
from datetime import datetime
from azure.iot.device import IoTHubDeviceClient, Message

# ===== Azure IoT Hub Setup =====
AZURE_CONNECTION_STRING = "HostName=smartbin-iot-hub.azure-devices.net;DeviceId=raspberrypi-bin;SharedAccessKey=x5LbbaI9GRKdI7j69iR9je+6f4myMxR0xrFs7TyONGA="
iot_client = IoTHubDeviceClient.create_from_connection_string(AZURE_CONNECTION_STRING)

def send_data_to_azure(classification, distances):
    message = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "classification": classification,
        "distances": distances
    }
    try:
        msg = Message(json.dumps(message))
        iot_client.send_message(msg)
        print("✅ Data sent to Azure IoT Hub:", message)
    except Exception as e:
        print(f"❌ Failed to send message to Azure: {e}")

# ===== GPIO Setup =====
GPIO.setmode(GPIO.BCM)

# Ultrasonic Sensors
TRIG_PAPER = 17
ECHO_PAPER = 27
TRIG_PLASTIC = 22
ECHO_PLASTIC = 23
TRIG_METAL = 5
ECHO_METAL = 6
TRIG_GLASS = 19
ECHO_GLASS = 26

# Servo Motors
SERVO_PAPER = 18
SERVO_PLASTIC = 24
SERVO_METAL = 12
SERVO_GLASS = 16

GPIO.setup(TRIG_PAPER, GPIO.OUT)
GPIO.setup(ECHO_PAPER, GPIO.IN)
GPIO.setup(TRIG_PLASTIC, GPIO.OUT)
GPIO.setup(ECHO_PLASTIC, GPIO.IN)
GPIO.setup(TRIG_METAL, GPIO.OUT)
GPIO.setup(ECHO_METAL, GPIO.IN)
GPIO.setup(TRIG_GLASS, GPIO.OUT)
GPIO.setup(ECHO_GLASS, GPIO.IN)

GPIO.setup(SERVO_PAPER, GPIO.OUT)
GPIO.setup(SERVO_PLASTIC, GPIO.OUT)
GPIO.setup(SERVO_METAL, GPIO.OUT)
GPIO.setup(SERVO_GLASS, GPIO.OUT)

servo_paper_pwm = GPIO.PWM(SERVO_PAPER, 50)
servo_plastic_pwm = GPIO.PWM(SERVO_PLASTIC, 50)
servo_metal_pwm = GPIO.PWM(SERVO_METAL, 50)
servo_glass_pwm = GPIO.PWM(SERVO_GLASS, 50)
servo_paper_pwm.start(0)
servo_plastic_pwm.start(0)
servo_metal_pwm.start(0)
servo_glass_pwm.start(0)

# ===== Camera Setup =====
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration(main={"size": (640, 480)}))
picam2.start()

API_URL = "https://charmed-strongly-lemur.ngrok-free.app/predict"
TEMP_IMAGE_PATH = "/tmp/image.jpg"

def set_servo_angle(servo_pwm, angle):
    duty_cycle = 2 + (angle / 18)
    servo_pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)
    servo_pwm.ChangeDutyCycle(0)

def get_distance(TRIG, ECHO):
    GPIO.output(TRIG, False)
    time.sleep(0.05)
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)
    start_time = time.time()
    while GPIO.input(ECHO) == 0:
        start_time = time.time()
    while GPIO.input(ECHO) == 1:
        stop_time = time.time()
    elapsed_time = stop_time - start_time
    distance = (elapsed_time * 34300) / 2
    return round(distance, 2)

def capture_and_send_image():
    picam2.capture_file(TEMP_IMAGE_PATH)
    try:
        with open(TEMP_IMAGE_PATH, 'rb') as file:
            response = requests.post(API_URL, files={'file': file})
        os.remove(TEMP_IMAGE_PATH)

        if response.status_code == 200:
            classification = response.json().get("predicted_class", "unknown")
            return classification
        else:
            print(f"Error: API request failed with status {response.status_code}")
            return None
    except Exception as e:
        print(f"Error sending image to API: {e}")
        return None

try:
    image_timer = time.time()
    while True:
        distance_paper = get_distance(TRIG_PAPER, ECHO_PAPER)
        distance_plastic = get_distance(TRIG_PLASTIC, ECHO_PLASTIC)
        distance_metal = get_distance(TRIG_METAL, ECHO_METAL)
        distance_glass = get_distance(TRIG_GLASS, ECHO_GLASS)
        print(f"Ultrasonic Paper Distance: {distance_paper} cm | Ultrasonic Plastic Distance: {distance_plastic} cm | Ultrasonic Metal Distance: {distance_metal} cm | Ultrasonic Glass Distance: {distance_glass} cm")
        
        if time.time() - image_timer >= 4:
            classification = capture_and_send_image()
            if classification:
                print(f"Detected: {classification}")
                
                distances = {
                    "paper": distance_paper,
                    "plastic": distance_plastic,
                    "metal": distance_metal,
                    "glass": distance_glass
                }

                send_data_to_azure(classification, distances)

                if classification.lower() == "paper":
                    print("Moving Servo for Paper.")
                    set_servo_angle(servo_paper_pwm, 90)
                    time.sleep(1)
                    set_servo_angle(servo_paper_pwm, 0)
                elif classification.lower() == "plastic":
                    print("Moving Servo for Plastic.")
                    set_servo_angle(servo_plastic_pwm, 90)
                    time.sleep(1)
                    set_servo_angle(servo_plastic_pwm, 0)
                elif classification.lower() == "metal":
                    print("Moving Servo for Metal.")
                    set_servo_angle(servo_metal_pwm, 90)
                    time.sleep(1)
                    set_servo_angle(servo_metal_pwm, 0)
                elif classification.lower() == "glass":
                    print("Moving Servo for Glass.")
                    set_servo_angle(servo_glass_pwm, 90)
                    time.sleep(1)
                    set_servo_angle(servo_glass_pwm, 0)

            image_timer = time.time()
        
        time.sleep(1)

except KeyboardInterrupt:
    print("Process stopped by user.")
finally:
    servo_paper_pwm.stop()
    servo_plastic_pwm.stop()
    servo_metal_pwm.stop()
    servo_glass_pwm.stop()
    GPIO.cleanup()
    picam2.stop()
    iot_client.shutdown()
