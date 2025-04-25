import RPi.GPIO as GPIO
import time
import requests
import os
import psycopg2
from picamera2 import Picamera2
from datetime import datetime

# Database Configuration
DB_CONFIG = {
    "dbname": "railway",
    "user": "postgres",
    "password": "BYQILvuJUEckVHrvpfNUTaRayUAsVlBO",
    "host": "interchange.proxy.rlwy.net",
    "port": "15479"
}

# GPIO setup
GPIO.setmode(GPIO.BCM)

# Ultrasonic Sensors
TRIG_PAPER, ECHO_PAPER = 17, 27
TRIG_PLASTIC, ECHO_PLASTIC = 22, 23
TRIG_METAL, ECHO_METAL = 5, 6
TRIG_GLASS, ECHO_GLASS = 19, 26

# Servo Motors
SERVO_PAPER, SERVO_PLASTIC, SERVO_METAL, SERVO_GLASS = 18, 24, 12, 16

# Configure GPIO
for trig, echo in [(TRIG_PAPER, ECHO_PAPER), (TRIG_PLASTIC, ECHO_PLASTIC), 
                   (TRIG_METAL, ECHO_METAL), (TRIG_GLASS, ECHO_GLASS)]:
    GPIO.setup(trig, GPIO.OUT)
    GPIO.setup(echo, GPIO.IN)

for servo in [SERVO_PAPER, SERVO_PLASTIC, SERVO_METAL, SERVO_GLASS]:
    GPIO.setup(servo, GPIO.OUT)

# Setup PWM for servos
servo_pwm = {
    "paper": GPIO.PWM(SERVO_PAPER, 50),
    "plastic": GPIO.PWM(SERVO_PLASTIC, 50),
    "metal": GPIO.PWM(SERVO_METAL, 50),
    "glass": GPIO.PWM(SERVO_GLASS, 50)
}
for pwm in servo_pwm.values():
    pwm.start(0)

# Initialize Camera
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration(main={"size": (640, 480)}))
picam2.start()

# API & Image Path
API_URL = "https://charmed-strongly-lemur.ngrok-free.app/predict"
TEMP_IMAGE_PATH = "/tmp/image.jpg"

def set_servo_angle(servo_pwm, angle):
    """Move the specified servo to a given angle (0° to 180°)."""
    duty_cycle = 2 + (angle / 18)  # Convert angle to duty cycle
    servo_pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)
    servo_pwm.ChangeDutyCycle(0)  # Stop PWM to prevent jittering

def get_distance(TRIG, ECHO):
    """Measure distance using HC-SR04 ultrasonic sensor."""
    GPIO.output(TRIG, False)
    time.sleep(0.05)
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)
    
    start_time = time.time()
    while GPIO.input(ECHO) == 0:
        start_time = time.time()
    
    stop_time = time.time()
    while GPIO.input(ECHO) == 1:
        stop_time = time.time()
    
    elapsed_time = stop_time - start_time
    distance = (elapsed_time * 34300) / 2  # Speed of sound = 34300 cm/s
    return round(distance, 2)

def insert_waste_record(type_id, image_path):
    """Insert a new waste record into the Waste table."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()

        query = "INSERT INTO Waste (TypeID, Image, ThrowDate) VALUES (%s, %s, %s)"
        cursor.execute(query, (type_id, psycopg2.Binary(image_data), datetime.now().date()))
        
        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ Inserted waste record - TypeID: {type_id}, Date: {datetime.now().date()}")
    
    except Exception as e:
        print(f"❌ Database error while inserting waste record: {e}")

def capture_and_send_image():
    """Capture an image and send it to the API for classification."""
    picam2.capture_file(TEMP_IMAGE_PATH)
    
    try:
        with open(TEMP_IMAGE_PATH, 'rb') as file:
            response = requests.post(API_URL, files={'file': file})

        if response.status_code == 200:
            classification = response.json().get("predicted_class", "unknown").lower()
            return classification, TEMP_IMAGE_PATH
        else:
            print(f"❌ Error: API request failed with status {response.status_code}")
            return None, None
    except Exception as e:
        print(f"❌ Error sending image to API: {e}")
        return None, None

def update_fill_level(bin_id, distance_cm):
    """Update the fill level of the smart bin in the database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("UPDATE smartbin SET filllevel = %s WHERE binid = %s", (distance_cm, bin_id))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ Updated SmartBin {bin_id} FillLevel to {distance_cm} cm")
    except Exception as e:
        print(f"❌ Database error: {e}")

def process_waste(classification, image_path):
    """Move the servo, capture distance, update database, and store waste record."""
    bin_mapping = {"paper": (1, TRIG_PAPER, ECHO_PAPER, servo_pwm["paper"]),
                   "plastic": (2, TRIG_PLASTIC, ECHO_PLASTIC, servo_pwm["plastic"]),
                   "metal": (3, TRIG_METAL, ECHO_METAL, servo_pwm["metal"]),
                   "glass": (4, TRIG_GLASS, ECHO_GLASS, servo_pwm["glass"])}

    if classification in bin_mapping:
        bin_id, TRIG, ECHO, servo = bin_mapping[classification]

        print(f"🔄 Moving Servo for {classification.capitalize()}")
        set_servo_angle(servo, 90)
        time.sleep(1)
        set_servo_angle(servo, 0)

        # Insert waste record into database
        insert_waste_record(bin_id, image_path)

        # Update fill level in smartbin table
        distance = get_distance(TRIG, ECHO)
        update_fill_level(bin_id, distance)

try:
    image_timer = time.time()
    while True:
        # Get distance readings and update fill levels
        bin_mapping = {"paper": (1, TRIG_PAPER, ECHO_PAPER, servo_pwm["paper"]),
                       "plastic": (2, TRIG_PLASTIC, ECHO_PLASTIC, servo_pwm["plastic"]),
                       "metal": (3, TRIG_METAL, ECHO_METAL, servo_pwm["metal"]),
                       "glass": (4, TRIG_GLASS, ECHO_GLASS, servo_pwm["glass"])}

        for waste_type, (bin_id, TRIG, ECHO, servo) in bin_mapping.items():
            distance = get_distance(TRIG, ECHO)
            update_fill_level(bin_id, distance)

        # Capture and process image every 2 seconds
        if time.time() - image_timer >= 2:
            classification, image_path = capture_and_send_image()
            if classification:
                print(f"🗑️ Detected: {classification}")
                process_waste(classification, image_path)
            image_timer = time.time()  # Reset timer

        time.sleep(1)  # Ultrasonic sensor reads every 1 second

except KeyboardInterrupt:
    print("🛑 Process stopped by user.")
finally:
    # Cleanup
    for pwm in servo_pwm.values():
        pwm.stop()
    GPIO.cleanup()
    picam2.stop()
