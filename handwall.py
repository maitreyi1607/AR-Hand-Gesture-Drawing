import cv2
import mediapipe as mp
import numpy as np

cv2.setNumThreads(2)

# Camera
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# MediaPipe
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Drawing settings
NEON_PINK = (92, 89, 255)
WHITE = (255, 255, 255)

DRAW_THICKNESS = 10
CORE_THICKNESS = 4

ERASER_SIZE = 30

# Smoothing
SMOOTHING = 0.25

smooth_x = None
smooth_y = None

previous_x = None
previous_y = None

glow_mask = None
core_mask = None

eraser_mode = False


# Finger gesture detection
def index_only_gesture(landmarks):

    index_up = (
        landmarks[8].y < landmarks[6].y - 0.02
    )

    middle_folded = (
        landmarks[12].y > landmarks[10].y
    )

    ring_folded = (
        landmarks[16].y > landmarks[14].y
    )

    pinky_folded = (
        landmarks[20].y > landmarks[18].y
    )

    return (
        index_up
        and middle_folded
        and ring_folded
        and pinky_folded
    )


def eraser_gesture(landmarks):

    index_up = (
        landmarks[8].y < landmarks[6].y - 0.02
    )

    middle_up = (
        landmarks[12].y < landmarks[10].y - 0.02
    )

    ring_folded = (
        landmarks[16].y > landmarks[14].y
    )

    pinky_folded = (
        landmarks[20].y > landmarks[18].y
    )

    return (
        index_up
        and middle_up
        and ring_folded
        and pinky_folded
    )


# Main loop
while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    if glow_mask is None:
        glow_mask = np.zeros((height, width), dtype=np.uint8)
        core_mask = np.zeros((height, width), dtype=np.uint8)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    drawing_active = False
    gesture_eraser_active = False

    cursor_x = None
    cursor_y = None

    # Hand detection
    if results.multi_hand_landmarks:

        landmarks = results.multi_hand_landmarks[0].landmark

        drawing_active = index_only_gesture(landmarks)

        gesture_eraser_active = eraser_gesture(landmarks)

        hand_active = drawing_active or gesture_eraser_active

        active_eraser = eraser_mode or gesture_eraser_active

        if hand_active:

            raw_x = int(landmarks[8].x * width)
            raw_y = int(landmarks[8].y * height)

            # Smooth fingertip movement
            if smooth_x is None:

                smooth_x = raw_x
                smooth_y = raw_y

            else:

                smooth_x = (
                    SMOOTHING * raw_x
                    + (1 - SMOOTHING) * smooth_x
                )

                smooth_y = (
                    SMOOTHING * raw_y
                    + (1 - SMOOTHING) * smooth_y
                )

            cursor_x = int(smooth_x)
            cursor_y = int(smooth_y)

            # Draw or erase
            if previous_x is not None:

                if active_eraser:

                    cv2.line(
                        glow_mask,
                        (previous_x, previous_y),
                        (cursor_x, cursor_y),
                        0,
                        ERASER_SIZE,
                        cv2.LINE_AA
                    )

                    cv2.line(
                        core_mask,
                        (previous_x, previous_y),
                        (cursor_x, cursor_y),
                        0,
                        ERASER_SIZE,
                        cv2.LINE_AA
                    )

                else:

                    cv2.line(
                        glow_mask,
                        (previous_x, previous_y),
                        (cursor_x, cursor_y),
                        255,
                        DRAW_THICKNESS,
                        cv2.LINE_AA
                    )

                    cv2.line(
                        core_mask,
                        (previous_x, previous_y),
                        (cursor_x, cursor_y),
                        255,
                        CORE_THICKNESS,
                        cv2.LINE_AA
                    )

            previous_x = cursor_x
            previous_y = cursor_y

        else:

            previous_x = None
            previous_y = None

            smooth_x = None
            smooth_y = None

    else:

        previous_x = None
        previous_y = None

        smooth_x = None
        smooth_y = None

        hand_active = False
        active_eraser = eraser_mode

    # Neon effect
    display_frame = frame.copy()

    blurred = cv2.GaussianBlur(
        glow_mask,
        (0, 0),
        12
    )

    alpha = (
        blurred.astype(np.float32) / 255.0
    ) * 0.75

    alpha = alpha[:, :, np.newaxis]

    pink_layer = np.zeros_like(display_frame)
    pink_layer[:] = NEON_PINK

    display_frame = (
        display_frame.astype(np.float32) * (1 - alpha)
        + pink_layer.astype(np.float32) * alpha
    ).astype(np.uint8)

    display_frame[glow_mask > 0] = NEON_PINK
    display_frame[core_mask > 0] = WHITE

    # Fingertip dot
    if cursor_x is not None:

        cv2.circle(
            display_frame,
            (cursor_x, cursor_y),
            3,
            WHITE,
            -1,
            cv2.LINE_AA
        )

    # Status
    if hand_active:

        status = "ERASING" if active_eraser else "DRAWING"

    else:

        status = "ERASER ON" if eraser_mode else "READY"

    cv2.putText(
        display_frame,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        NEON_PINK,
        2,
        cv2.LINE_AA
    )

    # Display
    cv2.imshow(
        "HandWall - AR Virtual Writing",
        display_frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):

        glow_mask[:] = 0
        core_mask[:] = 0

    elif key == ord('e'):

        eraser_mode = not eraser_mode

        previous_x = None
        previous_y = None

        print("Eraser:", "ON" if eraser_mode else "OFF")

    elif key == ord('s'):

        cv2.imwrite(
            "handwall_capture.png",
            display_frame
        )

        print("Image saved!")

    elif key == ord('q'):
        break


cap.release()
hands.close()
cv2.destroyAllWindows()