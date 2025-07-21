import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def draw_lines(img, lines):
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0,255,0), 5)

cap = cv2.VideoCapture("solidWhiteRight.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    height = frame.shape[0]
    width = frame.shape[1]
    roi_vertices = [(0, height), (width/2, height/2), (width, height)]
    cropped = region_of_interest(edges, np.array([roi_vertices], np.int32))

    lines = cv2.HoughLinesP(cropped, 1, np.pi/180, 50, maxLineGap=100)
    if lines is not None:
        draw_lines(frame, lines)

    cv2.imshow("Lane Detection", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
