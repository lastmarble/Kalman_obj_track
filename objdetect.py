import cv2
from Detector import detect
from KalmanFilter import KalmanFilter


def main():
    # Create opencv video capture object
    VideoCap = cv2.VideoCapture('video_preview_h264.mp4')

    # Variable used to control the speed of reading the video
    ControlSpeedVar = 100  # Lowest: 1 - Highest:100

    HiSpeed = 100

    # Create KalmanFilter object KF
    # KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)

    KF0 = KalmanFilter(0.1, 1, 1, 1, 0.1, 0.1)
    KF1 = KalmanFilter(0.1, 1, 1, 1, 0.1, 0.1)

    debugMode = 1

    while True:
        # Read frame
        ret, frame = VideoCap.read()

        # Detect object
        centers = detect(frame, debugMode)

        # If centroids are detected then track them
        def iterate(length, centers, obj):
            if len(centers) > 0:
                for i in length:
                    # Predict
                    (x, y) = obj.predict()
                    # Draw a rectangle as the predicted object position
                    cv2.rectangle(frame, (int(x - 49), int(y - 49)), (int(x + 49), int(y + 49)), (255, 0, 0), 2)

                    # Update
                    (x1, y1) = obj.update(centers[i])

                    # Draw a rectangle as the estimated object position
                    cv2.rectangle(frame, (int(x1 - 49), int(y1 - 49)), (int(x1 + 49), int(y1 + 49)), (0, 0, 255), 2)

                    cv2.putText(frame, "Estimated Position", (int(x1 + 15), int(y1 + 10)), 0, 0.5, (0, 0, 255), 2)
                    cv2.putText(frame, "Predicted Position", (int(x + 15), int(y)), 0, 0.5, (255, 0, 0), 2)
                    cv2.putText(frame, "Measured Position", (int(centers[i][0] + 15), int(centers[i][1] - 15)), 0, 0.5,
                                (0, 191, 255), 2)

                cv2.imshow('image', frame)

        iterate([0], centers, KF0)
        iterate([1], centers, KF1)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            VideoCap.release()
            cv2.destroyAllWindows()
            break

        cv2.waitKey(HiSpeed - ControlSpeedVar + 1)


if __name__ == "__main__":
    # execute main
    main()
