import pickle

import cv2
import mediapipe as mp
import numpy as np

# model_dict1 = pickle.load(open('./model1.p', 'rb'))
# model_dict2 = pickle.load(open('./model2.p', 'rb'))
# model1 = model_dict1['model1']
# model2 = model_dict2['model2']

with open('model1.p', 'rb') as f1:
    model1 = pickle.load(f1)

with open('model2.p', 'rb') as f2:
    model2 = pickle.load(f2)


cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2 ,min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'H', 3:'C'}
while True:

    data_aux = []
    x_ = []
    y_ = []

    data_aux2 = []
    x_2 = []
    y_2 = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        n = len(results.multi_hand_landmarks)
        if n==1:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
        
                
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction1 = model1.predict([np.asarray(data_aux)])

            predicted_character1 = labels_dict[int(prediction1[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character1, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)
        else:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    xtwo = hand_landmarks.landmark[i].x
                    ytwo = hand_landmarks.landmark[i].y

                    x_2.append(xtwo)
                    y_2.append(ytwo)

                for i in range(len(hand_landmarks.landmark)):
                    xtwo = hand_landmarks.landmark[i].x
                    ytwo = hand_landmarks.landmark[i].y
                    data_aux2.append(xtwo - min(x_2))
                    data_aux2.append(ytwo - min(y_2))
        
                
            x1 = int(min(x_2) * W) - 10
            y1 = int(min(y_2) * H) - 10

            x2 = int(max(x_2) * W) - 10
            y2 = int(max(y_2) * H) - 10

            prediction2 = model2.predict([np.asarray(data_aux2)])

            predicted_character2 = labels_dict[int(prediction2[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character2, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

        

    cv2.imshow('frame', frame)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()
