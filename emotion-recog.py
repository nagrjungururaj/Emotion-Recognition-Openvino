import cv2
import numpy as np

face_model = 'models/face-detection-retail-0005'
emote_model = 'models/emotions-recognition-retail-0003'
path =  'in.mp4'

cap = cv2.VideoCapture(path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

writer = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (frame_width,frame_height))

def putlabels(x):

    if str(x) == '0':
        return 'neutral'
    elif str(x) == '1':
        return 'happy'
    elif str(x) == '2':
        return 'sad'
    elif str(x) == '3':
        return 'surprise'
    else:
        return 'anger'

def net_face(frame, shape):

    net = cv2.dnn.readNet(face_model + '.xml', face_model+ '.bin')
    blob = cv2.dnn.blobFromImage(frame, size=shape, ddepth=cv2.CV_8U)
    net.setInput(blob)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    out = net.forward()
    return out

def net_emote(frame, shape):

    net = cv2.dnn.readNet(emote_model + '.xml', emote_model + '.bin')
    blob = cv2.dnn.blobFromImage(frame, size=shape, ddepth=cv2.CV_8U)
    net.setInput(blob)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    out = net.forward()
    return out
    
while(True):
    ret, frame = cap.read()
    
    if ret == False:
        print('End of video')
        break

    out = net_face(frame, (300,300))
    
    for detection in out.reshape(-1,7):
            
        conf = float(detection[2])
        xmin = int(detection[3] * frame.shape[1])
        ymin = int(detection[4] * frame.shape[0])
        xmax = int(detection[5] * frame.shape[1])
        ymax = int(detection[6] * frame.shape[0])

        if conf > 0.65:

            soft = net_emote(frame,(64,64))
            emo_pred = np.reshape(soft, (5,))
            emo_pred = list(emo_pred)
            
            cv2.rectangle(frame, (xmin,ymin),(xmax,ymax), (0,128,255), 2)    
            cv2.putText(frame, putlabels(emo_pred.index(max(emo_pred))) , (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
        cv2.imshow('Emotion-Recognition',frame)
        
    writer.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
writer.release()
