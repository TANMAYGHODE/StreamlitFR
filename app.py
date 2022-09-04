import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

st.title('Face Recognition App')
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Face Mesh Application using MediaPipe')
st.sidebar.subheader('Parameters')

@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

st.set_option('deprecation.showfileUploaderEncoding', False)

max_faces = st.sidebar.number_input('Maximum Number of Faces', value=1, min_value=1,max_value=10)
st.sidebar.markdown('---')
detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)

st.sidebar.markdown('---')

st.markdown(' ## Output')

stframe1 = st.empty()
stframe2 = st.empty()
stframe3 = st.empty()

vid = cv2.VideoCapture(0)


width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = int(vid.get(cv2.CAP_PROP_FPS))

#codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
codec = cv2.VideoWriter_fourcc('V','P','0','9')
out = cv2.VideoWriter('output1.mp4', codec, 15, (width, height))
# out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

st.sidebar.text('Input Video')

fps = 0
i = 0
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

kpi1, kpi2, kpi3 = st.columns(3)



st.markdown("<hr/>", unsafe_allow_html=True)

count=0

with mp_face_mesh.FaceMesh(
min_detection_confidence=detection_confidence,
min_tracking_confidence=tracking_confidence , 
max_num_faces = max_faces) as face_mesh:
    prevTime = 0

    with kpi1:
        st.markdown("**FrameRate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Detected Faces**")
        kpi2_text = st.markdown("0")

    with kpi3:
        st.markdown("**Image Width**")
        kpi3_text = st.markdown("0")

    while vid.isOpened():
        i +=1
        ret, frame = vid.read()
        if not ret:
            continue
        count=count+1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        face_count = 0
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_count += 1
                mp_drawing.draw_landmarks(
                image = frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        #Dashboard
        kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
        kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
        kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

        frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)
        frame = image_resize(image = frame, width = 640)
        # out.write(frame)
        stframe1.image([frame,frame],channels = 'BGR',use_column_width=True)
        stframe2.image(frame,channels = 'BGR',use_column_width=True)
        stframe3.image(frame,channels = 'BGR',use_column_width=True)
        print(count)

        if count==1000:
            break

# output_video = open('output1.mp4','rb')
# out_bytes = output_video.read()
# st.video(out_bytes)
vid.release()
out.release()