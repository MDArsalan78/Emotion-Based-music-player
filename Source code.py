from flask import Flask, render_template, request
import numpy as np
import cv2
from keras.models import load_model
import webbrowser

app = Flask(__name__)

app.config[&#39;SEND_FILE_MAX_AGE_DEFAULT&#39;] = 1

info = {}

haarcascade = &quot;haarcascade_frontalface_default.xml&quot;
label_map = [&#39;Anger&#39;, &#39;Neutral&#39;, &#39;Fear&#39;, &#39;Happy&#39;, &#39;Sad&#39;, &#39;Surprise&#39;]
print(&quot;+&quot;*50, &quot;loadin gmmodel&quot;)
model = load_model(&#39;model.h5&#39;)
cascade = cv2.CascadeClassifier(haarcascade)

@app.route(&#39;/&#39;)
def index():
return render_template(&#39;index.html&#39;)

@app.route(&#39;/choose_singer&#39;, methods = [&quot;POST&quot;])
def choose_singer():
info[&#39;language&#39;] = request.form[&#39;language&#39;]

print(info)
return render_template(&#39;choose_singer.html&#39;, data = info[&#39;language&#39;])

@app.route(&#39;/emotion_detect&#39;, methods=[&quot;POST&quot;])
def emotion_detect():
info[&#39;singer&#39;] = request.form[&#39;singer&#39;]

found = False

cap = cv2.VideoCapture(0)
while not(found):
_, frm = cap.read()
gray = cv2.cvtColor(frm,cv2.COLOR_BGR2GRAY)

faces = cascade.detectMultiScale(gray, 1.4, 1)

for x,y,w,h in faces:
found = True
roi = gray[y:y+h, x:x+w]
cv2.imwrite(&quot;static/face.jpg&quot;, roi)

roi = cv2.resize(roi, (48,48))

roi = roi/255.0

roi = np.reshape(roi, (1,48,48,1))

prediction = model.predict(roi)

print(prediction)

prediction = np.argmax(prediction)
prediction = label_map[prediction]

cap.release()

link =
f&quot;https://www.youtube.com/results?search_query={info[&#39;singer&#39;]}+{prediction}+{info[&#39;language&#39;]}+song&quot;
webbrowser.open(link)

return render_template(&quot;emotion_detect.html&quot;, data=prediction, link=link)

if __name__ == &quot;__main__&quot;:
app.run(debug=True)
