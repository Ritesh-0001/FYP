from django.shortcuts import render,redirect
from .forms import usernameForm,DateForm,UsernameAndDateForm, DateForm_2
from django.contrib import messages
from django.contrib.auth.models import User
import cv2
import dlib
import imutils
from imutils import face_utils
from imutils.video import VideoStream
from imutils.face_utils import rect_to_bb
from imutils.face_utils import FaceAligner
import time
from attendance_system_facial_recognition.settings import BASE_DIR
import os
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
from django.contrib.auth.decorators import login_required
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import datetime
from django_pandas.io import read_frame
from users.models import Present, Time
import seaborn as sns
import pandas as pd
from django.db.models import Count
#import mpld3
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from matplotlib import rcParams
import math
from imutils.face_utils import rect_to_bb
from django.views.decorators.csrf import ensure_csrf_cookie

mpl.use('Agg')


#utility functions:
def username_present(username):
	if User.objects.filter(username=username).exists():
		return True
	
	return False
import os
import cv2
import dlib
import numpy as np
import imutils
from imutils.face_utils import FaceAligner, rect_to_bb
from imutils.video import VideoStream
import face_recognition

import os
import cv2
import dlib
import numpy as np
import imutils
import face_recognition
from imutils.video import VideoStream
from imutils.face_utils import FaceAligner

def rect_to_bb(rect):
    try:
        x = rect.left()
        y = rect.top()
        w = rect.width()
        h = rect.height()
        return (x, y, w, h)
    except AttributeError:
        return (None, None, None, None)


import os
import cv2
import dlib
import numpy as np
import face_recognition
from imutils.video import VideoStream
import imutils
from imutils import face_utils
from pathlib import Path

def create_dataset(username: str, sample_limit: int = 300) -> None:
    directory = Path('face_recognition_data/training_dataset') / str(username)
    directory.mkdir(parents=True, exist_ok=True)
    
    predictor_path = Path('face_recognition_data/shape_predictor_68_face_landmarks.dat')
    if not predictor_path.exists():
        raise FileNotFoundError(f"Shape predictor file not found at {predictor_path}")
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(predictor_path))
    
    vs = VideoStream(src=0)
    vs.start()
    
    sample_num = 0
    
    try:
        while sample_num <= sample_limit:
            frame = vs.read()
            if frame is None:
                continue
                
            frame = imutils.resize(frame, width=800)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray_frame, 0)
            
            for face in faces:
                try:
                    x, y, w, h = face_utils.rect_to_bb(face)
                    
                    # Direct face extraction instead of using FaceAligner
                    face_region = frame[y:y+h, x:x+w]
                    if face_region.size == 0:
                        continue
                        
                    face_aligned = cv2.resize(face_region, (256, 256))
                    
                    sample_num += 1
                    face_path = directory / f"{sample_num}.jpg"
                    cv2.imwrite(str(face_path), face_aligned)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                except Exception as e:
                    print(f"[WARNING] Face processing error: {str(e)}")
                    continue
            
            cv2.imshow("Add Images", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n[INFO] Data collection stopped by user")
    finally:
        vs.stop()
        cv2.destroyAllWindows()

def predict(face_aligned, svc, threshold: float = 0.7) -> tuple:
    if face_aligned is None:
        return ([-1], [0])
        
    try:
        face_locations = face_recognition.face_locations(face_aligned, model="hog")
        if not face_locations:
            return ([-1], [0])
            
        face_encodings = face_recognition.face_encodings(face_aligned, face_locations, model="small")
        if not face_encodings:
            return ([-1], [0])
            
        probabilities = svc.predict_proba(face_encodings)
        max_prob_idx = np.argmax(probabilities[0])
        max_prob = probabilities[0][max_prob_idx]
        
        return ([max_prob_idx], [max_prob]) if max_prob > threshold else ([-1], [max_prob])
        
    except Exception as e:
        print(f"[WARNING] Prediction error: {str(e)}")
        return ([-1], [0])

def vizualize_Data(embedded, targets,):
	
	X_embedded = TSNE(n_components=2).fit_transform(embedded)

	for i, t in enumerate(set(targets)):
		idx = targets == t
		plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)

	plt.legend(bbox_to_anchor=(1, 1));
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()	
	plt.savefig('./recognition/static/recognition/img/training_visualisation.png')
	plt.close()



def update_attendance_in_db_in(present):
	today=datetime.date.today()
	time=datetime.datetime.now()
	for person in present:
		user=User.objects.get(username=person)
		try:
		   qs=Present.objects.get(user=user,date=today)
		except :
			qs= None
		
		if qs is None:
			if present[person]==True:
						a=Present(user=user,date=today,present=True)
						a.save()
			else:
				a=Present(user=user,date=today,present=False)
				a.save()
		else:
			if present[person]==True:
				qs.present=True
				qs.save(update_fields=['present'])
		if present[person]==True:
			a=Time(user=user,date=today,time=time, out=False)
			a.save()


			
		




def update_attendance_in_db_out(present):
	today=datetime.date.today()
	time=datetime.datetime.now()
	for person in present:
		user=User.objects.get(username=person)
		if present[person]==True:
			a=Time(user=user,date=today,time=time, out=True)
			a.save()
		




def check_validity_times(times_all):

	if(len(times_all)>0):
		sign=times_all.first().out
	else:
		sign=True
	times_in=times_all.filter(out=False)
	times_out=times_all.filter(out=True)
	if(len(times_in)!=len(times_out)):
		sign=True
	break_hourss=0
	if(sign==True):
			check=False
			break_hourss=0
			return (check,break_hourss)
	prev=True
	prev_time=times_all.first().time

	for obj in times_all:
		curr=obj.out
		if(curr==prev):
			check=False
			break_hourss=0
			return (check,break_hourss)
		if(curr==False):
			curr_time=obj.time
			to=curr_time
			ti=prev_time
			break_time=((to-ti).total_seconds())/3600
			break_hourss+=break_time


		else:
			prev_time=obj.time

		prev=curr

	return (True,break_hourss)


def convert_hours_to_hours_mins(hours):
	
	h=int(hours)
	hours-=h
	m=hours*60
	m=math.ceil(m)
	return str(str(h)+ " hrs " + str(m) + "  mins")

		

#used
def hours_vs_date_given_employee(present_qs,time_qs,admin=True):
	register_matplotlib_converters()
	df_hours=[]
	df_break_hours=[]
	qs=present_qs

	for obj in qs:
		date=obj.date
		times_in=time_qs.filter(date=date).filter(out=False).order_by('time')
		times_out=time_qs.filter(date=date).filter(out=True).order_by('time')
		times_all=time_qs.filter(date=date).order_by('time')
		obj.time_in=None
		obj.time_out=None
		obj.hours=0
		obj.break_hours=0
		if (len(times_in)>0):			
			obj.time_in=times_in.first().time
			
		if (len(times_out)>0):
			obj.time_out=times_out.last().time

		if(obj.time_in is not None and obj.time_out is not None):
			ti=obj.time_in
			to=obj.time_out
			hours=((to-ti).total_seconds())/3600
			obj.hours=hours
		

		else:
			obj.hours=0

		(check,break_hourss)= check_validity_times(times_all)
		if check:
			obj.break_hours=break_hourss


		else:
			obj.break_hours=0


		
		df_hours.append(obj.hours)
		df_break_hours.append(obj.break_hours)
		obj.hours=convert_hours_to_hours_mins(obj.hours)
		obj.break_hours=convert_hours_to_hours_mins(obj.break_hours)
			
	
	
	
	df = read_frame(qs)	
	
	
	df["hours"]=df_hours
	df["break_hours"]=df_break_hours

	print(df)
	
	sns.barplot(data=df,x='date',y='hours')
	plt.xticks(rotation='vertical')
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()
	if(admin):
		plt.savefig('./recognition/static/recognition/img/attendance_graphs/hours_vs_date/1.png')
		plt.close()
	else:
		plt.savefig('./recognition/static/recognition/img/attendance_graphs/employee_login/1.png')
		plt.close()
	return qs
	

#used
def hours_vs_employee_given_date(present_qs,time_qs):
	register_matplotlib_converters()
	df_hours=[]
	df_break_hours=[]
	df_username=[]
	qs=present_qs

	for obj in qs:
		user=obj.user
		times_in=time_qs.filter(user=user).filter(out=False)
		times_out=time_qs.filter(user=user).filter(out=True)
		times_all=time_qs.filter(user=user)
		obj.time_in=None
		obj.time_out=None
		obj.hours=0
		obj.hours=0
		if (len(times_in)>0):			
			obj.time_in=times_in.first().time
		if (len(times_out)>0):
			obj.time_out=times_out.last().time
		if(obj.time_in is not None and obj.time_out is not None):
			ti=obj.time_in
			to=obj.time_out
			hours=((to-ti).total_seconds())/3600
			obj.hours=hours
		else:
			obj.hours=0
		(check,break_hourss)= check_validity_times(times_all)
		if check:
			obj.break_hours=break_hourss


		else:
			obj.break_hours=0

		
		df_hours.append(obj.hours)
		df_username.append(user.username)
		df_break_hours.append(obj.break_hours)
		obj.hours=convert_hours_to_hours_mins(obj.hours)
		obj.break_hours=convert_hours_to_hours_mins(obj.break_hours)

	



	df = read_frame(qs)	
	df['hours']=df_hours
	df['username']=df_username
	df["break_hours"]=df_break_hours


	sns.barplot(data=df,x='username',y='hours')
	plt.xticks(rotation='vertical')
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()
	plt.savefig('./recognition/static/recognition/img/attendance_graphs/hours_vs_employee/1.png')
	plt.close()
	return qs


def total_number_employees():
	qs=User.objects.all()
	return (len(qs) -1)
	# -1 to account for admin 



def employees_present_today():
	today=datetime.date.today()
	qs=Present.objects.filter(date=today).filter(present=True)
	return len(qs)




#used	
def this_week_emp_count_vs_date():
	today=datetime.date.today()
	some_day_last_week=today-datetime.timedelta(days=7)
	monday_of_last_week=some_day_last_week-  datetime.timedelta(days=(some_day_last_week.isocalendar()[2] - 1))
	monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)
	qs=Present.objects.filter(date__gte=monday_of_this_week).filter(date__lte=today)
	str_dates=[]
	emp_count=[]
	str_dates_all=[]
	emp_cnt_all=[]
	cnt=0
	
	



	for obj in qs:
		date=obj.date
		str_dates.append(str(date))
		qs=Present.objects.filter(date=date).filter(present=True)
		emp_count.append(len(qs))


	while(cnt<5):

		date=str(monday_of_this_week+datetime.timedelta(days=cnt))
		cnt+=1
		str_dates_all.append(date)
		if(str_dates.count(date))>0:
			idx=str_dates.index(date)

			emp_cnt_all.append(emp_count[idx])
		else:
			emp_cnt_all.append(0)

	
	
	



	df=pd.DataFrame()
	df["date"]=str_dates_all
	df["Number of employees"]=emp_cnt_all
	
	
	sns.lineplot(data=df,x='date',y='Number of employees')
	plt.savefig('./recognition/static/recognition/img/attendance_graphs/this_week/1.png')
	plt.close()






#used
def last_week_emp_count_vs_date():
	today=datetime.date.today()
	some_day_last_week=today-datetime.timedelta(days=7)
	monday_of_last_week=some_day_last_week-  datetime.timedelta(days=(some_day_last_week.isocalendar()[2] - 1))
	monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)
	qs=Present.objects.filter(date__gte=monday_of_last_week).filter(date__lt=monday_of_this_week)
	str_dates=[]
	emp_count=[]


	str_dates_all=[]
	emp_cnt_all=[]
	cnt=0
	
	



	for obj in qs:
		date=obj.date
		str_dates.append(str(date))
		qs=Present.objects.filter(date=date).filter(present=True)
		emp_count.append(len(qs))


	while(cnt<5):

		date=str(monday_of_last_week+datetime.timedelta(days=cnt))
		cnt+=1
		str_dates_all.append(date)
		if(str_dates.count(date))>0:
			idx=str_dates.index(date)

			emp_cnt_all.append(emp_count[idx])
			
		else:
			emp_cnt_all.append(0)

	
	
	



	df=pd.DataFrame()
	df["date"]=str_dates_all
	df["emp_count"]=emp_cnt_all
	

	
	
	sns.lineplot(data=df,x='date',y='emp_count')
	plt.savefig('./recognition/static/recognition/img/attendance_graphs/last_week/1.png')
	plt.close()


		





# Create your views here.
def home(request):

	return render(request, 'recognition/home.html')

@login_required
def dashboard(request):
	if(request.user.username=='admin'):
		print("admin")
		return render(request, 'recognition/admin_dashboard.html')
	else:
		print("not admin")

		return render(request,'recognition/employee_dashboard.html')

@login_required
def add_photos(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	if request.method=='POST':
		form=usernameForm(request.POST)
		data = request.POST.copy()
		username=data.get('username')
		if username_present(username):
			create_dataset(username)
			messages.success(request, f'Dataset Created')
			return redirect('add-photos')
		else:
			messages.warning(request, f'No such username found. Please register employee first.')
			return redirect('dashboard')


	else:
		

			form=usernameForm()
			return render(request,'recognition/add_photos.html', {'form' : form})



import cv2
import dlib
import numpy as np
import pickle
import time
import datetime
from pathlib import Path
from imutils.video import VideoStream
import imutils
from imutils import face_utils
from sklearn.preprocessing import LabelEncoder
import face_recognition
from django.shortcuts import redirect

# def mark_your_attendance(request):
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmar

#     with open(svc_save_path, 'rb') as f:
#         svc = pickle.load(f)
    
#     encoder = LabelEncoder()
#     encoder.classes_ = np.load('face_recognition_data/classes.npy')

#     faces_encodings = np.zeros((1, 128))
#     no_of_faces = len(svc.predict_proba(faces_encodings)[0])
#     count = dict()
#     present = dict()
#     log_time = dict()
#     start = dict()
    
#     for i in range(no_of_faces):
#         count[encoder.inverse_transform([i])[0]] = 0
#         present[encoder.inverse_transform([i])[0]] = False

#     vs = VideoStream(src=0).start()
    
#     while True:
#         frame = vs.read()
#         if frame is None:
#             continue
            
#         frame = imutils.resize(frame, width=800)
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detector(gray_frame, 0)

#         for face in faces:
#             try:
#                 x, y, w, h = face_utils.rect_to_bb(face)
                
#                 # Direct face extraction and resizing
#                 face_region = frame[max(y, 0):min(y+h, frame.shape[0]), 
#                                   max(x, 0):min(x+w, frame.shape[1])]
#                 if face_region.size == 0:
#                     continue
                    
#                 face_aligned = cv2.resize(face_region, (96, 96))
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
                
#                 pred, prob = predict(face_aligned, svc)

#                 if pred != [-1]:
#                     person_name = encoder.inverse_transform(np.ravel([pred]))[0]
#                     pred = person_name
                    
#                     if count[pred] == 0:
#                         start[pred] = time.time()
#                         count[pred] = count.get(pred, 0) + 1

#                     if count[pred] == 4 and (time.time()-start[pred]) > 1.2:
#                         count[pred] = 0
#                     else:
#                         present[pred] = True
#                         log_time[pred] = datetime.datetime.now()
#                         count[pred] = count.get(pred, 0) + 1
#                         print(pred, present[pred], count[pred])
                    
#                     cv2.putText(frame, f"{person_name} {prob:.2f}", 
#                               (x+6, y+h-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#                 else:
#                     cv2.putText(frame, "unknown", 
#                               (x+6, y+h-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
#             except Exception as e:
#                 print(f"[WARNING] Face processing error: {str(e)}")
#                 continue

#         cv2.imshow("Mark Attendance - In - Press q to exit", frame)
        
#         if cv2.waitKey(50) & 0xFF == ord('q'):
#             break

#     vs.stop()
#     cv2.destroyAllWindows()
#     update_attendance_in_db_in(present)
#     return redirect('home')




from django.shortcuts import render
from django.http import JsonResponse
import cv2
import numpy as np
import base64
import dlib
import pickle
from sklearn.preprocessing import LabelEncoder
import time
import datetime
import json
from imutils import face_utils

def index(request):
    return render(request, 'recognition/home.html')  # Assuming you have an index.html
@ensure_csrf_cookie
def capture_in(request):
    return render(request, 'recognition/capture_in.html')
@ensure_csrf_cookie
def capture_out(request):
    return render(request, 'recognition/capture_out.html')

def mark_your_attendance(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        image_data = data['image']
        image_data = image_data.split(',')[1]
        image_data = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')
        svc_save_path = "face_recognition_data/svc.sav"

        with open(svc_save_path, 'rb') as f:
            svc = pickle.load(f)
        
        encoder = LabelEncoder()
        encoder.classes_ = np.load('face_recognition_data/classes.npy')

        faces_encodings = np.zeros((1, 128))
        no_of_faces = len(svc.predict_proba(faces_encodings)[0])
        count = dict()
        present = dict()
        log_time = dict()
        start = dict()
        
        for i in range(no_of_faces):
            count[encoder.inverse_transform([i])[0]] = 0
            present[encoder.inverse_transform([i])[0]] = False

        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_frame, 0)

        for face in faces:
            try:
                x, y, w, h = face_utils.rect_to_bb(face)
                
                # Direct face extraction and resizing
                face_region = img[max(y, 0):min(y+h, img.shape[0]), 
                                  max(x, 0):min(x+w, img.shape[1])]
                if face_region.size == 0:
                    continue
                    
                face_aligned = cv2.resize(face_region, (96, 96))
                
                pred, prob = predict(face_aligned, svc)

                if pred != [-1]:
                    person_name = encoder.inverse_transform(np.ravel([pred]))[0]
                    pred = person_name
                    
                    if count[pred] == 0:
                        start[pred] = time.time()
                        count[pred] = count.get(pred, 0) + 1

                    if count[pred] == 4 and (time.time()-start[pred]) > 1.2:
                        count[pred] = 0
                    else:
                        present[pred] = True
                        log_time[pred] = datetime.datetime.now()
                        count[pred] = count.get(pred, 0) + 1
                        print(pred, present[pred], count[pred])
                else:
                    print("Unknown face detected")
            
            except Exception as e:
                print(f"[WARNING] Face processing error: {str(e)}")
                continue

        update_attendance_in_db_in(present)
        return JsonResponse({'status': 'success', 'message': 'Attendance marked in time'})
    return JsonResponse({'status': 'error', 'message': 'Invalid request'}, status=400)

# def mark_your_attendance_out(request):
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')
#     svc_save_path = "face_recognition_data/svc.sav"

#     with open(svc_save_path, 'rb') as f:
#         svc = pickle.load(f)
    
#     encoder = LabelEncoder()
#     encoder.classes_ = np.load('face_recognition_data/classes.npy')

#     faces_encodings = np.zeros((1, 128))
#     no_of_faces = len(svc.predict_proba(faces_encodings)[0])
#     count = dict()
#     present = dict()
#     log_time = dict()
#     start = dict()
    
#     for i in range(no_of_faces):
#         count[encoder.inverse_transform([i])[0]] = 0
#         present[encoder.inverse_transform([i])[0]] = False

#     vs = VideoStream(src=0).start()
    
#     while True:
#         frame = vs.read()
#         if frame is None:
#             continue
            
#         frame = imutils.resize(frame, width=800)
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detector(gray_frame, 0)

#         for face in faces:
#             try:
#                 x, y, w, h = face_utils.rect_to_bb(face)
                
#                 # Direct face extraction and resizing
#                 face_region = frame[max(y, 0):min(y+h, frame.shape[0]), 
#                                   max(x, 0):min(x+w, frame.shape[1])]
#                 if face_region.size == 0:
#                     continue
                    
#                 face_aligned = cv2.resize(face_region, (96, 96))
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
                
#                 pred, prob = predict(face_aligned, svc)

#                 if pred != [-1]:
#                     person_name = encoder.inverse_transform(np.ravel([pred]))[0]
#                     pred = person_name
                    
#                     if count[pred] == 0:
#                         start[pred] = time.time()
#                         count[pred] = count.get(pred, 0) + 1

#                     if count[pred] == 4 and (time.time()-start[pred]) > 1.5:
#                         count[pred] = 0
#                     else:
#                         present[pred] = True
#                         log_time[pred] = datetime.datetime.now()
#                         count[pred] = count.get(pred, 0) + 1
#                         print(pred, present[pred], count[pred])
                    
#                     cv2.putText(frame, f"{person_name} {prob:.2f}", 
#                               (x+6, y+h-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#                 else:
#                     cv2.putText(frame, "unknown", 
#                               (x+6, y+h-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
#             except Exception as e:
#                 print(f"[WARNING] Face processing error: {str(e)}")
#                 continue

#         cv2.imshow("Mark Attendance - Out - Press q to exit", frame)
        
#         if cv2.waitKey(50) & 0xFF == ord('q'):
#             break

#     vs.stop()
#     cv2.destroyAllWindows()
#     update_attendance_in_db_out(present)
#     return redirect('home')

def mark_your_attendance_out(request):
    if request.method == 'POST':
        # Receive the image from the request body
        data = json.loads(request.body.decode('utf-8'))
        image_data = data['image']
        image_data = image_data.split(',')[1]  # Strip base64 header if included
        image_data = base64.b64decode(image_data)  # Decode the base64 image data
        np_arr = np.frombuffer(image_data, np.uint8)  # Convert the bytes to an array
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Decode into an image

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')
        svc_save_path = "face_recognition_data/svc.sav"

        with open(svc_save_path, 'rb') as f:
            svc = pickle.load(f)
        
        encoder = LabelEncoder()
        encoder.classes_ = np.load('face_recognition_data/classes.npy')

        faces_encodings = np.zeros((1, 128))
        no_of_faces = len(svc.predict_proba(faces_encodings)[0])
        count = dict()
        present = dict()
        log_time = dict()
        start = dict()
        
        for i in range(no_of_faces):
            count[encoder.inverse_transform([i])[0]] = 0
            present[encoder.inverse_transform([i])[0]] = False

        # Resize the image to improve processing time
        frame = imutils.resize(frame, width=800)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_frame, 0)

        for face in faces:
            try:
                x, y, w, h = face_utils.rect_to_bb(face)
                
                # Direct face extraction and resizing
                face_region = frame[max(y, 0):min(y+h, frame.shape[0]), 
                                  max(x, 0):min(x+w, frame.shape[1])]
                if face_region.size == 0:
                    continue
                    
                face_aligned = cv2.resize(face_region, (96, 96))
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
                
                pred, prob = predict(face_aligned, svc)

                if pred != [-1]:
                    person_name = encoder.inverse_transform(np.ravel([pred]))[0]
                    pred = person_name
                    
                    if count[pred] == 0:
                        start[pred] = time.time()
                        count[pred] = count.get(pred, 0) + 1

                    if count[pred] == 4 and (time.time()-start[pred]) > 1.5:
                        count[pred] = 0
                    else:
                        present[pred] = True
                        log_time[pred] = datetime.datetime.now()
                        count[pred] = count.get(pred, 0) + 1
                        print(pred, present[pred], count[pred])
                    
                    cv2.putText(frame, f"{person_name} {prob:.2f}", 
                              (x+6, y+h-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    cv2.putText(frame, "unknown", 
                              (x+6, y+h-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            except Exception as e:
                return JsonResponse({'status': 'error', 'message': 'error matching face'}, status=400)

                

        # For the current request, return the attendance result
        update_attendance_in_db_out(present)
        return JsonResponse({'status': 'success', 'message': 'Attendance marked out time'})
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request'}, status=400)


@login_required
def train(request):
	if request.user.username!='admin':
		return redirect('not-authorised')

	training_dir='face_recognition_data/training_dataset'
	
	
	
	count=0
	for person_name in os.listdir(training_dir):
		curr_directory=os.path.join(training_dir,person_name)
		if not os.path.isdir(curr_directory):
			continue
		for imagefile in image_files_in_folder(curr_directory):
			count+=1

	X=[]
	y=[]
	i=0


	for person_name in os.listdir(training_dir):
		print(str(person_name))
		curr_directory=os.path.join(training_dir,person_name)
		if not os.path.isdir(curr_directory):
			continue
		for imagefile in image_files_in_folder(curr_directory):
			print(str(imagefile))
			image=cv2.imread(imagefile)
			try:
				X.append((face_recognition.face_encodings(image)[0]).tolist())
				

				
				y.append(person_name)
				i+=1
			except:
				print("removed")
				os.remove(imagefile)

			


	targets=np.array(y)
	encoder = LabelEncoder()
	encoder.fit(y)
	y=encoder.transform(y)
	X1=np.array(X)
	print("shape: "+ str(X1.shape))
	np.save('face_recognition_data/classes.npy', encoder.classes_)
	svc = SVC(kernel='linear',probability=True)
	svc.fit(X1,y)
	svc_save_path="face_recognition_data/svc.sav"
	with open(svc_save_path, 'wb') as f:
		pickle.dump(svc,f)

	
	vizualize_Data(X1,targets)
	
	messages.success(request, f'Training Complete.')

	return render(request,"recognition/train.html")


@login_required
def not_authorised(request):
	return render(request,'recognition/not_authorised.html')



@login_required
def view_attendance_home(request):
	total_num_of_emp=total_number_employees()
	emp_present_today=employees_present_today()
	this_week_emp_count_vs_date()
	last_week_emp_count_vs_date()
	return render(request,"recognition/view_attendance_home.html", {'total_num_of_emp' : total_num_of_emp, 'emp_present_today': emp_present_today})


@login_required
def view_attendance_date(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	qs=None
	time_qs=None
	present_qs=None


	if request.method=='POST':
		form=DateForm(request.POST)
		if form.is_valid():
			date=form.cleaned_data.get('date')
			print("date:"+ str(date))
			time_qs=Time.objects.filter(date=date)
			present_qs=Present.objects.filter(date=date)
			if(len(time_qs)>0 or len(present_qs)>0):
				qs=hours_vs_employee_given_date(present_qs,time_qs)


				return render(request,'recognition/view_attendance_date.html', {'form' : form,'qs' : qs })
			else:
				messages.warning(request, f'No records for selected date.')
				return redirect('view-attendance-date')


			
			
			
		


	else:
		

			form=DateForm()
			return render(request,'recognition/view_attendance_date.html', {'form' : form, 'qs' : qs})


@login_required
def view_attendance_employee(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	time_qs=None
	present_qs=None
	qs=None

	if request.method=='POST':
		form=UsernameAndDateForm(request.POST)
		if form.is_valid():
			username=form.cleaned_data.get('username')
			if username_present(username):
				
				u=User.objects.get(username=username)
				
				time_qs=Time.objects.filter(user=u)
				present_qs=Present.objects.filter(user=u)
				date_from=form.cleaned_data.get('date_from')
				date_to=form.cleaned_data.get('date_to')
				
				if date_to < date_from:
					messages.warning(request, f'Invalid date selection.')
					return redirect('view-attendance-employee')
				else:
					

					time_qs=time_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
					present_qs=present_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
					
					if (len(time_qs)>0 or len(present_qs)>0):
						qs=hours_vs_date_given_employee(present_qs,time_qs,admin=True)
						return render(request,'recognition/view_attendance_employee.html', {'form' : form, 'qs' :qs})
					else:
						#print("inside qs is None")
						messages.warning(request, f'No records for selected duration.')
						return redirect('view-attendance-employee')



			
			
				
			else:
				print("invalid username")
				messages.warning(request, f'No such username found.')
				return redirect('view-attendance-employee')


	else:
		

			form=UsernameAndDateForm()
			return render(request,'recognition/view_attendance_employee.html', {'form' : form, 'qs' :qs})




@login_required
def view_my_attendance_employee_login(request):
	if request.user.username=='admin':
		return redirect('not-authorised')
	qs=None
	time_qs=None
	present_qs=None
	if request.method=='POST':
		form=DateForm_2(request.POST)
		if form.is_valid():
			u=request.user
			time_qs=Time.objects.filter(user=u)
			present_qs=Present.objects.filter(user=u)
			date_from=form.cleaned_data.get('date_from')
			date_to=form.cleaned_data.get('date_to')
			if date_to < date_from:
					messages.warning(request, f'Invalid date selection.')
					return redirect('view-my-attendance-employee-login')
			else:
					

					time_qs=time_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
					present_qs=present_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
				
					if (len(time_qs)>0 or len(present_qs)>0):
						qs=hours_vs_date_given_employee(present_qs,time_qs,admin=False)
						return render(request,'recognition/view_my_attendance_employee_login.html', {'form' : form, 'qs' :qs})
					else:
						
						messages.warning(request, f'No records for selected duration.')
						return redirect('view-my-attendance-employee-login')
	else:
		

			form=DateForm_2()
			return render(request,'recognition/view_my_attendance_employee_login.html', {'form' : form, 'qs' :qs})