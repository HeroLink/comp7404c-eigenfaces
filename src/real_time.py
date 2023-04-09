import cv2
from sklearn.datasets import fetch_lfw_people
from sklearn.discriminant_analysis import StandardScaler
from sklearn.svm import SVC

from pca import *

# Only use people with more than 70 images, default resize=0.5
lfw_people = fetch_lfw_people(
    data_home="../data",
    min_faces_per_person=70, 
    resize=1)
n_samples, img_h, img_w = lfw_people.images.shape

# Use the 1-D pixels as features, positions info is ignored
X = lfw_people.data
n_features = X.shape[1]

# The label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("#Samples:", n_samples)
print("#Features Dimention:", n_features)
print("#Classes:", n_classes)

# X: (N x d), N: number of samples, d: feature dimention
# 125, 94
print("X shape (N, d):", X.shape)
print(f"Image h, w: {img_h}, {img_w}")

k = 150
print(f"Extracting the top {k} eigenfaces from {len(X)} faces")

scaler = StandardScaler()
X = scaler.fit_transform(X)

mean = mean_face(X)
eigface, weights = PCA(X, mean, k)

print("Projecting the input data on the eigenfaces orthonormal basis")
X_train_pca = weights

clf = SVC(kernel='linear', probability=True)
clf.fit(X_train_pca, y)

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')

# Start the video capture
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(img_w*2, img_h*2))

    face_data = []
    # Apply PCA to the face regions
    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = gray[y:y+h, x:x+w]
        # Resize the face region to the same size as the training data
        face_roi = cv2.resize(face_roi, (img_w, img_h))
        # Normalize the face region
        face_roi = (face_roi - face_roi.mean()) / face_roi.std()
        # cv2.imshow("face", face_roi)
        face_data.append(face_roi.flatten())
    
    if len(face_data) > 0:
        face_data = np.array(face_data)
        # face_data = scaler.transform(face_data)
        # Project the face region onto the eigenfaces
        face_data_pca = project(face_data, mean, eigface)
        probas = clf.predict_proba(face_data_pca)
        labels = clf.predict(face_data_pca)

        titles = []
        for i in range(len(labels)):
            if probas[i][labels[i]] < 0.5:
                titles.append("Unknown")
            else:
                titles.append(target_names[labels[i]])

        # Draw a rectangle around each face
        i = 0
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
            cv2.putText(frame, titles[i], (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 4)
            i += 1

    # Display the resulting frame
    cv2.imshow('Face Recognition Using Eigenfaces', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()

# Close all windows
cv2.destroyAllWindows()