# End-to-End CV Project: Face Recognition

> **Building a Secure Access System** - FaceNet and ArcFace

---

## 🎯 The Goal

Build a system that recognizes employees from a webcam feed.
**Stack:** OpenCV (Detection), FaceNet (Embedding), Vector DB (Matching).

---

## 🧠 The Concept: Metric Learning

Standard Classification (Softmax) doesn't work for Face Recognition because we can't retrain the model every time a new employee joins.

**Solution:** Map faces to a 128-dimensional vector space.
- **Same Person:** Distance is small ($< 0.6$).
- **Different People:** Distance is large ($> 1.0$).

**Triplet Loss:**
$$ L = \max(d(A, P) - d(A, N) + \alpha, 0) $$
- **Anchor (A):** Face of Person X.
- **Positive (P):** Another face of Person X.
- **Negative (N):** Face of Person Y.
- **Goal:** Pull P close to A, push N away from A.

---

## 💻 Implementation (FaceNet + MTCNN)

```python
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import cv2

# 1. Load Models
# MTCNN: Detects face and crops it
mtcnn = MTCNN(keep_all=False, device='cuda') 
# InceptionResnet: Converts face to 128-d vector
resnet = InceptionResnetV1(pretrained='vggface2').eval().to('cuda')

# 2. Database (Simulated)
database = {}
def add_to_db(name, image_path):
    img = Image.open(image_path)
    img_cropped = mtcnn(img) # Returns tensor
    if img_cropped is not None:
        embedding = resnet(img_cropped.unsqueeze(0).to('cuda'))
        database[name] = embedding.detach()

add_to_db("Elon", "elon.jpg")

# 3. Real-time Recognition
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Detect
    boxes, _ = mtcnn.detect(img)
    
    if boxes is not None:
        for box in boxes:
            # Crop & Embed
            face = mtcnn.extract(img, [box], save_path=None)
            emb = resnet(face.to('cuda'))
            
            # Match
            min_dist = 100
            identity = "Unknown"
            
            for name, db_emb in database.items():
                dist = (emb - db_emb).norm().item()
                if dist < min_dist:
                    min_dist = dist
                    identity = name
            
            # Threshold
            if min_dist > 0.6:
                identity = "Unknown"
                
            # Draw
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 2)
            cv2.putText(frame, f"{identity} ({min_dist:.2f})", (int(box[0]), int(box[1]-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        
    cv2.imshow('Face Rec', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 🎓 Interview Focus

1.  **Why use MTCNN?**
    - It aligns the face (rotates it so eyes are horizontal) before feeding it to the recognition model. Alignment drastically improves accuracy.

2.  **ArcFace vs Triplet Loss?**
    - **Triplet Loss:** Hard to mine good triplets (training is slow).
    - **ArcFace:** Uses an angular margin penalty in the Softmax loss. Converges faster and produces better separation.

---

**Face Recognition: Your face is your password!**
