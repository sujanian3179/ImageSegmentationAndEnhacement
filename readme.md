# group-photo-portrait-crops (Classical CV, 2022/23-realistic)

This repo creates "contact-style" portrait crops (face + upper body) from a single group photo,
using classical computer vision (no modern foundation vision models).

It provides 2 cropping approaches + 1 enhancement step:

## Approach 1: Haar Face + Haar Upperbody
- Detect faces (Haar cascade)
- Detect upper bodies (Haar cascade)
- Match each face to the upper body with max overlap
- Crop using union(face, upperbody) + padding
- Fallback: face expansion heuristics if upperbody not found

Script:
- `src/crop_haar_face_upperbody.py`

## Approach 2: HOG Person + Haar Face
- Detect people with HOG+SVM (OpenCV default people detector)
- Detect faces with Haar
- Assign each face to the smallest person box that contains face center
- Crop top portion of person box
- Fallback: face expansion heuristics if person not found

Script:
- `src/crop_hog_person_then_face.py`

## Enhancement step (Classical)
- Adaptive gamma correction
- CLAHE on luminance
- Gray-world white balance
- Edge-preserving smoothing
- Edge-only sharpening (Canny mask + unsharp)
- Optional 2x bicubic upscaling

Script:
- `src/enhance_crops.py`

---

## Setup

### 1) Install dependencies
```bash
pip install -r requirements.txt
