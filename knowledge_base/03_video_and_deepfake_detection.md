# Video and face-manipulation detection

## FaceForensics++ and the standard video-deepfake benchmark

Rössler, Cozzolino, Verdoliva, Riess, Thies, and Nießner introduced
FaceForensics++ (ICCV 2019), one of the most widely used benchmarks for
facial-manipulation detection. It contains real videos alongside four
kinds of manipulations (DeepFakes, Face2Face, FaceSwap, and NeuralTextures)
at multiple compression levels, specifically so researchers could study
how video compression degrades detection accuracy — a very realistic
concern, since nearly all video shared online is recompressed by the
hosting platform. Models benchmarked on FaceForensics++ are typically
evaluated per-frame or with a temporal model that looks across frames for
inconsistencies (e.g. unnatural blinking, boundary artifacts around a
swapped face, or flickering).

Source: Rössler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., &
Nießner, M. (2019). FaceForensics++: Learning to Detect Manipulated Facial
Images. *ICCV 2019*. https://arxiv.org/abs/1901.08971

## Why frame-level analysis is not the same as video-level detection

Tolosana, Vera-Rodriguez, Fierrez, Morales, and Ortega-Garcia's survey,
"DeepFakes and Beyond: A Survey of Face Manipulation and Fake Detection"
(Information Fusion, 2020), lays out the landscape of face-manipulation
techniques and detection approaches, and distinguishes between:

- **Frame-level (spatial) detectors**: classify each frame independently,
  exactly like a still-image detector. They are simple and fast but
  cannot use temporal cues (motion, consistency across frames) at all.
- **Video-level (temporal) detectors**: explicitly model sequences of
  frames (e.g. with a recurrent network or 3D convolutions) to catch
  inconsistencies that only appear across time, such as unnatural blink
  patterns or flickering at manipulation boundaries.

VerityAI's video-analysis feature is a frame-level analysis only: it runs
the same still-image detector independently on a handful of evenly-spaced
frames and reports a simple average. It does **not** implement or claim
any temporal modeling. Per the distinction above, this means VerityAI's
video feature cannot detect manipulation signals that only show up as
*inconsistency across frames* — it can only flag that individual sampled
frames look visually similar to the generated images its image detector
was trained on. A video made of frames that each look "real" in isolation
but are stitched together in a temporally inconsistent way would not be
reliably caught by this method. This is why the application labels video
analysis as experimental and separates it clearly from the image-analysis
feature.

Source: Tolosana, R., Vera-Rodriguez, R., Fierrez, J., Morales, A., &
Ortega-Garcia, J. (2020). DeepFakes and Beyond: A Survey of Face
Manipulation and Fake Detection. *Information Fusion*. https://arxiv.org/abs/2001.00179

## Frame-score variance is not proof of manipulation

A natural temptation is to treat *inconsistent* scores across a video's
sampled frames (e.g. some frames score 90% AI, others score 10%) as
evidence of manipulation. The surveys above do not support this inference:
ordinary variation in lighting, motion blur, compression artifacts per
frame, and the detector's own noise floor can all produce frame-to-frame
score variance in genuinely unmanipulated video. Variance across frames
is, at most, a prompt to look closer — never conclusive evidence on its
own — which is why VerityAI's generated reports never state that
inconsistent frame scores "prove" a video is fake.
