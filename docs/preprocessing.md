# Preprocessing

Implemented in `src/mars_tb/data/preprocessing.py` and `augmentation.py`, in the manuscript's order (Table 2).

| Step | Operation | Parameters | Where |
|---|---|---|---|
| 1-2 | decode; scale to float [0,1] | 8-bit /255, 16-bit /65535; DICOM by `2^BitsStored-1`, MONOCHROME1 inverted | `read_image`, `read_dicom` |
| 3 | quality filter | reject shorter edge < 256 px or aspect < 0.5 or > 2.0 | `quality_check` |
| 4 | CLAHE on **uint8** | clipLimit 2.0, tiles 8x8 | `apply_clahe` |
| 5 | resize | 512x512, `cv2.INTER_CUBIC`, no padding, aspect not preserved | `resize_bicubic` |
| 6 | grey -> 3 channels | repeat | `preprocess_array` |
| 7 | ImageNet normalisation | mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225) | `normalize` |
| 8a-d | training only | hflip p 0.5; rotation +/-15 deg bicubic fill 0; brightness/contrast 0.2; random erasing p 0.1, scale (0.02, 0.2), ratio (0.3, 3.3), channel-mean fill | `TrainTransform` |

* **No augmentation outside training.** `build_transform` maps `val|test|external` to a deterministic
  normalise-only transform and raises on unknown split names (tested).
* **Order inside the training transform:** flip -> rotate -> jitter on the [0,1] tensor, then normalise, then erase,
  so the erasing fill value 0 is exactly the ImageNet channel mean.
* **Deviations, with reasons:** (1) DICOM scaling by `BitsStored` instead of a blanket /65535, which would
  leave 12-bit data in 6 % of the range before CLAHE; (2) rotation implemented with `grid_sample` because
  torchvision's tensor rotation has no bicubic mode. Table 2's "DICOM -> 16-bit PNG" intermediate is not
  reproduced; pixel values are used directly. No patient metadata ever leaves `read_dicom`.
* The image is resized to `backbone.img_size`; unit tests use 256 px, the manuscript configuration is 512.
