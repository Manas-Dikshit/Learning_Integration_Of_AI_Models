---
library_name: transformers.js
tags:
- pose-estimation
license: agpl-3.0
---

YOLOv8n-pose with ONNX weights to be compatible with Transformers.js.

## Usage (Transformers.js)

If you haven't already, you can install the [Transformers.js](https://huggingface.co/docs/transformers.js) JavaScript library from [NPM](https://www.npmjs.com/package/@huggingface/transformers) using:
```bash
npm i @huggingface/transformers
```

**Example:** Perform pose-estimation w/ `Xenova/yolov8n-pose`.

```js
import { AutoModel, AutoProcessor, RawImage } from '@huggingface/transformers';

// Load model and processor
const model_id = 'Xenova/yolov8n-pose';
const model = await AutoModel.from_pretrained(model_id);
const processor = await AutoProcessor.from_pretrained(model_id);

// Read image and run processor
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/football-match.jpg';
const image = await RawImage.read(url);
const { pixel_values } = await processor(image);

// Set thresholds
const threshold = 0.3; // Remove detections with low confidence
const iouThreshold = 0.5; // Used to remove duplicates
const pointThreshold = 0.3; // Hide uncertain points

// Predict bounding boxes and keypoints
const { output0 } = await model({ images: pixel_values });

// Post-process:
const permuted = output0[0].transpose(1, 0);
// `permuted` is a Tensor of shape [ 8400, 56 ]:
// - 8400 potential detections
// - 56 parameters for each box:
//   - 4 for the bounding box dimensions (x-center, y-center, width, height)
//   - 1 for the confidence score
//   - 17 * 3 = 51 for the pose keypoints: 17 labels, each with (x, y, visibilitiy)

// Example code to format it nicely:
const results = [];
const [scaledHeight, scaledWidth] = pixel_values.dims.slice(-2);
for (const [xc, yc, w, h, score, ...keypoints] of permuted.tolist()) {
    if (score < threshold) continue;

    // Get pixel values, taking into account the original image size
    const x1 = (xc - w / 2) / scaledWidth * image.width;
    const y1 = (yc - h / 2) / scaledHeight * image.height;
    const x2 = (xc + w / 2) / scaledWidth * image.width;
    const y2 = (yc + h / 2) / scaledHeight * image.height;
    results.push({ x1, x2, y1, y2, score, keypoints })
}


// Define helper functions
function removeDuplicates(detections, iouThreshold) {
    const filteredDetections = [];

    for (const detection of detections) {
        let isDuplicate = false;
        let duplicateIndex = -1;
        let maxIoU = 0;

        for (let i = 0; i < filteredDetections.length; ++i) {
            const filteredDetection = filteredDetections[i];
            const iou = calculateIoU(detection, filteredDetection);
            if (iou > iouThreshold) {
                isDuplicate = true;
                if (iou > maxIoU) {
                    maxIoU = iou;
                    duplicateIndex = i;
                }
            }
        }

        if (!isDuplicate) {
            filteredDetections.push(detection);
        } else if (duplicateIndex !== -1 && detection.score > filteredDetections[duplicateIndex].score) {
            filteredDetections[duplicateIndex] = detection;
        }
    }

    return filteredDetections;
}

function calculateIoU(detection1, detection2) {
    const xOverlap = Math.max(0, Math.min(detection1.x2, detection2.x2) - Math.max(detection1.x1, detection2.x1));
    const yOverlap = Math.max(0, Math.min(detection1.y2, detection2.y2) - Math.max(detection1.y1, detection2.y1));
    const overlapArea = xOverlap * yOverlap;

    const area1 = (detection1.x2 - detection1.x1) * (detection1.y2 - detection1.y1);
    const area2 = (detection2.x2 - detection2.x1) * (detection2.y2 - detection2.y1);
    const unionArea = area1 + area2 - overlapArea;

    return overlapArea / unionArea;
}

const filteredResults = removeDuplicates(results, iouThreshold);

// Display results
for (const { x1, x2, y1, y2, score, keypoints } of filteredResults) {
    console.log(`Found person at [${x1}, ${y1}, ${x2}, ${y2}] with score ${score.toFixed(3)}`)
    for (let i = 0; i < keypoints.length; i += 3) {
        const label = model.config.id2label[Math.floor(i / 3)];
        const [x, y, point_score] = keypoints.slice(i, i + 3);
        if (point_score < pointThreshold) continue;
        console.log(`  - ${label}: (${x.toFixed(2)}, ${y.toFixed(2)}) with score ${point_score.toFixed(3)}`);
    }
}
```

<details>

<summary>See example output</summary>

```
Found person at [536.1322975158691, 37.87850737571716, 645.2879905700684, 286.9420547962189] with score 0.791
  - nose: (445.81, 87.11) with score 0.936
  - left_eye: (450.90, 80.87) with score 0.976
  - right_eye: (439.37, 81.31) with score 0.664
  - left_ear: (460.76, 81.94) with score 0.945
  - left_shoulder: (478.06, 126.18) with score 0.993
  - right_shoulder: (420.69, 125.17) with score 0.469
  - left_elbow: (496.96, 178.36) with score 0.976
  - left_wrist: (509.41, 232.75) with score 0.892
  - left_hip: (469.15, 215.80) with score 0.980
  - right_hip: (433.73, 218.39) with score 0.794
  - left_knee: (471.45, 278.44) with score 0.969
  - right_knee: (439.23, 281.77) with score 0.701
  - left_ankle: (474.88, 345.49) with score 0.913
  - right_ankle: (441.99, 339.82) with score 0.664
Found person at [-0.15300750732421875, 59.96129276752472, 158.73897552490234, 369.92224643230435] with score 0.863
  - nose: (57.30, 95.37) with score 0.960
  - left_eye: (63.85, 89.48) with score 0.889
  - right_eye: (53.59, 91.60) with score 0.909
  - left_ear: (73.54, 92.67) with score 0.626
  - right_ear: (50.12, 95.95) with score 0.674
  - left_shoulder: (87.62, 132.72) with score 0.965
  - right_shoulder: (39.72, 136.82) with score 0.986
  - left_elbow: (108.17, 186.58) with score 0.857
  - right_elbow: (21.47, 184.66) with score 0.951
  - left_wrist: (113.36, 244.21) with score 0.822
  - right_wrist: (8.04, 240.50) with score 0.915
  - left_hip: (83.47, 234.43) with score 0.990
  - right_hip: (47.29, 237.45) with score 0.994
  - left_knee: (92.12, 324.78) with score 0.985
  - right_knee: (50.70, 325.75) with score 0.991
  - left_ankle: (101.13, 410.45) with score 0.933
  - right_ankle: (49.62, 410.14) with score 0.954
Found person at [104.13589477539062, 20.16922025680542, 505.84068298339844, 522.6950127601624] with score 0.770
  - nose: (132.51, 99.38) with score 0.693
  - left_eye: (138.68, 89.00) with score 0.451
  - left_ear: (145.60, 85.21) with score 0.766
  - left_shoulder: (188.92, 133.25) with score 0.996
  - right_shoulder: (163.12, 158.90) with score 0.985
  - left_elbow: (263.01, 205.18) with score 0.991
  - right_elbow: (181.52, 249.12) with score 0.949
  - left_wrist: (315.65, 259.88) with score 0.964
  - right_wrist: (125.19, 275.10) with score 0.891
  - left_hip: (279.47, 294.29) with score 0.998
  - right_hip: (266.84, 309.38) with score 0.997
  - left_knee: (261.67, 416.57) with score 0.989
  - right_knee: (256.66, 428.75) with score 0.982
  - left_ankle: (322.92, 454.74) with score 0.805
  - right_ankle: (339.15, 459.64) with score 0.780
Found person at [423.3617973327637, 72.75799512863159, 638.2988166809082, 513.1156357765198] with score 0.903
  - nose: (417.19, 137.27) with score 0.992
  - left_eye: (429.74, 127.59) with score 0.975
  - right_eye: (409.83, 129.06) with score 0.961
  - left_ear: (445.81, 133.82) with score 0.847
  - right_ear: (399.09, 132.99) with score 0.711
  - left_shoulder: (451.43, 195.71) with score 0.997
  - right_shoulder: (372.58, 196.25) with score 0.995
  - left_elbow: (463.89, 286.56) with score 0.991
  - right_elbow: (351.35, 260.40) with score 0.978
  - left_wrist: (488.70, 367.36) with score 0.986
  - right_wrist: (395.69, 272.20) with score 0.973
  - left_hip: (435.84, 345.96) with score 0.999
  - right_hip: (380.21, 355.38) with score 0.999
  - left_knee: (454.88, 456.63) with score 0.994
  - right_knee: (395.82, 478.67) with score 0.992
  - left_ankle: (453.75, 556.37) with score 0.889
  - right_ankle: (402.35, 582.09) with score 0.872
```
</details>