# Calibration Targets

This guide covers calibration target selection, preparation, and usage.

## Recommended Target: ChArUco Board

ChArUco boards combine the robustness of ArUco markers with the accuracy of chessboard corners.

**Recommended configuration:**
- 6×9 squares
- DICT_6X6_250 ArUco dictionary
- 30mm square size (for A4 paper)
- 22.5mm marker size (75% of square)

## Generating a ChArUco Board

### Using OpenCV (Python)

```python
import cv2
from cv2 import aruco

# Create board
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
board = aruco.CharucoBoard((9, 6), 0.030, 0.0225, dictionary)

# Generate high-resolution image (10 px/mm)
img = board.generateImage((2700, 1800))

# Save
cv2.imwrite('charuco_6x9.png', img)
```

### Using Online Tools

- [calib.io Pattern Generator](https://calib.io/pages/camera-calibration-pattern-generator)

## Printing Guidelines

1. **Print at 100% scale** - Do not allow resizing
2. **Use matte paper** - Avoids reflections
3. **Measure after printing** - Verify dimensions match configuration
4. **Mount on rigid backing** - Foam board or aluminum composite

## Updating Dimensions

After printing, measure actual square size and update `assets/calibration_targets/dimensions.yaml`:

```yaml
target:
  square_size_mm: 29.85  # Measured value
  marker_size_mm: 22.39  # Measured value
```

**A 1% dimension error causes 1% distance error.**

## Target Placement for Calibration

For best results:

1. **Cover the full field of view** - Target should fill the frame
2. **Use varied poses** - Rotate and tilt the target
3. **Ensure visibility in both cameras** - HSI and RGB must see the target
4. **Maintain focus** - Target should be sharp in all images
5. **Minimize motion blur** - Important for the slow-motion assumption

## Number of Views

| Requirement | Minimum Views | Recommended Views |
|-------------|---------------|-------------------|
| Basic calibration | 10 | 20 |
| High accuracy | 20 | 50+ |
| With validation | 30 | 50+ (split for validation) |
