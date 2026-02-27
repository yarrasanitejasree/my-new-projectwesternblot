# 🧪 Western Blot Automated Quantification API

## 📌 Overview

This FastAPI-based application performs automated Western blot image
analysis including:

-   Image preprocessing
-   Lane detection
-   Band detection
-   Molecular weight calibration (log scale interpolation)
-   Band intensity quantification
-   Optional reference-based concentration calculation
-   Annotated image generation
-   CSV export of results
-   3D intensity visualization (Plotly)

------------------------------------------------------------------------

## ⚙️ Workflow

### 1️⃣ Image Preprocessing

-   Convert image to grayscale
-   Normalize pixel values (0--255)
-   Invert image (bands become bright)
-   Apply Gaussian blur to reduce noise

### 2️⃣ Lane Detection

-   Sum pixel intensities vertically (column-wise)
-   Detect peaks in vertical intensity profile
-   Each peak represents a lane

### 3️⃣ Band Detection

-   Crop lane region
-   Sum pixel values horizontally (row-wise)
-   Detect peaks in horizontal profile
-   Each peak corresponds to a protein band

### 📊 Band Intensity Calculation

    Intensity = Sum of pixel values across lane width at band position

This represents **line-integrated optical density (1D integration)**.

------------------------------------------------------------------------

## 🧬 Molecular Weight Calibration

Using a selected ruler (ladder) lane:

1.  Detect ladder band positions
2.  Map pixel positions to log10(kDa) values
3.  Interpolate using log scale
4.  Convert back to kDa

```{=html}
<!-- -->
```
    kDa = 10^(interpolated_log_value)

------------------------------------------------------------------------

## 📈 Quantification

### Relative Quantity

    Relative Quantity = (Band Intensity / 100) × Volume Loaded

### Reference-Based Concentration (Optional)

If reference band is provided:

    Calculated Concentration = 
    (Band Intensity / Reference Intensity) × Reference Concentration

------------------------------------------------------------------------

## 📂 Generated Outputs

-   `/results/annotated.png` → Image with labeled bands
-   `/results/results.csv` → Quantification table
-   `/results/3d_plot.html` → Interactive 3D intensity plot

------------------------------------------------------------------------

## 🔌 API Endpoint

### `POST /analyze`

### Query Parameters

  Parameter                 Description
  ------------------------- -------------------------------------------
  ruler_lane                Index of ladder lane
  min_kda                   Minimum molecular weight
  max_kda                   Maximum molecular weight
  volume_loaded             Sample loading volume
  reference_intensity       Known reference band intensity (optional)
  reference_concentration   Known reference concentration (optional)

------------------------------------------------------------------------

## 🛠 Tech Stack

-   FastAPI
-   OpenCV
-   NumPy
-   SciPy (find_peaks)
-   Pandas
-   Plotly

------------------------------------------------------------------------


