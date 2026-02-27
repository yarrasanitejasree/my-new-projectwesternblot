# Western Blot Image Analyzer (FastAPI)

This project is a simple FastAPI application that automatically analyzes
Western blot images.

It detects lanes, finds protein bands, calculates molecular weight
(kDa), and estimates band intensity.

------------------------------------------------------------------------

## What This Project Does

1.  Upload a Western blot image.
2.  Detect lanes automatically.
3.  Detect bands inside each lane.
4.  Use ladder lane to calculate molecular weight (kDa).
5.  Calculate band intensity.
6.  Export results as:
    -   Annotated image
    -   CSV file
    -   3D intensity plot

------------------------------------------------------------------------

## How Intensity Is Calculated

For each detected band:

Intensity = Sum of pixel values across the lane width at that band
position.

This means: - Darker band → Higher intensity value - Lighter band →
Lower intensity value

Note: This is relative optical density (image-based), not absolute
protein amount.

------------------------------------------------------------------------

## How Molecular Weight (kDa) Is Calculated

1.  Select ladder (ruler) lane.
2.  Detect ladder band positions.
3.  Map pixel positions to log(kDa).
4.  Convert back to kDa using:

kDa = 10\^(interpolated log value)

------------------------------------------------------------------------

## API Endpoint

POST /analyze

### Parameters:

-   ruler_lane → Index of ladder lane
-   min_kda → Minimum molecular weight
-   max_kda → Maximum molecular weight
-   volume_loaded → Sample loading volume
-   reference_intensity → (Optional)
-   reference_concentration → (Optional)

------------------------------------------------------------------------

## Output Files

-   results/annotated.png → Image with labeled bands
-   results/results.csv → Table of values
-   results/3d_plot.html → Interactive 3D intensity surface

------------------------------------------------------------------------

## Technologies Used

-   FastAPI
-   OpenCV
-   NumPy
-   SciPy
-   Pandas
-   Plotly

------------------------------------------------------------------------

This project is designed for simple automated Western blot analysis
