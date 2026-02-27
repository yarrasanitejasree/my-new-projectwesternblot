"""
Western Blot Quantification API
--------------------------------
This FastAPI backend performs:

• Lane detection
• Band detection
• Molecular weight calibration (log scale)
• Band quantification
• Annotated image generation
• 3D intensity visualization
• CSV export of results

Author: Your Name
"""

from fastapi import FastAPI, File, UploadFile, Query
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import os
import shutil
import pandas as pd
from scipy.signal import find_peaks
import plotly.graph_objects as go

# -------------------------------------------
# Initialize FastAPI App
# -------------------------------------------
app = FastAPI(title="Western Blot Analyzer")

# Folders for uploads and results
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Allow browser access to result files
app.mount("/results", StaticFiles(directory="results"), name="results")


# -------------------------------------------
# Main API Endpoint
# -------------------------------------------
@app.post("/analyze")
async def analyze_western_blot(
    file: UploadFile = File(...),
    ruler_lane: int = Query(0),
    min_kda: float = Query(10.0),
    max_kda: float = Query(200.0),
    volume_loaded: float = Query(10.0),
    reference_intensity: float = Query(None),
    reference_concentration: float = Query(None)
):

    # -------------------------------------------
    # 1. Save Uploaded Image
    # -------------------------------------------
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # -------------------------------------------
    # 2. Image Preprocessing
    # -------------------------------------------
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # Normalize contrast
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    # Invert (bands become bright)
    img = 255 - img

    # Reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # -------------------------------------------
    # 3. Lane Detection (Vertical Projection)
    # -------------------------------------------
    vertical_profile = np.sum(img, axis=0)

    lane_peaks, _ = find_peaks(
        vertical_profile,
        distance=50,
        prominence=1000
    )

    if len(lane_peaks) == 0:
        return {"error": "No lanes detected"}

    band_data = {}

    # -------------------------------------------
    # 4. Band Detection per Lane
    # -------------------------------------------
    for i, lane_x in enumerate(lane_peaks):

        left = max(lane_x - 20, 0)
        right = min(lane_x + 20, img.shape[1])

        lane_region = img[:, left:right]

        horizontal_profile = np.sum(lane_region, axis=1)

        band_peaks, properties = find_peaks(
            horizontal_profile,
            distance=20,
            prominence=500
        )

        band_data[i] = {
            "positions": band_peaks,
            "intensities": horizontal_profile[band_peaks],
            "concentrations": []
        }

    # -------------------------------------------
    # 5. Molecular Weight Calibration (Log Scale)
    # -------------------------------------------
    if ruler_lane not in band_data:
        return {"error": "Invalid ruler lane index"}

    ruler_positions = band_data[ruler_lane]["positions"]

    if len(ruler_positions) < 2:
        return {"error": "Not enough ladder bands detected"}

    # Generate log10 kDa values
    log_kda_values = np.linspace(
        np.log10(max_kda),
        np.log10(min_kda),
        len(ruler_positions)
    )

    # Convert pixel position → kDa
    def pixel_to_kda(pixel):
        return 10 ** np.interp(
            pixel,
            ruler_positions,
            log_kda_values
        )

    # -------------------------------------------
    # 6. Band Quantification
    # -------------------------------------------
    results = []

    for lane, data in band_data.items():
        for pos, intensity in zip(
            data["positions"],
            data["intensities"]
        ):

            kda_value = pixel_to_kda(pos)

            # Relative Quantity Calculation
            relative_quantity = (
                intensity / 100.0
            ) * volume_loaded

            # Reference-Based Quantification
            if reference_intensity is not None and reference_concentration is not None:
                calculated_concentration = (
                    intensity / reference_intensity
                ) * reference_concentration
            else:
                calculated_concentration = None

            band_data[lane]["concentrations"].append(calculated_concentration)

            results.append({
                "Lane": lane,
                "kDa": round(float(kda_value), 2),
                "Intensity": float(intensity),
                "Relative Quantity": round(float(relative_quantity), 3),
                "Calculated Concentration (ng)": (
                    round(float(calculated_concentration), 3)
                    if calculated_concentration is not None else None
                )
            })

    # Save results to CSV
    df = pd.DataFrame(results)
    df = df.sort_values(by=["Lane", "kDa"])
    df.reset_index(drop=True, inplace=True)
    df.to_csv(os.path.join(RESULT_FOLDER, "results.csv"), index=False)

    # -------------------------------------------
    # 7. Annotated Image Output
    # -------------------------------------------
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for lane, data in band_data.items():
        lane_x = lane_peaks[lane]

        cv2.line(output_img, (lane_x, 0),
                 (lane_x, img.shape[0]), (255, 0, 0), 1)

        for idx, pos in enumerate(data["positions"]):
            kda_value = pixel_to_kda(pos)

            cv2.circle(output_img,
                       (lane_x, pos),
                       6,
                       (0, 0, 255),
                       -1)

            cv2.putText(output_img,
                        f"{round(kda_value,1)} kDa",
                        (lane_x + 8, pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1)

    cv2.imwrite(os.path.join(RESULT_FOLDER, "annotated.png"), output_img)

    # -------------------------------------------
    # 8. 3D Intensity Surface Plot
    # -------------------------------------------
    small_img = cv2.resize(img, (300, 300))

    x = np.arange(small_img.shape[1])
    y = np.arange(small_img.shape[0])
    x, y = np.meshgrid(x, y)

    fig = go.Figure(data=[go.Surface(z=small_img, x=x, y=y)])

    fig.update_layout(
        title="Western Blot Quantitative 3D Intensity Analysis",
        scene=dict(
            xaxis_title="Lane Width",
            yaxis_title="Migration Distance",
            zaxis_title="Intensity"
        )
    )

    fig.write_html(os.path.join(RESULT_FOLDER, "3d_plot.html"))

    # -------------------------------------------
    # 9. API Response
    # -------------------------------------------
    return {
        "status": "success",
        "lanes_detected": len(lane_peaks),
        "annotated_image": "/results/annotated.png",
        "csv_file": "/results/results.csv",
        "3d_plot": "/results/3d_plot.html",
        "results": results
    }