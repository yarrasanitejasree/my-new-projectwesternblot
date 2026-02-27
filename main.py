# ---------------- IMPORT REQUIRED LIBRARIES ----------------
# FastAPI for API creation
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.staticfiles import StaticFiles

# Image processing
import numpy as np
import cv2

# File handling
import os
import shutil

# Data handling
import pandas as pd

# Signal processing for peak detection
from scipy.signal import find_peaks

# 3D visualization
import plotly.graph_objects as go


# ---------------- INITIALIZE FASTAPI APP ----------------
app = FastAPI()

# Define folders to store uploaded files and results
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Expose results folder as static path
app.mount("/results", StaticFiles(directory="results"), name="results")


# ---------------- MAIN ANALYSIS ENDPOINT ----------------
@app.post("/analyze")
async def analyze_western_blot(
    file: UploadFile = File(...),  # Uploaded blot image
    ruler_lane: int = Query(0),  # Lane index containing ladder
    min_kda: float = Query(10.0),  # Minimum molecular weight
    max_kda: float = Query(200.0),  # Maximum molecular weight
    volume_loaded: float = Query(10.0),  # Sample loading volume
    reference_intensity: float = Query(None),  # Intensity of reference band
    reference_concentration: float = Query(None)  # Known concentration of reference band
):

    # ---------------- SAVE UPLOADED IMAGE ----------------
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ---------------- LOAD AND PREPROCESS IMAGE ----------------
    # Convert to grayscale
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # Normalize intensity to 0–255
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    # Invert image (bands become bright for detection)
    img = 255 - img

    # Apply Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # ---------------- DETECT LANES ----------------
    # Sum pixel intensities vertically to find lane peaks
    vertical_profile = np.sum(img, axis=0)

    lane_peaks, _ = find_peaks(
        vertical_profile,
        distance=50,       # Minimum spacing between lanes
        prominence=1000    # Minimum lane strength
    )

    if len(lane_peaks) == 0:
        return {"error": "No lanes detected"}

    band_data = {}

    # ---------------- DETECT BANDS IN EACH LANE ----------------
    for i, lane_x in enumerate(lane_peaks):

        # Extract narrow vertical region around lane
        left = max(lane_x - 20, 0)
        right = min(lane_x + 20, img.shape[1])
        lane_region = img[:, left:right]

        # Sum horizontally to get band intensity profile
        horizontal_profile = np.sum(lane_region, axis=1)

        # Detect band peaks
        band_peaks, properties = find_peaks(
            horizontal_profile,
            distance=20,     # Minimum band spacing
            prominence=500   # Band strength threshold
        )

        band_data[i] = {
            "positions": band_peaks,
            "intensities": horizontal_profile[band_peaks],
            "concentrations": []
        }

    # ---------------- MOLECULAR WEIGHT CALIBRATION ----------------
    # Use ladder lane for log-scale calibration

    if ruler_lane not in band_data:
        return {"error": "Invalid ruler lane index"}

    ruler_positions = band_data[ruler_lane]["positions"]

    if len(ruler_positions) < 2:
        return {"error": "Not enough ladder bands detected"}

    # Create log scale between max_kda and min_kda
    log_kda_values = np.linspace(
        np.log10(max_kda),
        np.log10(min_kda),
        len(ruler_positions)
    )

    # Function to convert pixel position to kDa
    def pixel_to_kda(pixel):
        return 10 ** np.interp(
            pixel,
            ruler_positions,
            log_kda_values
        )

    # ---------------- BAND QUANTIFICATION ----------------
    results = []

    for lane, data in band_data.items():
        for pos, intensity in zip(data["positions"], data["intensities"]):

            # Calculate molecular weight
            kda_value = pixel_to_kda(pos)

            # Relative quantity (intensity-based scaling)
            relative_quantity = (intensity / 100.0) * volume_loaded

            # Concentration calculation using reference band
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

    # Convert to DataFrame and save CSV
    df = pd.DataFrame(results)
    df = df.sort_values(by=["Lane", "kDa"])
    df.reset_index(drop=True, inplace=True)

    csv_path = os.path.join(RESULT_FOLDER, "results.csv")
    df.to_csv(csv_path, index=False)

    # ---------------- CREATE ANNOTATED IMAGE ----------------
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for lane, data in band_data.items():
        lane_x = lane_peaks[lane]

        # Draw lane center line
        cv2.line(output_img, (lane_x, 0), (lane_x, img.shape[0]), (255, 0, 0), 1)

        for idx, pos in enumerate(data["positions"]):

            intensity = data["intensities"][idx]
            conc = data["concentrations"][idx]
            kda_value = pixel_to_kda(pos)

            # Highlight reference band (if exact intensity match)
            if reference_intensity is not None and intensity == reference_intensity:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red

            cv2.circle(output_img, (lane_x, pos), 6, color, -1)

            # Add label text
            label = f"{round(kda_value,1)} kDa"
            if conc is not None:
                label += f" | {round(conc,2)} ng"

            cv2.putText(output_img,
                        label,
                        (lane_x + 8, pos - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1)

    # Save annotated image
    image_path = os.path.join(RESULT_FOLDER, "annotated.png")
    cv2.imwrite(image_path, output_img)

    # ---------------- GENERATE 3D INTENSITY PLOT ----------------
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

    plot_path = os.path.join(RESULT_FOLDER, "3d_plot.html")
    fig.write_html(plot_path)

    # ---------------- RETURN RESPONSE ----------------
    return {
        "status": "success",
        "lanes_detected": len(lane_peaks),
        "annotated_image": "/results/annotated.png",
        "csv_file": "/results/results.csv",
        "3d_plot": "/results/3d_plot.html",
        "results": results
    }