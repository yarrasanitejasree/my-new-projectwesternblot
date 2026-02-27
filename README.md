# 🧪 Western Blot Automated Quantification API

## 📌 Overview
This FastAPI application performs automated Western blot image analysis, including:

- Image preprocessing
- Lane detection
- Band detection
- Molecular weight calibration (log scale interpolation)
- Band intensity quantification
- Optional reference-based concentration calculation
- Annotated image generation
- CSV export of results
- 3D intensity visualization (Plotly)

---

## ⚙️ Workflow

```text
    +------------------+
    |  Upload Image    |
    +--------+---------+
             |
             v
    +------------------+
    | Preprocessing    |
    | (Grayscale,      |
    |  Normalize, Blur)|
    +--------+---------+
             |
             v
    +------------------+
    | Lane Detection   |
    | (Column Sum,     |
    |  Peak Detection) |
    +--------+---------+
             |
             v
    +------------------+
    | Band Detection   |
    | (Row Sum, Peaks) |
    +--------+---------+
             |
             v
    +------------------+
    | Molecular Weight |
    | Calibration      |
    +--------+---------+
             |
             v
    +------------------+
    | Quantification   |
    | (Relative &      |
    | Reference-Based) |
    +--------+---------+
             |
             v
    +------------------+
    | Outputs          |
    | (Annotated Image,|
    | CSV, 3D Plot)    |
    +------------------+
1️⃣ Image Preprocessing

Convert image to grayscale

Normalize pixel values (0–255)

Invert image (bands become bright)

Apply Gaussian blur to reduce noise

2️⃣ Lane Detection

Sum pixel intensities vertically (column-wise)

Detect peaks in vertical intensity profile

Each peak represents a lane

3️⃣ Band Detection

Crop lane region

Sum pixel values horizontally (row-wise)

Detect peaks in horizontal profile

Each peak corresponds to a protein band

📊 Band Intensity Calculation

Intensity = Sum of pixel values across lane width at band position

🧬 Molecular Weight Calibration

Using a selected ruler (ladder) lane:

Detect ladder band positions

Map pixel positions to log10(kDa) values

Interpolate using log scale

Convert back to kDa:
kDa = 10^(interpolated_log_value)
📈 Quantification

Relative Quantity:
Relative Quantity = (Band Intensity / 100) × Volume Loaded
Reference-Based Concentration (Optional):
Calculated Concentration = (Band Intensity / Reference Intensity) × Reference Concentration
📂 Generated Outputs
| Output                 | Description                   |
| ---------------------- | ----------------------------- |
| /results/annotated.png | Image with labeled bands      |
| /results/results.csv   | Quantification table          |
| /results/3d_plot.html  | Interactive 3D intensity plot |
🔌 API Endpoint

POST /analyze

Query Parameters
| Parameter               | Description                               |
| ----------------------- | ----------------------------------------- |
| ruler_lane              | Index of ladder lane                      |
| min_kda                 | Minimum molecular weight                  |
| max_kda                 | Maximum molecular weight                  |
| volume_loaded           | Sample loading volume                     |
| reference_intensity     | Known reference band intensity (optional) |
| reference_concentration | Known reference concentration (optional)  |
🛠 Tech Stack

FastAPI

OpenCV

NumPy

SciPy (find_peaks)

Pandas

Plotly
💻 Installation
# Clone the repository
git clone https://github.com/yarrasanitejasree/western-blot-analyzer-1.git
cd western-blot-analyzer-1

# Create a virtual environment (recommended)
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# If requirements.txt doesn’t exist
pip install fastapi uvicorn opencv-python numpy scipy pandas plotly
▶️ How to Run
# Start FastAPI server
uvicorn main:app --reload

---

This version will render **headings, lists, tables, and code blocks** properly on GitHub.  

---

If you want, I can also **make the Flowchart as an actual GitHub-friendly ASCII diagram or Mermaid diagram** so it looks even nicer in preview.  

Do you want me to do that?
