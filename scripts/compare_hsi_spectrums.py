import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. Load the image or generate a dummy one for demonstration
    if len(sys.argv) < 2:
        print("Usage: python compare_hsi_spectrums.py <path_to_hsi_image>")
        print("Since no image was provided, generating a dummy HSI image for demonstration...")
        # Generate dummy 2D image: spatial x, spectral y
        x = np.linspace(0, 10, 320)
        y = np.linspace(0, 5, 240)
        X, Y = np.meshgrid(x, y)
        img = (np.sin(X*2) + np.cos(Y*3) + 2) * 50
        img += np.random.normal(0, 5, img.shape)  # add noise
        img = img.astype(np.uint8)
    else:
        img_path = sys.argv[1]
        if img_path.endswith('.npy'):
            img = np.load(img_path)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            
        if img is None:
            print(f"Error: Could not load image from {img_path}")
            return
            
        # Convert to grayscale (1 channel) if the loaded image is RGB
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f"Loaded image with shape: {img.shape} (y spectral, x spatial)")
    
    # 2. Spectral Calibration (pixel row to wavelength)
    print("\n--- Spectral Axis Calibration ---")
    print("We need to map the image rows to wavelengths.")
    try:
        row_r = float(input("Enter row pixel index for Red band (620nm): "))
        row_g = float(input("Enter row pixel index for Green band (540nm): "))
        row_b = float(input("Enter row pixel index for Blue band (465nm): "))
    except ValueError:
        print("Invalid input. Using default values for demonstration (R=200, G=150, B=100).")
        row_r, row_g, row_b = 200, 150, 100
        
    # Linearly fit the 3 points to a line (wavelength = m * row + c)
    # We use polyfit of degree 1 (linear regression)
    rows_calib = np.array([row_r, row_g, row_b])
    wls_calib = np.array([620, 540, 465])
    
    coeffs = np.polyfit(rows_calib, wls_calib, 1)
    poly = np.poly1d(coeffs)
    
    # Generate the full wavelength axis mapping for the image rows
    img_rows = np.arange(img.shape[0])
    wavelengths = poly(img_rows)
    
    print(f"\nLinear calibration equation: Wavelength = {coeffs[0]:.4f} * row + {coeffs[1]:.4f}")
    
    # 3. Show the image and ask for two clicks for spatial columns
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.imshow(img, cmap='gray', aspect='auto')
    ax.set_title("Click 2 columns (spatial locations) in the image\n(Middle-click or press Enter to finish early)")
    ax.set_xlabel("Spatial Axis (x)")
    ax.set_ylabel("Spectral Axis (rows)")
    
    print("\nPlease click on two points in the figure window...")
    # ginput pauses and waits for user clicks
    points = plt.ginput(2, timeout=-1)
    plt.close(fig)
    
    if len(points) < 2:
        print("Did not select 2 points. Exiting.")
        return
        
    x1, y1 = points[0]
    x2, y2 = points[1]
    
    col1 = int(round(x1))
    col2 = int(round(x2))
    
    # Ensure columns are within image bounds
    col1 = np.clip(col1, 0, img.shape[1] - 1)
    col2 = np.clip(col2, 0, img.shape[1] - 1)
    
    print(f"Selected spatial column 1: x = {col1}")
    print(f"Selected spatial column 2: x = {col2}")
    
    # Extract the spectrums (columns)
    spectrum_1 = img[:, col1]
    spectrum_2 = img[:, col2]
    
    # 4. Plot the extracted spectrums with wavelength on the x-axis
    plt.figure(figsize=(10, 5))
    plt.plot(wavelengths, spectrum_1, label=f"Spectrum at x={col1}", color='blue')
    plt.plot(wavelengths, spectrum_2, label=f"Spectrum at x={col2}", color='orange')
    
    # Add vertical lines for RGB reference to easily visually see the alignment
    plt.axvline(x=620, color='red', linestyle='--', alpha=0.5, label='Red (620nm)')
    plt.axvline(x=540, color='green', linestyle='--', alpha=0.5, label='Green (540nm)')
    plt.axvline(x=465, color='blue', linestyle='--', alpha=0.5, label='Blue (465nm)')
    
    plt.title("Comparison of Selected Spectrums")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
