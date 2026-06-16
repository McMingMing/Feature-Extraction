import sys
import os
import spectrumDataImaging
import sCurveTesting

TARGET_FILE = '/Users/minhnguyen/try-scipy/Brillouin Point Data/20230420 Point 1 Brillouin.tdms'

def main():
    file_name = TARGET_FILE.split('/')[-1]
    print(f"\nProcessing TDMS Pipeline for:\n{file_name}\n")
    
    # 1. Run the 2D Viewer Module
    print("Generating 2D Spatial Overview...")
    viewer_data = spectrumDataImaging.run_viewer(TARGET_FILE)
    
    # 2. Run the 1D Analyzer Module
    print("Extracting 1D Depth Profile and filtering outliers...")
    analyzer_data = sCurveTesting.run_analyzer(TARGET_FILE)

    # 3. Consolidated Master Terminal Output
    print("\n" + "═"*55)
    print(f" BRILLOUIN SUMMARY: {file_name}")
    print("═"*55)
    
    print(" [ SPATIAL OVERVIEW ]")
    print(f"  Cube Dimensions        : {viewer_data['cube_shape']}")
    print(f"  2D Peak Separation     : {viewer_data['sep']:3d} px")
    print("-" * 55)
    
    print(" [ STIFFNESS ANALYSIS ]")
    print(f"  Raw Plateau Mean       : {analyzer_data['raw_mean']:.4f} px")
    print(f"  Clean Plateau Mean     : {analyzer_data['clean_mean']:.4f} px")
    print("-" * 55)
    
    print(" [ MACHINE LEARNING FEATURES ]")
    print(f"  Cornea Thickness Frame : Frame {analyzer_data['edge_frame']}")
    print(f"  Tissue Uniformity (Std): {analyzer_data['plateau_std']:.4f} px")
    print(f"  Transition Steepness   : {analyzer_data['steepness']:.4f}")
    print(f"  Signal Noise / Outliers: {analyzer_data['outliers_removed']} points rejected")
    
    print("═"*55 + "\n")

if __name__ == '__main__':
    main()