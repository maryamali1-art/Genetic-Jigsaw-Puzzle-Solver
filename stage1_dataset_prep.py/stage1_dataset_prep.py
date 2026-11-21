import os
import sys
import cv2
import numpy as np
import json

def slice_custom_image(image_path=r"C:\Users\master\Desktop\stage1_dataset_prep.py\photo_2026-11-14_20-34-30.jpg", output_dir="puzzle_dataset", grid_rows=3, grid_cols=3):
    """
    Slices a custom image into a 3x3 grid (9 pieces) and saves the individual pieces in the Dataset directory.
    """
    # 1. Verify that the uploaded image exists
    if not os.path.exists(image_path):
        print("=" * 60)
        print(f"❌ Error: Image not found at the specified path: '{image_path}'")
        print("=" * 60)
        return

    # 2. Read the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image file '{image_path}'. Make sure the file is valid and in a supported format (JPG/PNG).")
        
    h, w, _ = img.shape
    print(f"[+] Your image was loaded successfully! Original dimensions: {w}x{h} pixels.")

    # 3. Calculate single piece dimensions and crop excess edges to fit the 3x3 grid evenly
    piece_h = h // grid_rows
    piece_w = w // grid_cols
    
    img_cropped = img[:piece_h * grid_rows, :piece_w * grid_cols]
    
    # 4. Create the standalone Dataset directory
    os.makedirs(output_dir, exist_ok=True)
    
    metadata = {
        "original_image": image_path,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "total_pieces": grid_rows * grid_cols,
        "piece_height": piece_h,
        "piece_width": piece_w,
        "pieces": []
    }
    
    # 5. Programmatically slice and store the 9 pieces
    count = 0
    for r in range(grid_rows):
        for c in range(grid_cols):
            y1, y2 = r * piece_h, (r + 1) * piece_h
            x1, x2 = c * piece_w, (c + 1) * piece_w
            
            piece_img = img_cropped[y1:y2, x1:x2]
            piece_filename = f"piece_{count:02d}.png"
            piece_path = os.path.join(output_dir, piece_filename)
            
            cv2.imwrite(piece_path, piece_img)
            
            metadata["pieces"].append({
                "id": count,
                "original_row": r,
                "original_col": c,
                "filename": piece_filename
            })
            count += 1
            
    # 6. Save the metadata and original layout file
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
        
    print(f"[+] Your image was successfully sliced into {count} pieces ({grid_rows}x{grid_cols} grid)!")
    print(f"[+] All pieces have been saved to the directory: '{output_dir}/'")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_image = sys.argv[1]
    else:
        target_image = "photo_2026-11-14_20-34-30.jpg"
        
    slice_custom_image(image_path=target_image)