import os
from pathlib import Path

# --- CONFIGURATION ---
IMAGE_FOLDER = r'C:\Users\kg060\Desktop\tatkal\cnn-captcha-solver\captcha_images'

def rename_images_sequential():
    """Rename all images to 1.png, 2.png, 3.png, etc."""
    
    # Validate folder
    if not os.path.exists(IMAGE_FOLDER):
        print(f"❌ Folder not found: {IMAGE_FOLDER}")
        return
    
    # Get all image files
    files = [f for f in os.listdir(IMAGE_FOLDER) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    if not files:
        print(f"❌ No images found in {IMAGE_FOLDER}")
        return
    
    # Sort files to maintain consistent order
    files.sort()
    
    print(f"📂 Found {len(files)} images")
    print(f"📁 Folder: {IMAGE_FOLDER}")
    print("="*50)
    
    # Create temporary names first to avoid conflicts
    temp_renames = []
    
    # Step 1: Rename to temporary names
    print("Step 1: Creating temporary names...")
    for i, filename in enumerate(files, 1):
        old_path = os.path.join(IMAGE_FOLDER, filename)
        extension = Path(filename).suffix
        temp_name = f"temp_{i:06d}{extension}"
        temp_path = os.path.join(IMAGE_FOLDER, temp_name)
        
        try:
            os.rename(old_path, temp_path)
            temp_renames.append((temp_path, i, extension))
        except OSError as e:
            print(f"❌ Failed to rename {filename}: {e}")
            return
    
    print("✅ Temporary names created")
    
    # Step 2: Rename to final sequential names
    print("\nStep 2: Renaming to final names...")
    success_count = 0
    
    for temp_path, number, extension in temp_renames:
        new_filename = f"{number}{extension}"
        new_path = os.path.join(IMAGE_FOLDER, new_filename)
        
        try:
            os.rename(temp_path, new_path)
            success_count += 1
            if success_count % 100 == 0:
                print(f"Progress: {success_count}/{len(temp_renames)}", end='\r')
        except OSError as e:
            print(f"❌ Failed to rename {Path(temp_path).name} to {new_filename}: {e}")
    
    print(f"\n✅ Successfully renamed {success_count} images")
    
    # Summary
    print("="*50)
    print("📊 SUMMARY")
    print("="*50)
    print(f"Total images: {len(files)}")
    print(f"Successfully renamed: {success_count}")
    print(f"Format: 1{Path(files[0]).suffix}, 2{Path(files[0]).suffix}, ...")
    print("="*50)

if __name__ == "__main__":
    try:
        # Ask for confirmation
        print("⚠️  WARNING: This will rename ALL images in the folder!")
        print(f"Folder: {IMAGE_FOLDER}")
        response = input("\nContinue? (yes/no): ").strip().lower()
        
        if response in ['yes', 'y']:
            rename_images_sequential()
        else:
            print("❌ Operation cancelled")
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")