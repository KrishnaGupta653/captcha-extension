import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import re
import shutil

# --- CONFIGURATION ---
# 1. Folder containing your UNLABELED images
SOURCE_FOLDER = r'C:\Users\kg060\Desktop\tatkal\dataset\raw_captcha_images'

# 2. Folder where LABELED images will be moved (Created automatically if not exists)
DEST_FOLDER = r'C:\Users\kg060\Desktop\tatkal\dataset\labeled_captcha_images'

class CaptchaLabeler:
    def __init__(self, root, source_folder, dest_folder):
        self.root = root
        self.source_folder = source_folder
        self.dest_folder = dest_folder
        self.history = []  # Stack to store actions for "Undo"

        self.root.title("Manual Captcha Labeler v2")
        self.root.geometry("500x450")

        # Create destination folder if it doesn't exist
        if not os.path.exists(self.dest_folder):
            os.makedirs(self.dest_folder)

        # Get list of images
        self.image_files = self.get_unlabeled_images()
        
        # We always start at index 0 because "processed" files are moved away.
        # The "0th" file is always the next one in the queue.
        self.current_index = 0

        # --- GUI ELEMENTS ---
        
        # 1. Info Label
        self.lbl_info = tk.Label(root, text="", font=("Arial", 10))
        self.lbl_info.pack(pady=5)

        # 2. Image Display
        self.lbl_image = tk.Label(root)
        self.lbl_image.pack(pady=20)

        # 3. Instruction
        tk.Label(root, text="Type Captcha & Press Enter:", font=("Arial", 10, "bold")).pack()

        # 4. Entry Box
        self.entry_var = tk.StringVar()
        self.entry = tk.Entry(root, textvariable=self.entry_var, font=("Courier", 24), justify='center')
        self.entry.pack(pady=10, ipady=5)
        self.entry.focus_set()

        # 5. Buttons Frame
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=20)
        
        # Previous Button
        self.btn_prev = tk.Button(btn_frame, text="<< Previous (Undo)", command=self.undo_previous, state=tk.DISABLED, width=20)
        self.btn_prev.pack(side=tk.LEFT, padx=5)

        # Skip Button
        tk.Button(btn_frame, text="Skip", command=self.skip_image, width=10).pack(side=tk.LEFT, padx=5)
        
        # Save Button
        tk.Button(btn_frame, text="Save (Enter)", command=self.save_and_next, width=15, bg="#ddffdd").pack(side=tk.LEFT, padx=5)
        
        # Bindings
        self.root.bind('<Return>', lambda event: self.save_and_next())
        self.root.bind('<Control-z>', lambda event: self.undo_previous()) # Ctrl+Z for undo

        # Check if we have images
        if not self.image_files:
            messagebox.showinfo("Done", "No images found in source folder!")
            root.destroy()
            return

        self.load_current_image()

    def get_unlabeled_images(self):
        """Reloads file list from source directory"""
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
        # Sort to keep order consistent
        return sorted([f for f in os.listdir(self.source_folder) if f.lower().endswith(valid_exts)])

    def load_current_image(self):
        # Refresh list in case something changed, but try to stick to index
        self.image_files = self.get_unlabeled_images()

        if self.current_index >= len(self.image_files):
            self.lbl_image.config(image='')
            self.lbl_info.config(text="All images in this folder are done!")
            self.entry_var.set("")
            return

        filename = self.image_files[self.current_index]
        filepath = os.path.join(self.source_folder, filename)

        # Update Info
        remaining = len(self.image_files)
        done_count = len(os.listdir(self.dest_folder))
        self.lbl_info.config(text=f"To Do: {remaining} | Finished: {done_count}\nCurrent File: {filename}")

        # Enable/Disable Previous Button
        if self.history:
            self.btn_prev.config(state=tk.NORMAL)
        else:
            self.btn_prev.config(state=tk.DISABLED)

        # Display Image
        try:
            pil_image = Image.open(filepath)
            # Zoom in 2x
            w, h = pil_image.size
            pil_image = pil_image.resize((w*2, h*2), Image.Resampling.NEAREST)
            self.tk_image = ImageTk.PhotoImage(pil_image)
            self.lbl_image.config(image=self.tk_image)
            
            # Pre-fill text if filename looks like a label
            name_stem = os.path.splitext(filename)[0]
            self.entry_var.set("")
            
            # Heuristic: if filename is 4-8 chars and has no underscores, it might be the label
            if 4 <= len(name_stem) <= 8 and "_" not in name_stem:
                self.entry_var.set(name_stem)
                self.entry.select_range(0, tk.END)

        except Exception as e:
            print(f"Error loading image: {e}")
            self.skip_image()

    def sanitize_filename(self, text):
        return re.sub(r'[<>:"/\\|?*]', '_', text)

    def save_and_next(self):
        if self.current_index >= len(self.image_files):
            return

        user_input = self.entry_var.get().strip()
        if not user_input:
            return # Don't save empty

        filename = self.image_files[self.current_index]
        old_path = os.path.join(self.source_folder, filename)
        
        # Construct new path
        extension = os.path.splitext(filename)[1]
        safe_name = self.sanitize_filename(user_input)
        new_filename = f"{safe_name}{extension}"
        new_path = os.path.join(self.dest_folder, new_filename)

        # Handle duplicates in DEST folder
        counter = 1
        while os.path.exists(new_path):
            new_filename = f"{safe_name}_{counter}{extension}"
            new_path = os.path.join(self.dest_folder, new_filename)
            counter += 1

        try:
            # MOVE file to new folder
            shutil.move(old_path, new_path)
            
            # Record history for Undo (Source Name, Where it went)
            self.history.append({
                'original_name': filename,
                'current_path': new_path
            })
            
            print(f"Moved: {filename} -> {new_filename}")
            
            # Since we moved the file, it's gone from the list.
            # We stay at index 0 (or current) to view the *next* file which slides into this spot.
            self.load_current_image()

        except OSError as e:
            messagebox.showerror("Error", f"Could not move file:\n{e}")

    def skip_image(self):
        # Just increment index to look at next file without moving current one
        self.current_index += 1
        self.load_current_image()

    def undo_previous(self):
        if not self.history:
            return

        # Get last action
        last_action = self.history.pop()
        current_path = last_action['current_path'] # Where it is now (Labeled)
        original_name = last_action['original_name'] # What it was called (Unlabeled)
        
        original_path = os.path.join(self.source_folder, original_name)

        try:
            if os.path.exists(current_path):
                # Move it BACK to source
                shutil.move(current_path, original_path)
                print(f"Undo: Moved back {os.path.basename(current_path)}")
                
                # Reset index to 0 (or find where the file went in the sort order)
                # Simple approach: Reset to 0 so the user sees it (or the first available)
                self.current_index = 0 
                self.load_current_image()
            else:
                messagebox.showerror("Error", "Cannot undo: File not found in labeled folder!")
        except Exception as e:
            messagebox.showerror("Error", f"Undo failed: {e}")

if __name__ == "__main__":
    if not os.path.exists(SOURCE_FOLDER):
        print(f"Error: Source folder not found {SOURCE_FOLDER}")
    else:
        root = tk.Tk()
        app = CaptchaLabeler(root, SOURCE_FOLDER, DEST_FOLDER)
        root.mainloop()