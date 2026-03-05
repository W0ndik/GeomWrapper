from tkinter import Tk
from tkinter import filedialog


def pick_stl_file() -> str:
    root = Tk()
    root.withdraw()
    root.update()
    path = filedialog.askopenfilename(
        title="Select STL file",
        filetypes=[("STL files", "*.stl"), ("All files", "*.*")]
    )
    root.destroy()
    return path or ""