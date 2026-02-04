if __name__ == "__main__":
    import tkinter as tk
    root = tk.Tk()
    root.title("GUI sanity check")
    tk.Label(root, text="GUI is working").pack(padx=20, pady=20)
    root.mainloop()
