import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import os
from PIL import Image, ImageTk
import logging


class ThreatHunterUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Enterprise AI Threat Hunter")
        self.root.geometry("900x600")
        self.root.configure(bg="#1e1e1e")

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("Treeview", background="#2d2d2d", foreground="white", fieldbackground="#2d2d2d",
                             borderwidth=0)
        self.style.map("Treeview", background=[('selected', '#4a90e2')])

        self.create_widgets()
        self.load_data()

    def create_widgets(self):
        title_label = tk.Label(self.root, text="AI-POWERED ANOMALY DETECTION BROWSER", font=("Helvetica", 18, "bold"),
                               fg="#4a90e2", bg="#1e1e1e", pady=20)
        title_label.pack()

        self.table_frame = tk.Frame(self.root, bg="#1e1e1e")
        self.table_frame.pack(fill="both", expand=True, padx=20)

        self.tree = ttk.Treeview(self.table_frame, columns=("ID", "Timestamp", "Metric", "Score", "Action"), show="headings")
        self.tree.heading("ID", text="ID")
        self.tree.heading("Timestamp", text="Timestamp")
        self.tree.heading("Metric", text="Primary Metric")
        self.tree.heading("Score", text="Anomaly Score")
        self.tree.heading("Action", text="AI Response")

        self.tree.column("ID", width=50)
        self.tree.column("Score", width=100)
        self.tree.pack(fill="both", expand=True)
        self.btn_frame = tk.Frame(self.root, bg="#1e1e1e", pady=20)
        self.btn_frame.pack()

        self.view_xai_btn = tk.Button(self.btn_frame, text="VIEW XAI EXPLANATION", command=self.show_xai, bg="#4a90e2", fg="white", font=("Helvetica", 10, "bold"), padx=20)
        self.view_xai_btn.pack(side="left", padx=10)

        self.refresh_btn = tk.Button(self.btn_frame, text="REFRESH LOGS", command=self.load_data, bg="#2d2d2d", fg="white", font=("Helvetica", 10), padx=20)
        self.refresh_btn.pack(side="left", padx=10)

    def load_data(self):
        for i in self.tree.get_children():
            self.tree.delete(i)

        try:
            if os.path.exists("data/raw/system_logs.csv"):
                df = pd.read_csv("data/raw/system_logs.csv").tail(15)
                for idx, row in df.iterrows():
                    status = "Isolate Host" if row['cpu_usage'] > 80 else "Monitoring"
                    self.tree.insert("", "end", values=(idx, "Recent", "CPU Usage", f"{row['cpu_usage']:.2f}", status))
            else:
                messagebox.showwarning("Warning", "Log file not found. Run the generator first.")
        except Exception as e:
            logging.error(f"UI Loading Error: {e}")

    def show_xai(self):
        xai_path = "logs/global_feature_importance.png"
        if os.path.exists(xai_path):
            img_win = tk.Toplevel(self.root)
            img_win.title("SHAP Feature Importance")
            img = Image.open(xai_path)
            img = img.resize((600, 400), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            lbl = tk.Label(img_win, image=photo)
            lbl.image = photo
            lbl.pack()
        else:
            messagebox.showinfo("XAI Status", "No XAI plot found. Run main.py to generate explanations.")


if __name__ == "__main__":
    root = tk.Tk()
    app = ThreatHunterUI(root)
    root.mainloop()
