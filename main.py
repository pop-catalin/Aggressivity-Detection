from View.UI import UI
import tkinter as tk
from Model.Model import Model
from Controller.Controller import Controller

def main():
    model = Model()
    controller = Controller(model, 'dataset')
    
    root = tk.Tk()
    ui = UI(root, "Aggressivity Detection - A Security Tool", controller)

if __name__ == '__main__':
    main()