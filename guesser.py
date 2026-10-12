import tkinter as tk
import numpy as np
import os
from network import load_parameters, forward_pass, get_predictions
#-----------------------------INITIAL VARIABLES-----------------------------


drawings = []
digits = []
drawing = np.zeros((28,28))

data = "dataset.npz"
oldx = None
oldy = None
digit = None

W1, W2, b1, b2 = load_parameters("digits_weights.npz")
# -----------------------------FUNCTIONS-----------------------------


def load_saved():
    if os.path.exists(data):
        saved = np.load(data)
        return list(saved["drawings"]), list(saved["labels"])
    return [],[]



def save_all():
    np.savez(data,
             drawings=np.array(drawings, dtype=np.float32),
             labels=np.array(digits, dtype=np.int64))




def assign_pixel(x,y):
    # ys and xs are inverted because the list is rows by columns 
    if 0 <= x < 28 and 0 <= y < 28:
        drawing[y, x] = 1
        if x > 0 and drawing[y, x-1] == 0:
            drawing[y, x-1] = 0.5      # the cell to the left
        if x < 27 and drawing[y, x+1] == 0:
            drawing[y, x+1] = 0.5      # the cell to the right
        if y > 0 and drawing[y-1, x] == 0:
            drawing[y-1, x] = 0.5      # the cell above
        if y < 27 and drawing[y+1, x] == 0:
            drawing[y+1, x] = 0.5      # the cell below
            

def grayscale(val): 
    val *= 255
    val = int(val)
    return f"#{val:02x}{val:02x}{val:02x}"            



def draw_pixels(drawing):
    canvas.delete("all")
    for i in range(len(drawing)):
        for j in range(len(drawing[i])):                                
            y1 = i * 10
            x1 = j * 10
            x2 = x1 + 10
            y2 = y1 + 10
            canvas.create_rectangle(x1, y1, x2, y2,fill=(grayscale(drawing[i][j])))




def click(event):
    global oldx, oldy

    canvas.focus_set()
    
    x = event.x // 10
    y = event.y // 10

    assign_pixel(x, y)
    oldx = x
    oldy = y


    
def drag(event):
    global oldx, oldy
    x = event.x // 10
    y = event.y // 10

    if oldx != None:
        
        dx = x-oldx
        dy = y-oldy
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            assign_pixel(x, y)
        else:
            for i in range(steps + 1):
                px = round(oldx + dx * i / steps)
                py = round(oldy + dy * i / steps)

                assign_pixel(px, py)
    else:
        assign_pixel(x, y)

    oldx = x
    oldy = y



def key_press(event):
    
    global drawing, digit
    
    if event.keysym == "c":
        drawing = np.zeros((28,28))
        





def predict_current(drawing):
    flat = drawing.flatten().reshape(1, -1)  # reshape to (1, 784) for forward pass
    pred = get_predictions(flat, W1, W2, b1, b2)
    prediction_label.config(text=f"Prediction: {pred[0]}")


# -----------------------------MAIN PROGRAM-----------------------------


root = tk.Tk()
root.minsize(400, 400)
root.title("data collect")

tk.Label(root, text="DATA COLLECTION").pack()

canvas = tk.Canvas(root, width=280, height=280, background="black")
canvas.pack()

#keybinds associated to functions
canvas.bind("<Button-1>", click)
canvas.bind("<B1-Motion>", drag)
canvas.bind("<Key>", key_press)

tk.Label(root, text="left click to draw, c to clear").pack()
prediction_label = tk.Label(root,text="Prediction: -")

drawings, digits = load_saved()

def update():
    draw_pixels(drawing)
    root.after(16, update)
    predict_current(drawing)

status = tk.Label(root, text="")
status.pack()

prediction_label.pack()

root.after(0, update)

root.mainloop()



