import tkinter as tk
import numpy as np
import os

#-----------------------------INITIAL VARIABLES-----------------------------


drawings = []
digits = []
drawing = np.zeros((28,28))

data = "dataset.npz"
oldx = None
oldy = None
digit = None


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
            

def grayscale(val): #sets the value of the square to black gray or white values based on the cells value(0=black,1=white)
    val *= 255
    val = int(val)
    return f"#{val:02x}{val:02x}{val:02x}"            



def draw_pixels(drawing): # draws each cell of the grid
    canvas.delete("all")
    for i in range(len(drawing)):
        for j in range(len(drawing[i])):                                
            y1 = i * 10
            x1 = j * 10
            x2 = x1 + 10
            y2 = y1 + 10
            canvas.create_rectangle(x1, y1, x2, y2,fill=(grayscale(drawing[i][j])))




def click(event): # called when the user clicks on the canvas
    global oldx, oldy

    canvas.focus_set()
    
    x = event.x // 10
    y = event.y // 10

    assign_pixel(x, y)
    oldx = x
    oldy = y


    
def drag(event): # called when the user drags on the canvas
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



def key_press(event): # called when the user presses any key
    
    global drawing, digit # access the gloabal variables
    
    if event.keysym == "c":
        drawing = np.zeros((28,28)) # clear drawing
        
    elif len(event.char) == 1 and event.char in "0123456789":
        saveDrawing(drawing, int(event.char))
        drawing = np.zeros((28, 28)) # saves the drawing along with the label of whatever number was pressed


def saveDrawing(drawing,digit):
    if drawing.sum() == 0: # make sure your not submitting an empty drawing by accident
        return
    
    global drawings, digits # access the gloabal variables
    
    drawings.append(drawing.flatten()) # flatten because the network takes in the inputs as lists not 28x28 arrays
    digits.append(digit)
    save_all()
    update_counts()

    
def update_counts(): # display how many of each saved number there are
    counts = np.bincount(digits, minlength=10)
    status.config(text="  ".join(f"{d}s: {counts[d]}" for d in range(10)))

# -----------------------------MAIN PROGRAM-----------------------------


root = tk.Tk() # the main display pop up
root.minsize(400, 400)
root.title("data collect")

tk.Label(root, text="DATA COLLECTION").pack()

canvas = tk.Canvas(root, width=280, height=280, background="black") # drawing board
canvas.pack()

#keybinds associated to functions
canvas.bind("<Button-1>", click)
canvas.bind("<B1-Motion>", drag)
canvas.bind("<Key>", key_press) # assigns these inputs to when the user is using the canvas

tk.Label(root, text="left click to draw, submit with 0-9, c to clear").pack()


drawings, digits = load_saved()

def update(): # runs once every main cycle
    draw_pixels(drawing)
    root.after(16, update)

status = tk.Label(root, text="")
status.pack()

update_counts()

root.after(0, update)

root.mainloop()



