import tkinter as tk
import sys
import os

#ssh jd-baglass@10.16.40.89
if os.environ.get('Display','') == '':
    print('no display found. using 0.0')
    os.environ.__setitem__('DISPLAY',':0.0')
def alarm(location):
    # Alarm function
    # Initialize Pygame for sound
    #pygame.init()
    #pygame.mixer.init()
    # Load the sound file (replace 'alarm_sound.wav' with your sound file)

    #pygame.mixer.music.load('/home/jd-baglass/Downloads/videoplayback.wav')
    #root = tk.Tk()
    #large_font = (Helvetica, 36)
    # Create the main window (full-screen)
    root = tk.Tk()
    root.title("Jam Detected")
    root.wm_attributes("-topmost", True)

    # Get screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Set window size to cover the entire screen
    root.geometry(f"{screen_width}x{screen_height}")
    # Remove window decorations (optional)
    root.overrideredirect(True)
    # Play the alarm sound
    #pygame.mixer.music.play(-1)  # -1 means play the sound in an infinite loop
    # Wait for user input to stop the alarm
    # exit button
    exit_button = tk.Button(root, justify="center", text = f"JAM DETECTED {location}!", font='Helvetica 16 bold', command=lambda: root.quit())
    exit_button.pack(ipadx=5, ipady=5, expand=True)
    #label = tk.Label(root, text=fJAM DETECTED {location}!, font=large_font, bg='red')
    #label.pack(expand=True)
    def change_color(i=0):
        colors = ('red', 'blue')
        if i < 2:
            root.config(bg=colors[i])
            root.after(500, change_color, i+1)
        else:
            root.config(bg=colors[0])
            root.after(500, change_color, 0)
            root.config(bg='red')
    change_color()
    root.attributes('-alpha', 0.8)
    root.mainloop()
    root.destroy()
    root.quit()
#pygame.mixer.music.stop()
import socket

ip = "10.16.40.89" # IP of Raspberry Pi start server
serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serv.bind((ip, 8080))
serv.listen(5)
print("SERVER: started")

while True:
    # establish connection
    conn, addr = serv.accept()
    from_client = ''
    print("SERVER: connection to Client established")

    while True:
        # receive data and print
        data = conn.recv(4096).decode()
        if not data: break
        from_client = data
        alarm(from_client)

    # close connection and exit
    conn.close()
    break