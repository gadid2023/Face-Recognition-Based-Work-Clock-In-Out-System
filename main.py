import tkinter as tk
import datetime
import cv2
import util
from PIL import Image, ImageTk
import os.path
from deepface import DeepFace
import pickle
import util


class App:
    def __init__(self):
        self.main_window = tk.Tk()  # Create the main window
        self.main_window.attributes("-fullscreen", True)  # Set window size and position

        self.login_button_main_window = util.get_button(self.main_window, 'login', 'green', self.login )
        self.login_button_main_window.place(x=1050, y=500)

        self.new_user_main_window = util.get_button(self.main_window, "New User", 'red', self.register)
        self.new_user_main_window.place(x=1050, y=600)

        self.capture_label = util.get_img_label(self.main_window)
        self.capture_label.place(x=100, y=100, width=900, height=600)  # Adjust position as needed

        # Add capture frame from the webcam
        self.add_capture_frame(self.capture_label)



        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        self.log_path = './log.txt'
        log_dir = os.path.dirname(self.log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.main_window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.main_window.destroy()
                


    def add_capture_frame(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)

        self._label = label
        self.process_webcam()


    def process_webcam(self):

        try:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                return
        
            self.most_recent_capture_arr = frame
            img_rgb = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)

            self.most_recent_capture_pil = Image.fromarray(img_rgb)
            imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)

            self._label.imgtk = imgtk
            self._label.configure(image=imgtk)

            self._label.after(20, self.process_webcam)

        except Exception as e:
            print(f"Error in process_webcam: {str(e)}")

    def login(self):
        try:
            name = util.recognize(self.most_recent_capture_arr, self.db_dir)

            if name in ['unknown_person', 'no_person_found']:
                util.msg_box('oops...', 'Unknown user. Please register new user or try again.')
            else:
                util.msg_box('Welcome back !', 'Welcome, {}.'.format(name))
                with open(self.log_path, 'a') as f:
                    f.write(f'{name}, {datetime.datetime.now()}')
                    f.close()
        except Exception as e:
            util.msg_box('Error', f'Login failed: {str(e)}')

    def register(self):

        self.add_img_to_label(self.capture_label)

        self.entry_text_register_new_user = util.get_entry_text(self.main_window)
        self.entry_text_register_new_user.place(x=750, y=150)

        self.text_label_register_new_user = util.get_text_label(self.main_window, 'Please, \ninput username:')
        self.text_label_register_new_user.place(x=750, y=70)

        self.accept_button_new_user = util.get_button(self.main_window, "Accept",'green', self.accept_register_new_user)
        self.accept_button_new_user.place(x=750, y=200)




    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)

        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c")
        try:
                try:
                    
                    embeddings = DeepFace.represent(
                        self.register_new_user_capture, 
                        model_name="Facenet",
                        enforce_detection = True
                    )[0]['embedding']
                except ValueError:
                    util.msg_box('Error','No face detected in image')
                    return
                except Exception as e:
                    util.msg_box('Error', f'Failed to process face: {str(e)}')
                    return
                
                try:

                    file_path = os.path.join(self.db_dir, f'{name}.pickle')
                    with open(file_path, 'wb') as file:
                        pickle.dump(embeddings, file)
                except Exception as e:
                    util.msg_box('Error', f'Failed to save user data: {str(e)}')
                    

                util.msg_box('Success!', 'User was registered successfully !')
                    
                self.entry_text_register_new_user.destroy()
                self.text_label_register_new_user.destroy()
                self.accept_button_new_user.destroy()

        except Exception as e:
                util.msg_box('Error',f'Failed to register user: {str(e)}')

    def start(self):
        self.main_window.mainloop()  # Start the Tkinter loop (keeps the window open)

# Initialize and run the application
if __name__ == "__main__":
    app = App()
    app.start()