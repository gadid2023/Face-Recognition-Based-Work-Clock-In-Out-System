import os
import pickle

import tkinter as tk
from tkinter import messagebox
from deepface import DeepFace
import numpy as np
import logging
from scipy.spatial.distance import cosine



def get_button(window, text, color, command, fg='white'):
    button = tk.Button(
                        window,
                        text=text,
                        activebackground="black",
                        activeforeground="White",
                        fg=fg,
                        bg=color,
                        command=command,
                        height=2,
                        width=20,
                        font=('Helvetica bold', 20)
                    )

    return button


def get_img_label(window):
    label = tk.Label(window)
    label.grid(row=0, column=0)
    return label


def get_text_label(window, text):
    label = tk.Label(window, text=text)
    label.config(font=("sans-serif", 21), justify="left")
    return label


def get_entry_text(window):
    inputtxt = tk.Text(window,
                       height=2,
                       width=15, font=("Arial", 32))
    return inputtxt


def msg_box(title, description):
    messagebox.showinfo(title, description)



def recognize(img, db_path):
    # it is assumed there will be at most 1 match in the db
    try: 
        logging.info("Starting face recognization")

        
        embeddings_unknown = DeepFace.represent(
            img,
            model_name = "Facenet",
            enforce_detection=True
            )[0]['embedding']
    except ValueError:
        return 'no_person_found'
    except Exception as e:
        print(f"Error getting embeddings: {str(e)}")
        return 'no_person_found'
    


    db_dir = sorted(os.listdir(db_path))

    if not db_dir:
        return 'unknown_person'
    


    min_distance = float('inf')
    best_match_name = 'unknown_person'
    threshold = 0.6  # Adjust this threshold based on your needs

    # Compare with each known face
    for file_name in db_dir:
        try:
            with open(os.path.join(db_path, file_name), 'rb') as file:
                known_embedding = np.array(pickle.load(file)).flatten()
                
                # Ensure both embeddings are numpy arrays and have the right shape
                embeddings_unknown = np.array(embeddings_unknown).flatten()  # Flatten if necessary
        
                # Check if both embeddings have the correct length
                if known_embedding.shape[0] != 128:
                    logging.warning(f"Skipping {file_name}: Invalid embedding length.")
                    continue

                # Calculate cosine similarity between embeddings
                distance = cosine(
                    embeddings_unknown,
                    known_embedding
                )
                
                # Update best match if this distance is smaller
                if distance < min_distance:
                    min_distance = distance
                    if distance < threshold:
                        best_match_name = file_name[:-7]  # Remove .pickle extension

        except (pickle.PickleError, ValueError) as e:
            logging.error(f"Error processing {file_name}: {str(e)}")
            continue
    if best_match_name == 'unknown_person':
        logging.info("No matching face found in the database.")
    else:
        logging.info(f"Match found: {best_match_name} with a distance of {min_distance:.4f}")

    return best_match_name



