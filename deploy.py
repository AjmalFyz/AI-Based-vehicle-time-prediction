import os
import torch
import cv2
import time
import json
import numpy as np
import easyocr
import datetime

# DEFINING GLOBAL VARIABLES
EASY_OCR = easyocr.Reader(['en'])  # initiating easyocr
OCR_TH = 0.2

# Function to save extracted text in JSON file


def save_text_to_json(file_path, text_dict):
    with open(file_path, 'w') as json_file:
        json.dump(text_dict, json_file)

# Function to run detection


def detectx(frame, model):
    frame = [frame]
    print(f"[INFO] Detecting...")
    results = model(frame)
    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cordinates

# Function to plot the BBox and results


def plot_boxes(results, frame, classes):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    print(f"[INFO] Total {n} detections...")
    print(f"[INFO] Looping through all detections...")
    text_dict = {}
    for cls in classes:
        text_dict[cls] = []
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.55:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] *
                                                        y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
            text_d = classes[int(labels[i])]
            coords = [x1, y1, x2, y2]
            plate_info = recognize_plate_easyocr(
                img=frame, coords=coords, reader=EASY_OCR, region_threshold=OCR_TH)
            if plate_info is not None:
                if text_d in text_dict:
                    text_dict[text_d].append(plate_info)
                else:
                    text_dict[text_d] = [plate_info]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1 - 20), (x2, y1), (0, 255, 0), -1)
                cv2.putText(frame, f"{plate_info}", (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame, text_dict

# Function to recognize license plate numbers using EasyOCR


def recognize_plate_easyocr(img, coords, reader, region_threshold):
    xmin, ymin, xmax, ymax = coords
    nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)]
    ocr_result = reader.readtext(nplate)
    text = filter_text(region=nplate, ocr_result=ocr_result,
                       region_threshold=region_threshold)
    if len(text) == 1:
        text = text[0].upper()
        return text
    return None

# Function to filter out wrong detections


def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0] * region.shape[1]
    plate = []
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        if length * height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate

# Main function


def main(img_folder_path):
    img_out_folder = "./output-images"
    json_file_path = "./json-files/extracted_text.json"

    # Load the custom trained model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./last.pt')
    classes = model.module.names if hasattr(model, 'module') else model.names

    # Create a list to store extracted text
    extracted_text_list = []

    # Process images in a folder
    if img_folder_path is not None:
        print(f"[INFO] Working with images in folder: {img_folder_path}")

        for file_name in os.listdir(img_folder_path):
            if file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png'):
                img_path = os.path.join(img_folder_path, file_name)

                print(f"[INFO] Processing image: {img_path}")

                start_time = time.time()  # Record the start time

                frame = cv2.imread(img_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = detectx(frame, model=model)

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                frame, text_dict = plot_boxes(results, frame, classes=classes)

                image_info_dict = {}  # Create a dictionary to store the extracted information

                for text_type, plate_info_list in text_dict.items():
                    if len(plate_info_list) > 0:
                        # Store the first value from the list of plate_info
                        image_info_dict[text_type] = plate_info_list[0]

                # Add additional information to the image_info_dict
                now = datetime.datetime.now()
                image_info_dict['detection_day'] = now.day
                image_info_dict['detection_date'] = now.strftime("%Y-%m-%d")
                image_info_dict['detection_month'] = now.strftime("%B")
                image_info_dict['detection_year'] = now.year
                image_info_dict['detection_time'] = now.strftime("%H:%M:%S")

                extracted_text_list.append({
                    "filename": file_name,
                    "data": image_info_dict
                })

                elapsed_time = time.time() - start_time  # Calculate the elapsed time

                img_name = img_path.split('/')[-1]
                img_out_path = os.path.join(img_out_folder, img_name)

                cv2.imwrite(img_out_path, frame)

                print(f"[INFO] Saved output image: {img_out_path}")
                print(f"[INFO] Elapsed time: {elapsed_time} seconds")

        # Save the extracted text to the JSON file
        save_text_to_json(json_file_path, extracted_text_list)

        print(f"[INFO] Saved extracted text to: {json_file_path}")
        print(
            f"[INFO] Finished processing images in folder: {img_folder_path}")

    else:
        print("[ERROR] No input specified.")


# Call the main function with the image folder
img_folder_path = "./in-folder"
main(img_folder_path)
