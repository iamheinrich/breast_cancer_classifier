import pandas as pd
import pickle
import png
import pydicom
from pathlib import Path
import numpy as np
import os

NUM_EXAM_SAMPLES = 2
EXAMS_PATH = "/Users/hendrik/Studium/Master/Thesis/Data/physionet.org/files/vindr-mammo/1.0.0/images"
ANNOTATIONS_PATH = "/Users/hendrik/Studium/Master/Thesis/Data/physionet.org/files/vindr-mammo/1.0.0/breast-level_annotations.csv"
IMG_OUTPUT_PATH = "/Users/hendrik/Studium/Master/Thesis/Code/breast_cancer_classifier/vin_dr_mamo_data/images"
EXAM_LIST_OUTPUT_PATH = "/Users/hendrik/Studium/Master/Thesis/Code/breast_cancer_classifier/vin_dr_mamo_data/"


def save_dicom_image_as_png(dicom_filename, png_filename, bitdepth=12):
    """
    Save 12-bit mammogram from dicom as rescaled 16-bit png file.
    :param dicom_filename: path to input dicom file.
    :param png_filename: path to output png file.
    :param bitdepth: bit depth of the input image. Set it to 12 for 12-bit mammograms.
    """
    try:
        dicom = pydicom.read_file(dicom_filename)
        image = dicom.pixel_array

        # Normalize the pixel values to fit into the specified bit depth
        max_pixel_value = np.max(image)
        scale_factor = (2 ** bitdepth - 1) / max_pixel_value
        image = (image * scale_factor).astype(np.uint16)  # Ensure image is of type uint16

        with open(png_filename, 'wb') as f:
            writer = png.Writer(height=image.shape[0], width=image.shape[1], bitdepth=bitdepth, greyscale=True)
            writer.write(f, image.tolist())
    except FileNotFoundError:
        print(f"File {dicom_filename} not found.")
    except Exception as e:
        print(f"Error while saving {dicom_filename} as PNG: {e}")


def get_img_view(img_filename, annotations):
    """
    Find image_id ("id") laterality ("R" or "L") and view_position ("CC" or "MLO")
    :param img_filename: image filename without extension
    :param annotations: DataFrame containing the annotations
    :return: view in the format "laterality-view_position"
    """
    try:
        row = annotations.loc[annotations['image_id'] == img_filename, ["laterality", "view_position"]]
        laterality = row['laterality'].values[0]
        view_position = row['view_position'].values[0]
        return f"{laterality}-{view_position}"
    except IndexError:
        print(f"Annotations for image {img_filename} not found.")
        return None
    except Exception as e:
        print(f"Error while getting view for {img_filename}: {e}")
        return None


def get_label(img_filename, annotations):
    """
    Get the label for an image based on BI-RADS level
    :param img_filename: image filename without extension
    :param annotations: DataFrame containing the annotations
    :return: label as integer
    """
    try:
        row = annotations.loc[annotations['image_id'] == img_filename, ["breast_birads"]]
        birads_level = row["breast_birads"].values[0]
        if birads_level in ["BI-RADS 1", "BI-RADS 2", "BI-RADS 3"]:
            return 0
        elif birads_level in ["BI-RADS 4", "BI-RADS 5"]:
            return 1
        else:
            raise ValueError("BI-RADS level must be between 1 and 5.")
    except IndexError:
        print(f"Annotations for image {img_filename} not found.")
        return None
    except Exception as e:
        print(f"Error while getting label for {img_filename}: {e}")
        return None


def create_exam_list():
    """
    Create a list of exams and save it to a pickle file.
    """
    # Check if paths exist
    if not os.path.exists(ANNOTATIONS_PATH):
        print(f"Annotations file not found at {ANNOTATIONS_PATH}.")
        return
    if not os.path.exists(EXAMS_PATH):
        print(f"Exams directory not found at {EXAMS_PATH}.")
        return
    if not os.path.exists(IMG_OUTPUT_PATH):
        print(f"Image output path not found. Creating directory {IMG_OUTPUT_PATH}.")
        os.makedirs(IMG_OUTPUT_PATH, exist_ok=True)
    if not os.path.exists(EXAM_LIST_OUTPUT_PATH):
        print(f"Exam list output path not found. Creating directory {EXAM_LIST_OUTPUT_PATH}.")
        os.makedirs(EXAM_LIST_OUTPUT_PATH, exist_ok=True)

    try:
        annotations = pd.read_csv(ANNOTATIONS_PATH)
    except Exception as e:
        print(f"Error reading annotations file: {e}")
        return

    labels = {}
    exam_list = []
    exams_path = Path(EXAMS_PATH) #Path to folder that contains exams folder, which contains 4 dicom images
    png_img_output_path = Path(IMG_OUTPUT_PATH)

    #Iterate over exam folders
    for i, exam in enumerate(exams_path.iterdir()):

        # Stop once enough samples are generated
        if i >= NUM_EXAM_SAMPLES:
            break

        # Labels being initially 0
        left_label = 0
        right_label = 0

        #Dict for exam. Structure according to breast cancer classifier sample
        exam_dict = {
            'horizontal_flip': 'NO',
            'L-CC': [],
            'L-MLO': [],
            'R-MLO': [],
            'R-CC': []
        }

        #Iterate over dicom images in exam folder
        for img in exam.glob('*.dicom'):
            png_filename = png_img_output_path / (img.stem + ".png")
            save_dicom_image_as_png(img, png_filename)

            img_filename = img.stem
            view = get_img_view(img_filename, annotations)
            if view is None:
                continue
            exam_dict[view].append(img_filename)

            img_label = get_label(img_filename, annotations)
            if img_label is None:
                continue

            if view == "L-CC" or view == "L-MLO":
                if img_label == 1:
                    left_label = 1
            elif view == "R-CC" or view == "R-MLO":
                if img_label == 1:
                    right_label = 1

        exam_list.append(exam_dict)
        exam_id = exam.stem
        labels[exam_id] = {
            "L": left_label,
            "R": right_label
        }

    try:
        with open(EXAM_LIST_OUTPUT_PATH + "exam_labels.pkl", "wb") as f:
            pickle.dump(labels, f)
    except Exception as e:
        print(f"Error while saving pickle file: {e}")

    try:
        with open(EXAM_LIST_OUTPUT_PATH + "exam_list_before_cropping.pkl", 'wb') as f:
            pickle.dump(exam_list, f)
    except Exception as e:
        print(f"Error while saving pickle file: {e}")


def main():
    create_exam_list()


if __name__ == "__main__":
    main()
