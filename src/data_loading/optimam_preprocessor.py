from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import omidb
from omidb import client, study, series, image
import pydicom
import png
import numpy as np
import pickle
from enum import Enum

class View(str, Enum):
    """Standard mammographic views"""
    L_CC = "L-CC"   # Left Cranio-Caudal
    L_MLO = "L-MLO" # Left Medio-Lateral Oblique
    R_CC = "R-CC"   # Right Cranio-Caudal
    R_MLO = "R-MLO" # Right Medio-Lateral Oblique

@dataclass
class DatabaseConfig:
    """Configuration for OPTIMAM database paths"""
    data_path: Path
    images_path: Path
    output_path: Path

class OptimamPreprocessor:
    """
    A class to explore and process the OPTIMAM mammography database.
    
    The OPTIMAM database is structured as:
    - Clients (patients) have multiple Episodes
    - Episodes contain medical procedures/events and associated Studies
    - Studies are imaging examinations with multiple Series
    - Series contain images for one specific position/view
    - In mammography, each Series typically contains one image
    """
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.db = omidb.DB(
            str(config.data_path), 
            str(config.images_path), 
            ignore_missing_images=False
        )
        
    def get_clients(self) -> List[client.Client]:
        """Get all clients from the database."""
        return [client for client in self.db]
    
    def is_screening_event(self, study: study.Study) -> bool:
        """
        Check if a study is associated with a screening event.
        Note: Events are properties of Episodes, not Studies directly.
        Studies are linked to events through their containing Episode.
        """
        return omidb.events.Event.screening in study.event_type
    
    def get_studies_with_complete_views(self, client: client.Client) -> List[study.Study]:
        """
        Get studies that have all four standard mammogram views.
        
        A complete study should have 4 series (potentially 8 if both FOR PROCESSING
        and FOR PRESENTATION are included), one for each standard view:
        - L-CC (Left Cranio-Caudal)
        - L-MLO (Left Medio-Lateral Oblique)
        - R-CC (Right Cranio-Caudal)
        - R-MLO (Right Medio-Lateral Oblique)
        """
        complete_studies = []
        for episode in client.episodes:
            if not episode.studies:
                continue
            for study in episode.studies:
                if study.series and len(study.series) >= 4:  # Minimum 4 series for complete views
                    complete_studies.append(study)
        return complete_studies
    
    @staticmethod
    def get_view_from_dcm(dcm: pydicom.dataset.FileDataset) -> str:
        """
        Extract view information from DICOM metadata.
        Each series represents one specific position/view.
        """
        return f"{dcm.ImageLaterality}-{dcm.ViewPosition}"
    
    @staticmethod
    def create_img_path(client_id: str, study_id: str, img_id: str) -> str:
        """Create standardized image path following OPTIMAM structure."""
        return f"{client_id}/{study_id}/{img_id}.dcm"
    
    def create_exam_list(self, num_exams: int = None) -> Tuple[List[Dict], List[List[str]]]:
        """
        Create a list of complete exams with all four standard views.
        
        Args:
            num_exams: Optional limit on number of exams to process
            
        Returns:
            Tuple of (exam_list, image_paths)
        """
        exam_list = []
        exam_list_img_paths = []
        
        for client in self.get_clients():
            if num_exams and len(exam_list) >= num_exams:
                break
                
            for study in self.get_studies_with_complete_views(client):
                study_dict = {view.value: [] for view in View}
                study_dict['horizontal_flip'] = 'NO'
                study_img_paths = []
                
                for series in study.series:
                    if len(series.images) != 1:
                        continue
                        
                    img = series.images[0]
                    if img.dcm.PresentationIntentType != "FOR PRESENTATION":
                        continue
                        
                    view = self.get_view_from_dcm(img.dcm)
                    if view not in study_dict:
                        continue
                        
                    img_id = img.dcm.SOPInstanceUID
                    study_dict[view] = [img_id]
                    img_path = self.create_img_path(client.id, study.id, img_id)
                    study_img_paths.append(img_path)
                
                # Check if all views are present
                if all(study_dict[view.value] for view in View):
                    exam_list.append(study_dict)
                    exam_list_img_paths.append(study_img_paths)
                    
        return exam_list, exam_list_img_paths
    
    @staticmethod
    def save_dicom_as_png(
        dicom_path: Path,
        png_path: Path
    ) -> None:
        """
        Convert DICOM mammogram to 16-bit PNG format, scaling to use the full dynamic range.
        
        Args:
            dicom_path: Path to input DICOM file
            png_path: Path to output PNG file
        """
        try:
            dicom_file = pydicom.read_file(str(dicom_path))
            image = dicom_file.pixel_array
            
            # Normalize pixel values to full 16-bit range
            max_pixel_value = np.max(image)
            
            if max_pixel_value == 0: # Handle blank images
                scaled_image = image.astype(np.uint16)
            else:
                # Scale to use the full 0 - (2^16 - 1) range
                scale_factor = (2**16 - 1) / max_pixel_value
                scaled_image = (image * scale_factor).astype(np.uint16)
            
            # Ensure the output directory for the PNG file exists
            png_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(png_path, 'wb') as f:
                writer = png.Writer(
                    height=scaled_image.shape[0],
                    width=scaled_image.shape[1],
                    bitdepth=16,  # Output 16-bit PNG
                    greyscale=True
                )
                writer.write(f, scaled_image.tolist())
                
        except Exception as e:
            print(f"Error processing {dicom_path}: {e}")
            
    def generate_nyu_classifier_input_pickle(self, num_exams: int) -> None:
        """
        Generates the input data (pickled exam list and PNG images) 
        for the classifier from a specified number of exams.
        
        Args:
            num_exams: Number of exams to process
        """
        exam_list, img_paths = self.create_exam_list(num_exams)
        
        # Save exam list
        exam_list_path = self.config.output_path / "exam_list_before_cropping.pkl"
        # Ensure the output directory for the pickle file exists
        exam_list_path.parent.mkdir(parents=True, exist_ok=True)
        with open(exam_list_path, 'wb') as f:
            pickle.dump(exam_list, f)
            
        # Convert images
        images_output_path = self.config.output_path / "images"
        images_output_path.mkdir(exist_ok=True)
        
        for exam_paths in img_paths:
            for img_path in exam_paths:
                dicom_path = self.config.images_path / img_path
                png_filename = Path(img_path).stem + ".png"
                png_path = images_output_path / png_filename
                self.save_dicom_as_png(dicom_path, png_path)

def main():
    """Main function to demonstrate usage."""
    config = DatabaseConfig(
        data_path=Path('/Users/hendrik/Studium/Master/Thesis/Data/OMI-DB Sample/DATA'),
        images_path=Path('/Users/hendrik/Studium/Master/Thesis/Data/OMI-DB Sample/IMAGES'),
        output_path=Path('/Users/hendrik/Studium/Master/Thesis/Code/breast_cancer_classifier/omi_db_data_test')
    )
    
    preprocessor = OptimamPreprocessor(config)
    preprocessor.generate_nyu_classifier_input_pickle(num_exams=4)

if __name__ == "__main__":
    main()
