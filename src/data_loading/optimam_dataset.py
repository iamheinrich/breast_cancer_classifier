from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import torch
from torch.utils.data import Dataset
import pydicom
import os
import json
import numpy as np
from dataclasses import dataclass

@dataclass
class OptimamConfig:
    """Minimal configuration for OPTIMAM dataset"""
    data_path: str
    images_path: str

class OptimamDataset(Dataset):
    """Minimal PyTorch Dataset for OPTIMAM mammography data."""
    
    def __init__(self, config: OptimamConfig):
        self.config = config
        self.studies = self._load_studies()
        print(f"Loaded {len(self.studies)} complete studies")
    
    def __len__(self) -> int:
        return len(self.studies)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        """Get a data sample and its label."""
        study = self.studies[idx]
        return study['images'], study['label']
    
    def get_metadata(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a sample."""
        return self.studies[idx]['metadata']
    
    def _load_studies(self) -> List[Dict]:
        """Load all valid studies with 4 views."""
        studies = []
        
        for client_id in os.listdir(self.config.data_path):
            if not os.path.isdir(os.path.join(self.config.data_path, client_id)):
                continue
                
            # Load client data
            try:
                client_path = os.path.join(self.config.data_path, client_id, f"IMAGEDB_{client_id}.json")
                with open(client_path, 'r') as f:
                    client_data = json.load(f)
            except Exception:
                continue
            
            # Get classification from lesion markings
            classification = self._get_classification(client_data)
            if classification is None:
                continue
            
            # Process each study
            if 'STUDIES' in client_data:
                for study_id, study_data in client_data['STUDIES'].items():
                    if not isinstance(study_data, dict):
                        continue
                    
                    # Try to load all 4 views
                    study = self._process_study(client_id, study_id, classification)
                    if study:
                        studies.append(study)
        
        return studies
    
    def _get_classification(self, client_data: dict) -> Optional[int]:
        """Get binary classification from lesion markings."""
        if 'STUDIES' not in client_data or 'LESIONS' not in client_data:
            return None
        
        has_malignant = False
        has_benign = False
        
        for study_data in client_data['STUDIES'].values():
            if not isinstance(study_data, dict):
                continue
            
            for image_data in study_data.values():
                if not isinstance(image_data, dict):
                    continue
                
                for mark_container in image_data.values():
                    if not isinstance(mark_container, dict):
                        continue
                    
                    for mark_data in mark_container.values():
                        if not isinstance(mark_data, dict):
                            continue
                        
                        # Check for malignant indicators
                        if mark_data.get('Mass') == 1 and mark_data.get('MassClassification') in ['ill_defined', 'spiculated']:
                            has_malignant = True
                        if mark_data.get('SuspiciousCalcifications') or mark_data.get('ArchitecturalDistortion'):
                            has_malignant = True
                        
                        # Check for benign indicators
                        if mark_data.get('BenignClassification') or mark_data.get('FatNecrosis'):
                            has_benign = True
        
        return 1 if has_malignant else (0 if has_benign else None)
    
    def _process_study(self, client_id: str, study_id: str, classification: int) -> Optional[Dict]:
        """Process a study if it has all 4 views."""
        try:
            study_path = os.path.join(self.config.images_path, client_id, study_id)
            if not os.path.exists(study_path):
                return None
            
            # Load NBSS data for metadata
            nbss_path = os.path.join(self.config.data_path, client_id, f"NBSS_{client_id}.json")
            with open(nbss_path, 'r') as f:
                nbss_data = json.load(f)
            
            # Find episode containing this study
            episode_id = None
            for ep_id, episode in nbss_data.items():
                if isinstance(episode, dict) and 'StudyList' in episode:
                    if study_id in str(episode['StudyList']):
                        episode_id = ep_id
                        break
            
            if not episode_id:
                return None
                
            views = {}
            # Initialize metadata fields
            metadata = {
                # Technical/Equipment Factors
                'manufacturer': None,
                'equipment_model': None,
                'equipment_id': None,
                'bits_stored': None,
                'photometric_interpretation': None,
                'processing_type': None,
                
                # Patient Demographics & Clinical Context
                'patient_age_at_screening': None,
                'patient_age': nbss_data[episode_id].get('PatientAge'),
                'study_date': None,
                'episode_type': nbss_data[episode_id].get('EpisodeType'),
                'episode_action': nbss_data[episode_id].get('EpisodeAction'),
                'actual_episode_year': nbss_data[episode_id].get('ActualEpisodeOpenedYear'),
                
                # Image Properties
                'image_laterality': [],
                'view_positions': [],
                
                # Clinical Assessment
                'screening_opinions': {
                    'left': nbss_data[episode_id].get('SCREENING', {}).get('left_opinion'),
                    'right': nbss_data[episode_id].get('SCREENING', {}).get('right_opinion'),
                    'overall': nbss_data[episode_id].get('SCREENING', {}).get('Opinion')
                },
                'diagnostic_outcome': nbss_data[episode_id].get('SCREENING', {}).get('diagnosticsetoutcome'),
                
                # Reader Information
                'film_readers': set(),  # Will collect all unique reader IDs
                'number_of_readers': 0,
                
                # Site Information
                'site': None,  # Will be extracted from IMAGEDB
                
                # Additional Clinical Findings
                'has_mass': False,
                'has_calcification': False,
                'has_architectural_distortion': False,
                'lesion_conspicuity': set()  # Will collect conspicuity levels of any lesions
            }
            
            # Load IMAGEDB data for additional metadata
            try:
                imagedb_path = os.path.join(self.config.data_path, client_id, f"IMAGEDB_{client_id}.json")
                with open(imagedb_path, 'r') as f:
                    imagedb_data = json.load(f)
                metadata['site'] = imagedb_data.get('Site')
                
                # Extract lesion information if available
                study_data = imagedb_data.get('STUDIES', {}).get(study_id, {})
                for series in study_data.values():
                    if isinstance(series, dict):
                        for image in series.values():
                            if isinstance(image, dict):
                                for mark in image.values():
                                    if isinstance(mark, dict):
                                        if mark.get('Mass'):
                                            metadata['has_mass'] = True
                                        if mark.get('SuspiciousCalcifications'):
                                            metadata['has_calcification'] = True
                                        if mark.get('ArchitecturalDistortion'):
                                            metadata['has_architectural_distortion'] = True
                                        if mark.get('Conspicuity'):
                                            metadata['lesion_conspicuity'].add(mark['Conspicuity'])
            except Exception:
                pass  # Continue without IMAGEDB metadata if file can't be read
            
            for dicom_file in os.listdir(study_path):
                if not dicom_file.endswith('.dcm'):
                    continue
                
                dcm = pydicom.dcmread(os.path.join(study_path, dicom_file))
                if not (hasattr(dcm, 'ImageLaterality') and hasattr(dcm, 'ViewPosition')):
                    continue
                
                # Skip raw images
                if hasattr(dcm, 'PresentationIntentType') and dcm.PresentationIntentType != "FOR PRESENTATION":
                    continue
                
                # Map view position
                lat, pos = dcm.ImageLaterality, dcm.ViewPosition
                if lat == 'L' and pos == 'CC':
                    view = 'l_cc'
                elif lat == 'L' and pos == 'MLO':
                    view = 'l_mlo'
                elif lat == 'R' and pos == 'CC':
                    view = 'r_cc'
                elif lat == 'R' and pos == 'MLO':
                    view = 'r_mlo'
                else:
                    continue
                
                if view not in views:
                    views[view] = dcm
                    
                    # Extract metadata from first view's NBSS data
                    if metadata['manufacturer'] is None and episode_id in nbss_data:
                        screening = nbss_data[episode_id].get('SCREENING', {})
                        if screening:
                            # Try to get equipment info from either L or R view
                            l_view = screening.get('L', {})
                            r_view = screening.get('R', {})
                            metadata.update({
                                'manufacturer': (l_view.get('EquipmentMakeModel') or 
                                              r_view.get('EquipmentMakeModel')),
                                'equipment_id': (l_view.get('EquipmentUsed') or 
                                             r_view.get('EquipmentUsed')),
                                'patient_age_at_screening': (l_view.get('AgeAtScreening') or 
                                                          r_view.get('AgeAtScreening'))
                            })
                            
                            # Collect reader information
                            for side in [l_view, r_view]:
                                if side.get('FilmReaders'):
                                    metadata['film_readers'].update(side['FilmReaders'].split())
                    
                    # Extract DICOM metadata
                    if metadata['study_date'] is None:
                        metadata.update({
                            'study_date': getattr(dcm, 'StudyDate', None),
                            'bits_stored': getattr(dcm, 'BitsStored', None),
                            'photometric_interpretation': getattr(dcm, 'PhotometricInterpretation', None),
                            'processing_type': getattr(dcm, 'PresentationIntentType', None)
                        })
                    
                    # Collect all lateralities and view positions
                    if lat not in metadata['image_laterality']:
                        metadata['image_laterality'].append(lat)
                    if pos not in metadata['view_positions']:
                        metadata['view_positions'].append(pos)
            
            # Post-process metadata
            metadata['film_readers'] = list(metadata['film_readers'])  # Convert set to list
            metadata['number_of_readers'] = len(metadata['film_readers'])
            metadata['lesion_conspicuity'] = list(metadata['lesion_conspicuity'])  # Convert set to list
            
            # Check if we have all views
            required_views = ['l_cc', 'l_mlo', 'r_cc', 'r_mlo']
            if not all(view in views for view in required_views):
                return None
            
            # Load images
            images = {}
            for view, dcm in views.items():
                img = dcm.pixel_array.astype(float)
                img = (img - img.min()) / (img.max() - img.min())  # Simple normalization
                img = torch.from_numpy(img).float().unsqueeze(0)  # Add channel dimension
                images[view] = img
            
            return {
                'images': images,
                'label': classification,
                'metadata': metadata
            }
            
        except Exception:
            return None

def main():
    """Simple test."""
    config = OptimamConfig(
        data_path='/Users/hendrik/Studium/Master/Thesis/Data/OMI-DB Sample/DATA',
        images_path='/Users/hendrik/Studium/Master/Thesis/Data/OMI-DB Sample/IMAGES'
    )
    
    dataset = OptimamDataset(config)
    if len(dataset) > 0:
        images, label = dataset[0]
        metadata = dataset.get_metadata(0)
        print(f"\nFirst item:")
        for view, img in images.items():
            print(f"{view}: {img.shape}")
        print(f"Label: {label}")
        print(f"Metadata: {metadata}")

if __name__ == "__main__":
    main() 