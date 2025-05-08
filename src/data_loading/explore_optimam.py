from pathlib import Path
from typing import Dict, Any
from collections import defaultdict
import omidb
import pydicom
import numpy as np
from optimam_explorer import OptimamExplorer as BaseOptimamExplorer, DatabaseConfig
import json

class OptimamExplorer(BaseOptimamExplorer):
    """Extended OptimamExplorer with additional exploration methods"""
    
    def analyze_dataset_statistics(self) -> Dict[str, Any]:
        """
        Analyze basic statistics about the dataset structure.
        
        Counts entities at each level of the hierarchy:
        - Clients (patients)
        - Episodes (clinical events/procedures) and their types
        - Studies (imaging examinations)
        - Series (position/view specific image sets)
        - Images (individual mammograms)
        
        Returns:
            Dict containing statistics about the dataset structure
        """
        stats = {
            'total_clients': 0,
            'total_episodes': 0,
            'episodes_without_studies': 0,  # Episodes without imaging examinations
            'episodes_without_studies_but_with_screening': 0,  # Episodes with screening info but no studies
            'total_studies': 0,
            'studies_without_series': 0,    # Studies without any positions/views
            'total_series': 0,
            'series_without_images': 0,     # Series without actual images
            'total_images': 0,
            'problematic_images': 0         # Images with metadata issues
        }
        
        view_counts = {}          # Distribution of mammographic views
        manufacturer_counts = {}   # Distribution of imaging equipment
        episodes_by_study_count = {}  # How many studies per episode
        episode_types = {}        # Distribution of episode types
        episode_actions = {}      # Distribution of episode actions
        episodes_without_studies_by_type = {}  # Episodes without studies by type
        example_episodes_without_studies = []  # Example episode IDs without studies
        episodes_without_studies_but_with_screening = []  # Special cases like 9999
        
        for client in self.get_clients():
            stats['total_clients'] += 1
            
            for episode in client.episodes:
                stats['total_episodes'] += 1
                
                # Track episode types and actions
                if episode.type:
                    episode_type = f"{episode.type.value} ({episode.type.name})"
                    episode_types[episode_type] = episode_types.get(episode_type, 0) + 1
                if episode.action:
                    episode_action = f"{episode.action.value} ({episode.action.name})"
                    episode_actions[episode_action] = episode_actions.get(episode_action, 0) + 1
                
                # Count imaging examinations per clinical event
                study_count = len(episode.studies) if episode.studies else 0
                episodes_by_study_count[study_count] = episodes_by_study_count.get(study_count, 0) + 1
                
                if not episode.studies:
                    stats['episodes_without_studies'] += 1
                    # Track by type
                    episode_type_key = episode_type if episode.type else "UNKNOWN"
                    episodes_without_studies_by_type[episode_type_key] = episodes_without_studies_by_type.get(episode_type_key, 0) + 1
                    
                    # Check if episode has screening info despite no studies
                    has_screening = hasattr(episode, 'SCREENING') and episode.SCREENING
                    has_equipment = (hasattr(episode.SCREENING, 'L') and episode.SCREENING.L.get('EquipmentMakeModel')) or \
                                  (hasattr(episode.SCREENING, 'R') and episode.SCREENING.R.get('EquipmentMakeModel')) \
                                  if has_screening else False
                    
                    if has_screening and has_equipment:
                        stats['episodes_without_studies_but_with_screening'] += 1
                        episodes_without_studies_but_with_screening.append({
                            'client_id': client.id,
                            'episode_id': episode.id,
                            'type': episode_type_key,
                            'equipment': episode.SCREENING.L.get('EquipmentMakeModel') or episode.SCREENING.R.get('EquipmentMakeModel'),
                            'date_taken': episode.SCREENING.L.get('DateTaken') or episode.SCREENING.R.get('DateTaken')
                        })
                    
                    # Collect example IDs (limit to 3)
                    if len(example_episodes_without_studies) < 3:
                        example_episodes_without_studies.append({
                            'client_id': client.id,
                            'episode_id': episode.id,
                            'type': episode_type_key
                        })
                    continue
                    
                for study in episode.studies:
                    stats['total_studies'] += 1
                    
                    if not study.series:
                        stats['studies_without_series'] += 1
                        continue
                        
                    stats['total_series'] += len(study.series)
                    
                    for series in study.series:
                        if not series.images:
                            stats['series_without_images'] += 1
                            continue
                            
                        for image in series.images:
                            stats['total_images'] += 1
                            try:
                                view = self.get_view_from_dcm(image.dcm)
                                view_counts[view] = view_counts.get(view, 0) + 1
                                manufacturer = image.dcm.Manufacturer
                                manufacturer_counts[manufacturer] = manufacturer_counts.get(manufacturer, 0) + 1
                            except Exception as e:
                                stats['problematic_images'] += 1
                                print(f"Error processing image: {e}")
        
        return {
            'basic_stats': stats,
            'view_distribution': view_counts,
            'manufacturer_distribution': manufacturer_counts,
            'episodes_by_study_count': episodes_by_study_count,
            'episode_types': episode_types,
            'episode_actions': episode_actions,
            'episodes_without_studies_analysis': {
                'by_type': episodes_without_studies_by_type,
                'examples': example_episodes_without_studies,
                'with_screening_info': episodes_without_studies_but_with_screening[:5]  # Show first 5 examples
            }
        }
    
    def explore_example_exam(self) -> Dict[str, Any]:
        """
        Explore details of complete imaging examinations.
        
        Examines the relationship between:
        - Studies (complete imaging examinations)
        - Series (specific positions/views)
        - Images (actual mammograms)
        
        Returns:
            Dict containing detailed information about example examinations
        """
        examples = []
        
        for client in self.get_clients():
            studies = self.get_studies_with_complete_views(client)
            if not studies:
                continue
                
            for study in studies:  # 
                study_info = {
                    'client_id': client.id,
                    'study_id': study.id,
                    'series_info': [],
                    'total_series': len(study.series) if study.series else 0
                }
                
                if not study.series:
                    continue
                    
                for series in study.series:
                    if not series.images:
                        continue
                        
                    img = series.images[0]
                    try:
                        # Extract key series information
                        series_info = {
                            'series_id': series.id,
                            'view_position': getattr(img.dcm, 'ViewPosition', 'N/A'),
                            'laterality': getattr(img.dcm, 'ImageLaterality', 'N/A'),
                            'view_description': getattr(img.dcm, 'SeriesDescription', 'N/A'),
                            'processing_type': getattr(img.dcm, 'PresentationIntentType', 'N/A'),
                            'image_size': img.dcm.pixel_array.shape if hasattr(img.dcm, 'pixel_array') else None,
                            'manufacturer': getattr(img.dcm, 'Manufacturer', 'N/A'),
                            'photometric_interpretation': getattr(img.dcm, 'PhotometricInterpretation', 'N/A'),
                            'bits_stored': getattr(img.dcm, 'BitsStored', 'N/A')
                        }
                        study_info['series_info'].append(series_info)
                        
                    except Exception as e:
                        print(f"Error processing series metadata: {e}")
                        continue
                
                if study_info['series_info']:  # Only add if we found valid series
                    examples.append(study_info)
                    
                if len(examples) >= 3:  # Limit to 3 example studies total
                    break
            
            if len(examples) >= 3:
                break
        
        return examples

def main():
    """Explore the OPTIMAM dataset and print findings."""
    config = DatabaseConfig(
        data_path=Path('/Users/hendrik/Studium/Master/Thesis/Data/OMI-DB Sample/DATA'),
        images_path=Path('/Users/hendrik/Studium/Master/Thesis/Data/OMI-DB Sample/IMAGES'),
        output_path=Path('/Users/hendrik/Studium/Master/Thesis/Code/breast_cancer_classifier/omi_db_data')
    )
    
    explorer = OptimamExplorer(config)
    
    # Analyze dataset statistics
    print("\n=== Dataset Statistics ===")
    stats = explorer.analyze_dataset_statistics()
    print(json.dumps(stats, indent=2))
    
    # Explore example exams
    print("\n=== Example Studies and Series Details ===")
    examples = explorer.explore_example_exam()
    print(json.dumps(examples, indent=2))

if __name__ == "__main__":
    main() 