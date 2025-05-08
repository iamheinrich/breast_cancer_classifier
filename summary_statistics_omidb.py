import matplotlib.pyplot as plt
from collections import Counter
import omidb

OMIDB_DATA_PATH = '/Users/hendrik/Studium/Master/Thesis/Data/OMI-DB Sample/DATA'
OMIDB_IMAGES_PATH = '/Users/hendrik/Studium/Master/Thesis/Data/OMI-DB Sample/IMAGES'
db = omidb.DB(OMIDB_DATA_PATH, OMIDB_IMAGES_PATH, ignore_missing_images=False)

clients = [client for client in db]
[client.id for client in clients]


def study_is_screening_event(study):
    if omidb.events.Event.screening in study.event_type:
        return True
    else:
        return False


def get_studies_with_at_least_four_series(episodes):
    studies_with_at_least_four_series = []
    for episode in episodes:
        if episode.studies:
            for study in episode.studies:
                if study.series:
                    if len(study.series) >= 4:
                        studies_with_at_least_four_series.append(study.id)
        else:
            continue
    return studies_with_at_least_four_series


def get_view_from_dcm(dcm):
    laterality = dcm.ImageLaterality
    view_position = dcm.ViewPosition
    full_view = f"{laterality}-{view_position}"
    return full_view


def get_presentation_intent_type_from_dcm(dcm):
    return dcm.PresentationIntentType


def get_image_id_from_dcm(dcm):
    return dcm.SOPInstanceUID


def create_img_path(client_id, study_id, img_id):
    return f"{client_id}/{study_id}/{img_id}.dcm"


def count_images(clients):
    statistics = {
        'total_images': 0,
        'presentation_images': 0,
        'complete_studies': 0,
        'studies_with_4_series': 0
    }

    for client in clients:
        for episode in client.episodes:
            for study in episode.studies:
                if len(study.series) >= 4:
                    statistics['studies_with_4_series'] += 1

                    # Track views found in this study
                    views_found = set()

                    for series in study.series:
                        for image in series.images:
                            statistics['total_images'] += 1

                            # Count presentation images
                            if hasattr(image, 'dcm'):
                                if get_presentation_intent_type_from_dcm(image.dcm) == "FOR PRESENTATION":
                                    statistics['presentation_images'] += 1
                                    views_found.add(
                                        get_view_from_dcm(image.dcm))

                    # Check if study has all required views
                    if {'L-CC', 'L-MLO', 'R-CC', 'R-MLO'}.issubset(views_found):
                        statistics['complete_studies'] += 1

        print(f"Processed client {client.id}")

    return statistics


def analyze_study_distribution(clients):
    study_stats = {
        'studies_per_client': {},
        'total_studies': 0,
        'max_studies': 0,
        'min_studies': float('inf'),
        'client_with_max': '',
        'client_with_min': ''
    }

    for client in clients:
        study_count = sum(len(episode.studies) for episode in client.episodes)
        study_stats['studies_per_client'][client.id] = study_count
        study_stats['total_studies'] += study_count

        # Update max/min statistics
        if study_count > study_stats['max_studies']:
            study_stats['max_studies'] = study_count
            study_stats['client_with_max'] = client.id
        if study_count < study_stats['min_studies']:
            study_stats['min_studies'] = study_count
            study_stats['client_with_min'] = client.id

    # Calculate average
    num_clients = len(clients)
    study_stats['avg_studies'] = study_stats['total_studies'] / \
        num_clients if num_clients > 0 else 0

    return study_stats


def analyze_series_distribution(clients):
    series_stats = {
        'max_series': 0,
        'min_series': float('inf'),
        'total_series': 0,
        'total_studies': 0,
        'study_with_max': '',
        'study_with_min': '',
        'client_with_max': '',
        'client_with_min': ''
    }

    for client in clients:
        for episode in client.episodes:
            for study in episode.studies:
                series_count = len(study.series)
                series_stats['total_series'] += series_count
                series_stats['total_studies'] += 1

                # Update max/min statistics
                if series_count > series_stats['max_series']:
                    series_stats['max_series'] = series_count
                    series_stats['study_with_max'] = study.id
                    series_stats['client_with_max'] = client.id
                if series_count < series_stats['min_series']:
                    series_stats['min_series'] = series_count
                    series_stats['study_with_min'] = study.id
                    series_stats['client_with_min'] = client.id

    # Calculate average
    series_stats['avg_series'] = (series_stats['total_series'] /
                                  series_stats['total_studies'] if series_stats['total_studies'] > 0 else 0)

    return series_stats


def analyze_images_per_series_distribution(clients):
    image_stats = {
        'max_images': 0,
        'min_images': float('inf'),
        'total_images': 0,
        'total_presentation_images': 0,
        'total_series': 0,
        'series_with_max': '',
        'series_with_min': '',
        'study_with_max': '',
        'study_with_min': '',
        'client_with_max': '',
        'client_with_min': ''
    }
    
    for client in clients:
        for episode in client.episodes:
            for study in episode.studies:
                # Only process studies with 4 or more series
                if len(study.series) >= 4:
                    for series in study.series:
                        # Count all images and presentation images separately
                        image_count = 0
                        for image in series.images:
                            image_stats['total_images'] += 1
                            if hasattr(image, 'dcm'):
                                if get_presentation_intent_type_from_dcm(image.dcm) == "FOR PRESENTATION":
                                    image_count += 1
                                    image_stats['total_presentation_images'] += 1
                        
                        image_stats['total_series'] += 1
                        
                        # Update max/min statistics using presentation images count
                        if image_count > image_stats['max_images']:
                            image_stats['max_images'] = image_count
                            image_stats['series_with_max'] = series.id
                            image_stats['study_with_max'] = study.id
                            image_stats['client_with_max'] = client.id
                        if image_count < image_stats['min_images'] and image_count > 0:
                            image_stats['min_images'] = image_count
                            image_stats['series_with_min'] = series.id
                            image_stats['study_with_min'] = study.id
                            image_stats['client_with_min'] = client.id
    
    # Calculate average based on presentation images
    image_stats['avg_images'] = (image_stats['total_presentation_images'] / 
                                image_stats['total_series'] if image_stats['total_series'] > 0 else 0)
    
    return image_stats


def main():
    stats = count_images(clients)
    study_stats = analyze_study_distribution(clients)
    series_stats = analyze_series_distribution(clients)
    image_stats = analyze_images_per_series_distribution(clients)

    print("\nSummary Statistics:")
    print(f"Total images: {stats['total_images']}")
    print(f"Presentation images: {stats['presentation_images']}")
    print(f"Studies with 4+ series: {stats['studies_with_4_series']}")
    print(f"Complete studies: {stats['complete_studies']}")

    print("\nStudy Distribution Statistics:")
    print(f"Total number of studies: {study_stats['total_studies']}")
    print(f"Average studies per client: {study_stats['avg_studies']:.2f}")
    print(f"Maximum studies: {study_stats['max_studies']} (Client: {
          study_stats['client_with_max']})")
    print(f"Minimum studies: {study_stats['min_studies']} (Client: {
          study_stats['client_with_min']})")

    print("\nSeries Distribution Statistics:")
    print(f"Total number of series: {series_stats['total_series']}")
    print(f"Average series per study: {series_stats['avg_series']:.2f}")
    print(f"Maximum series: {series_stats['max_series']} "
          f"(Study: {series_stats['study_with_max']}, Client: {series_stats['client_with_max']})")
    print(f"Minimum series: {series_stats['min_series']} "
          f"(Study: {series_stats['study_with_min']}, Client: {series_stats['client_with_min']})")

    print("\nImages per Series Distribution Statistics:")
    print(f"Total number of images: {image_stats['total_images']}")
    print(f"Average images per series: {image_stats['avg_images']:.2f}")
    print(f"Maximum images: {image_stats['max_images']} "
          f"(Series: {image_stats['series_with_max']}, "
          f"Study: {image_stats['study_with_max']}, "
          f"Client: {image_stats['client_with_max']})")
    print(f"Minimum images: {image_stats['min_images']} "
          f"(Series: {image_stats['series_with_min']}, "
          f"Study: {image_stats['study_with_min']}, "
          f"Client: {image_stats['client_with_min']})")


if __name__ == "__main__":
    main()
