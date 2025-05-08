# Data Restructuring for Neural Network Training

## Overview

To transform the hierarchical OPTIMAM data into a format suitable for neural network training, we need to create a flattened structure where:

- Input (X): 4 synchronized mammogram views (L-CC, L-MLO, R-CC, R-MLO)
- Output (Y): Binary label (0 = Not Malignant, 1 = Malignant)

## Proposed Data Structure

```
training_data/
├── metadata.csv          # Contains all sample metadata
├── images/              # Preprocessed images
│   └── exam_<id>/       # One directory per exam
│       ├── L_CC.png     # Left Cranio-Caudal view
│       ├── L_MLO.png    # Left Medio-Lateral Oblique view
│       ├── R_CC.png     # Right Cranio-Caudal view
│       └── R_MLO.png    # Right Medio-Lateral Oblique view
└── labels.csv           # Contains binary labels
```

### metadata.csv Structure

```csv
exam_id,client_id,episode_id,study_id,date,left_breast_label,right_breast_label,age,manufacturer
exam_001,demd1457,9994,1.2.826...,2021-01-01,0,1,54,HOLOGIC
...
```

### labels.csv Structure

```csv
exam_id,label
exam_001,1
exam_002,0
...
```

## Data Processing Pipeline

1. **Extract Complete Exams**

```python
@dataclass
class Exam:
    exam_id: str
    client_id: str
    episode_id: str
    study_id: str
    images: Dict[str, Path]  # View -> Image path
    left_breast_label: int   # 0 or 1
    right_breast_label: int  # 0 or 1
    metadata: Dict[str, Any]
```

2. **Image Preprocessing**

```python
def preprocess_exam(exam: Exam) -> ProcessedExam:
    """
    1. Load DICOM images
    2. Convert to standardized format (PNG)
    3. Apply preprocessing:
       - Resize to standard dimensions
       - Normalize pixel values
       - Apply breast segmentation
       - Standardize orientation
    4. Save in training_data/images/exam_<id>/
    """
    pass
```

3. **Label Extraction**

```python
def extract_label(exam: Exam) -> int:
    """
    Convert episode-level labels to binary classification:
    - If either breast is malignant -> 1
    - If both breasts are benign -> 0
    """
    return int(exam.left_breast_label == 1 or exam.right_breast_label == 1)
```

## PyTorch DataLoader Implementation

```python
class MammographyDataset(Dataset):
    def __init__(self, data_dir: Path, transform=None):
        self.data_dir = data_dir
        self.metadata = pd.read_csv(data_dir / "metadata.csv")
        self.labels = pd.read_csv(data_dir / "labels.csv")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        exam_id = self.metadata.iloc[idx]["exam_id"]

        # Load all 4 views
        images = {}
        for view in ["L_CC", "L_MLO", "R_CC", "R_MLO"]:
            img_path = self.data_dir / "images" / exam_id / f"{view}.png"
            img = Image.open(img_path)
            if self.transform:
                img = self.transform(img)
            images[view] = img

        # Get label
        label = self.labels.iloc[idx]["label"]

        return images, label

# Usage Example
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = MammographyDataset(
    data_dir=Path("training_data"),
    transform=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4
)
```

## Key Considerations

1. **Label Definition**

   - Binary classification (0/1)
   - Malignant if either breast is malignant
   - Consider confidence scores from multiple readers

2. **Data Balance**

   - Usually imbalanced (more normal cases)
   - Consider weighted sampling
   - Implement appropriate loss functions

3. **Image Preprocessing**

   - Consistent image dimensions
   - Standardized orientation
   - Proper normalization
   - Breast segmentation

4. **Data Splitting**

   - Split by patient, not by exam
   - Ensure no patient overlap between train/val/test
   - Maintain class distribution in splits

5. **Validation Strategy**
   - Use multiple reader opinions as soft labels
   - Compare with radiologist performance
   - Consider AUROC as primary metric

## Memory Considerations

1. **Lazy Loading**

   - Load images on demand
   - Use memory mapping for large datasets
   - Implement caching if needed

2. **Preprocessing**

   - Preprocess images once
   - Save in efficient format
   - Consider compression

3. **Batch Size**
   - 4 images per exam
   - Account for GPU memory
   - Use gradient accumulation if needed
