# DATASET

## image
image resolution 
- 2048 x 2048
- JPEG
- GRAYSCALE

## annotation dict
Annotation File Structure

JSON is a dictionary

{
"info": info,
"images": [image],
"annotations": [annotation],
"licenses": [license],
"categories": [category],
}

info{
"year": int,
"version": str,
"description": str,
"contributor": str,
"url": str,
"date_created": datetime,
}

license{
"id": int,
"name": str,
"url": str,
}

categories[
{
   "id": int,
   "name": str, -> left, right, unidentifiable, ambigious
   "supercategory": str,  ->always filament
}]

annotation{
"id": str,
"image_id": int,
"category_id": int,
"segmentation": [[]],   ->polygon coordinates describing the filament boundary
                        coordinates are flattened, read as (x, y) pairs
"area": float,          -> in px^2
"spine": [],            -> coordinates of the filament's central spine
"bbox": [x, y, width, height], ->bounding box around the filament
"iscrowd": 0 or 1,
}

image{
"id": str,
"width": int,
"height": int,
"file_name": str,
"license": int,
"date_captured": datetime,
}

# Comp Features

Fine structures / barbs matter
Metric: PQ
IoU matching threshold: > 0.5
PQ penalizes:
FP
FN
fragmentation
over-merging
Important secondary metrics:
Dice
IoU
one-to-many
many-to-one


# Dataset Findings

~42% observations have 2–3 annotator passes
Same physical observation can have multiple COCO image IDs
Validation should group by physical observation
Annotator matched IoU ≈ 0.52
Instance F1@0.5 ≈ 0.54
Annotation noise → real performance ceiling
More-complete annotation often ≈ superset of smaller annotation


# Preprocessing

## Disk Extraction
- Used Global Otsu and got a threshold around  71
- Got largest component which is a circle
- Had 1 or 2 tiny holes
- Used morphological closing with 15x15 kernel to remove holes but disk ends up bigger so keeping this away for now

## Intensity Normalization

- All JPEG values are 0–255 but images have different sources and taken from different observatories
- so the overall brightness can be different. very bright could mean 250 in one but 200 in another
- Use per-image normalization inside disk
- normalization done on disk only so the outer dark part doesnt effect
- Median = typical disk intensity
- MAD = robust intensity spread
- normalized = (image - median) / MAD
- Normalized image kept as NumPy array, original JPEG unchanged

## Resolution

- Whole-image resizing risks losing thin filaments/barbs
- Median filament becomes ~13 px at 512x512
- Median filament becomes ~27 px at 1024x1024
- Use full-resolution overlapping patches
- No aggressive whole-image resizing



# Classical CV Experiments

## Filament Thresholding

- Applied second Otsu inside disk
- Produced crude filament mask
- Many noisy regions and very fragmented
- the outer darker layer of the disk became white
- Worse than raw grayscale

## Soft Threshold

- Tried soft version around Otsu threshold
- Still not better than raw grayscale and even worse than thresholding

## Adaptive Threshold

- Tried local/adaptive thresholding
- Very noisy
- Detected many local intensity fluctuations

## Gaussian Background Subtraction

- Tried background subtraction using Gaussian blur
- Intended to reduce broad brightness variation / limb darkening
- Produced noise/speckle
- Not useful enough

## Connected Components

- Used connected components on binary filament mask
- Produced many tiny components about 4000
- Strong fragmentation
- Many predictions had poor overlap with GT



