# CymruFluency Dataset - README

## Dataset Title
**CymruFluency**

## Description
The CymruFluency dataset contains multi-modal 3D face and visual speech data collected from 33 participants. Each participant's data includes:

- `.obj` files: 3D face meshes per frame  
- `.png` images: frame-wise textures  
- `.wav` audio files: speech recordings  
- `.ljson` files: facial landmarks in iBug68 format

`Fluency-score-table.xlsx` contains the fluency score for each utterance of every participant.

This dataset is suitable for research in:

- Visual speech synthesis  
- Talking head generation  
- Fluency assessment  
- Multimodal learning

## Dataset Components
The dataset is split into the following archives for easier distribution:

1. `CymruFluency-part1.zip` → participants 01–12  
2. `CymruFluency-part2.zip` → participants 13–22  
3. `CymruFluency-part3.zip` → participants 23–27  
4. `CymruFluency-part4.zip` → participants 28–33  
5. `CymruFluency-Audios.zip` → `.wav` files for all participants  
6. `CymruFluency-Landmarks.zip` → `.ljson` landmark files archived in `.7z` format  
7. `CymruFluency-main.zip` → project code.

Each participant folder is named consistently and contains 10 utterance-specific archives (either `.7z` or raw `.wav` depending on the archive).

## Dataset Contents
The dataset is organized as follows:

```
dataset/
├── 1/
│ ├── 1.7z
│ ├── 2.7z
│ └── ...
├── 2/
│ ├── 1.7z
│ └── ...
├── ...
├── 33/
│ └── ...
```


Each `.7z` archive contains:

- A sequence of `.obj` files (3D landmark meshes per frame)  
- Corresponding `.png` images (frame-aligned)

## Instructions

1. Download all four `.zip` parts.  
2. Extract each `.zip` to the same `dataset/` directory.  
3. Ensure that all subfolders (`participant_01` to `participant_33`) are present after extraction.

**Note:** Audio and landmark zips contain corresponding data from all participants. Simply extract them into the parent directory.

## Citation

If you use this dataset in your research or publications, please cite it using the following concept DOI:  
[https://doi.org/10.5281/zenodo.15397513](https://doi.org/10.5281/zenodo.15397513)

## Contact

For questions or collaborations, contact the dataset authors.
