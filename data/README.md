# Download instructions

## Test data
If you are only interested in inference, you can download just the Slakh2100 test-set that we used. It is available at the following [link](https://drive.google.com/file/d/1Xo-bGORndJhenHvzf3eY5Lt0XCTgElZP/view?usp=sharing).
```bash
# Move into this directory
cd data/

# Extract dataset here
tar -xvf slakh2100-testset-22050.tar.xz
```
The test set alone should occupy around 7GB of memory. 

## Complete dataset (train, validation, and test sets)
If you are interested in training some models of your own, you need to download the complete dataset.

Instructions for downloading are the following:

 1. Download the compressed data for each stem (bass, drums, etc.)
 2. Extract the data in this folder (i. e. `data/`)
 3. [optional] Delete the compressed data 
 4. Run the shell script `convert_data_format.sh` 

In the sections below, you can find a more precise description for each of these steps.

### 1. Download compressed data
You can download the data we used in our experiments from the following links:
 
 - [Bass Data](https://drive.google.com/file/d/1T7rbuwyqR73K__0L3nF550rVBXgrpYVT/view?usp=sharing)
 - [Drums Data](https://drive.google.com/file/d/1vieJQdvN22YrTdBMMvZXw1xr1rdko1pm/view?usp=sharing)
 - [Guitar Data](https://drive.google.com/file/d/1Uo3iN4lIecJ8SJulEKlhD96Bgd2CGf8F/view?usp=sharing)
 - [Piano Data](https://drive.google.com/file/d/1w3Zou4oL_DfdJm1o_Y-qLCvVYG8W6J62/view?usp=sharing)

Move the downloaded files into the `data/` directory.

```
data/bass_22050.tar.xz
data/drums_22050.tar.xz
data/guitar_22050.tar.xz
data/piano_22050.tar.xz
```

### 2. Extract data
```bash
# Move inside this directory
cd data/

# Decompress and extract data
tar -xvf bass_22050.tar.xz 
tar -xvf drums_22050.tar.xz
tar -xvf guitar_22050.tar.xz
tar -xvf piano_22050.tar.xz
```
This step might take a while, especially depending on your hardware. If you have a fast internet connection, consider instead downloading the zipped versions from [here](https://drive.google.com/drive/folders/1lCr93-47J3lsm_X5sBWGc9J1UfAz9pJE?usp=sharing).

After the extraction of all the sources dataset, you should have four directories:
```
data/bass_22050/
data/drums_22050/
data/guitar_22050/
data/piano_22050/
```

### 3. Delete compressed data
To free up some space it is possible now to delete the compressed version of the data. It will no longer be necessary.
```
rm data/*_22050.tar.xz 
```

### 4. Convert data
Before being able to use the dataset for training, it is necessary to run the following command:
```bash
# Move inside this directory
cd data/

 # Make script executable 
chmod +x ./convert_data_format.sh

# Convert the format of your data
./convert_data_format.sh
```
This command will convert the downloaded data into a format that the training script can digest. In particular, after running everything, your `data/` directory should contain the `slakh2100` folder, organized in the following fashion:
```
data/
 └─── slakh2100/
       └─── train/
             └─── Track00001/
                   └─── bass.wav
                   └─── drums.wav
                   └─── guitar.wav
                   └─── piano.wav
            ...
      ...
```
> ⚠️ **NOTE:**
> After running the script, the space occupied by `data/` should not change drastically, since all the files are hard-links, and are not actually copied.

