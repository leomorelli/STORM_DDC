

import numpy as np
import random
import scipy.io as sio
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
import math
import time
from numpy import ceil, quantile
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
from scipy.io import savemat



#### 1) LOAD DATA ####

# Initialize variables
TrueLocalizations = []
Photons = []
Resolution = []
min_loc = 2000

# Use tkinter file dialog to mimic uigetfile
# This is useful to upload the mat file we need
Tk().withdraw()
filename_full = askopenfilename(filetypes=[("MAT files", "*.mat")], title="Select HMM .mat file")
if not filename_full:
    print('Error! No (or wrong) file selected!')
    exit()

# Separate pathname and filename
pathname, filename_only = os.path.split(filename_full)

# This will load in all of the localizations from your structure file
full_filename = os.path.join(pathname, filename_only)
try:
    data = sio.loadmat(full_filename, squeeze_me=True, struct_as_record=False)
except Exception as e:
    print(f"Error loading MAT file: {e}")
    exit()

# Better handling of loaded data - handle both lists and numpy arrays
if 'LocalizationsFinal' in data:
    #check if there is only one ROI or not
    if type(data['LocalizationsFinal'][0][0]) == np.ndarray:
        LocalizationsFinal = data['LocalizationsFinal']
    else:
        LocalizationsFinal = np.array([data['LocalizationsFinal']])
else:
    LocalizationsFinal = []


if 'Frame_Information' in data:
    #check if there is only one ROI or not
    if type(data['Frame_Information'][0]) == np.ndarray:
        Frame_Information = data['Frame_Information']
    else:
        Frame_Information = np.array([data['Frame_Information']])
else:
    Frame_Information = []

if 'TrueLocalizations' in data:
    #check if there is only one ROI or not
    if type(data['TrueLocalizations'][0][0]) == np.ndarray:
        TrueLocalizations = data['TrueLocalizations']
    else:
        TrueLocalizations = np.array([data['TrueLocalizations']])
else:
    TrueLocalizations = []

if 'Photons' in data:
    #check if there is only one ROI or not
    if type(data['Photons'][0]) == np.ndarray:
        Photons = data['Photons']
    else:
        Photons = np.array([data['Photons']])
else:
    Photons = []

if 'Resolution' in data:
    Resolution = data['Resolution']
else:
    Resolution = []

Condition = filename_only




# If photons cell does not exist, generate one
if len(Photons) == 0 or Photons is None:
    Photons = []  # reinitialize as an empty list
    for i in range(len(Frame_Information)):
        # Add array of one to the list for each frame in photons
        Photons.append(np.ones((len(Frame_Information[i]), 1), dtype=int))

#### 2) SPLITTING ####

# Initialize arrays to store split up images
LocalizationsFinal_Split = []
TrueLocalizations_Split = []
Frame_Information_Split = []
Photons_Split = []
Came_from_image = []
Parameters_to_split = []
temp_numb_of_loc = []
addonarray = []
cut1array = []
cut2array = []
cut3array = []




# Just in case you do not actually know the true localizations
if len(TrueLocalizations) == 0:
    TrueLocalizations = []
    for ijk in range(len(Frame_Information)):
        TrueLocalizations.append(np.empty((0, 3)))  # Empty array with 3 columns

# Process the localizations to ensure 3D format
processed_localizations = []
processed_true_localizations = []

# Print debug info about loaded data
print(f"Number of frames: {len(LocalizationsFinal)}")
print(LocalizationsFinal)


# Adding zero to the z coordinate if it is not present
for sdfv in range(len(LocalizationsFinal)):
    # Debug info
    print(f"Processing frame {sdfv+1}: ", end="")
    
    # Convert to numpy array
    try:
        if not isinstance(LocalizationsFinal[sdfv], np.ndarray):
            LF = np.array(LocalizationsFinal[sdfv], dtype=float)
        else:
            LF = LocalizationsFinal[sdfv].copy()
        print(f"shape: {LF.shape if hasattr(LF, 'shape') else 'scalar'}, size: {LF.size if hasattr(LF, 'size') else 1}")
        
        # Handle different shapes
        if LF.shape[1] == 2:
            processed_localizations.append(
                np.column_stack((
                    LF,
                    np.zeros((LF.shape[0], 1))
                    ))
                )
        else:
            # Already has 3+ columns, use first 3
            processed_localizations.append(LF[:, :3])
            print(f"Handled 2D array with final shape {processed_localizations[-1].shape}")
    except Exception as e:
        print(f"Error processing LocalizationsFinal for frame {sdfv}: {e}")
        processed_localizations.append(np.empty((0, 3)))
    
    # Do the same for TrueLocalizations
    try:
        if not isinstance(TrueLocalizations[sdfv], np.ndarray):
            TL = np.array(TrueLocalizations[sdfv], dtype=float)
        else:
            TL = TrueLocalizations[sdfv].copy()
        
        
        # Handle different shapes
        if TL.shape[1] == 2:
            processed_true_localizations.append(
                np.column_stack((
                    TL,
                    np.zeros((TL.shape[0], 1))
                    ))
                )
        else:
            # Already has 3+ columns, use first 3
            processed_true_localizations.append(TL[:, :3])
            print(f"Handled 2D array with final shape {processed_true_localizations[-1].shape}")
    except Exception as e:
        print(f"Error processing TrueLocalizations for frame {sdfv}: {e}")
        processed_true_localizations.append(np.empty((0, 3)))

# Replace the original lists with processed ones
LocalizationsFinal = processed_localizations
if len(processed_true_localizations) > 0:
    TrueLocalizations = processed_true_localizations




### 2a) estimate right splitting parameters

## This snippet is part of a loop that processes data from cell arrays named Frame_Information, LocalizationsFinal, and TrueLocalizations. 

# Initializes a counter, numeric arrays, and cell arrays to store processed data.
counter = 0
temp_numb_of_loc = []
addonarray = []
cut1array = []
cut2array = []
cut3array = []

#Iterates over each frame or data group in Frame_Information
# Prints the current index (ksu) and initializes temporary variables.
for ksu in range(len(Frame_Information)):
    print(ksu)
    pp = 0.2
    temp1 = 0
    temp2 = 0
    #Extracts the X, Y, Z columns from the ksu-th frame of LocalizationsFinal
    X1 = LocalizationsFinal[ksu][:, 0]
    X2 = LocalizationsFinal[ksu][:, 1]
    X3 = LocalizationsFinal[ksu][:, 2]

    if len(TrueLocalizations[ksu]) > 0:
        X1t = TrueLocalizations[ksu][:, 0]
        X2t = TrueLocalizations[ksu][:, 1]
        X3t = TrueLocalizations[ksu][:, 2]

    # Handle 2D data by setting Z values to 0
    # If the 3rd dimension (X3) is empty (i.e., 2D data), it creates a zero-valued array the same 
    # size as X2, essentially padding it with a Z-coordinate of 0.
    if X3.size == 0:
        X3 = np.zeros_like(X2)

    X4 = Frame_Information[ksu]

    addon = 250  # Buffer region for image splitting

    scorefinal = float('inf') #scorefinal is initialized to infinity, to store a minimum score later.
    onwers = 1  # Assuming this is a placeholder or initial value

    ## Phase Space Search

    
    while onwers < 300:
        print(f'Frac done with phase space search to split image {onwers / 300:.3f}')
    
        temp_numb_of_loc = []
        onwers += 1
        pp = np.random.rand() * 0.95 + 0.05
        pp2 = np.random.rand() * 0.95 + 0.05
    
        if np.max(X3) != 0:
            pp3 = np.random.rand() * 0.95 + 0.05
        else:
            pp3 = 1


        # The code uses nested loops to scan through the data volume in quantile-based bins 
        # (adaptive divisions, not uniform grid spacing), and for each subregion, it finds the number of points (localizations) inside.
        for i in range(1, int(ceil(1 / pp)) + 1):
            if i * pp < 1:
                cut1 = np.quantile(X1, [(i - 1) * pp, i * pp])
            else:
                cut1 = np.quantile(X1, [(i - 1) * pp, 1])
        
            for ii in range(1, int(ceil(1 / pp2)) + 1):
                if ii * pp2 < 1:
                    cut2 = np.quantile(X2, [(ii - 1) * pp2, ii * pp2])
                else:
                    cut2 = np.quantile(X2, [(ii - 1) * pp2, 1])
        
                for iii in range(1, int(ceil(1 / pp3)) + 1):
                    if iii * pp3 < 1:
                        cut3 = np.quantile(X3, [(iii - 1) * pp3, iii * pp3])
                    else:
                        cut3 = np.quantile(X3, [(iii - 1) * pp3, 1])
        
                    # Find indices within the buffered region
                    mask = (
                        (X1 >= cut1[0] - addon) & (X1 <= cut1[1] + addon) &
                        (X2 >= cut2[0] - addon) & (X2 <= cut2[1] + addon) &
                        (X3 >= cut3[0] - addon) & (X3 <= cut3[1] + addon)
                    )
        
                    temp_numb_of_loc.append(np.sum(mask))

        ## This block scores each trial split and retains the best (lowest-score) configuration.
        
        # min_loc is likely a predefined scalar target for the ideal number of localizations per region.
        # Resid stores the deviation from that ideal.
        # Calculates mean squared error of how far the observed region counts are from the ideal.
        Resid = np.array(temp_numb_of_loc) - min_loc
        score = np.sum(Resid ** 2) / len(temp_numb_of_loc)
        
        if score < scorefinal:
            scorefinal = score
            ppf = pp
            ppf2 = pp2
            ppf3 = pp3
            onwers = -onwers
            temp_numb_of_locf = temp_numb_of_loc.copy()


    addont = addon  

    # Use the best-found splitting parameters
    pp = ppf
    pp2 = ppf2
    pp3 = ppf3

    # If not enough points, skip splitting
    if len(X1) < min_loc:
        ppf = 1
        ppf2 = 1
        ppf3 = 1

        pp = ppf
        pp2 = ppf2
        pp3 = ppf3

    flag = 0


    ### 2b) apply splitting parameters
    for i in range(1, int(np.ceil(1 / pp)) + 1):
        if flag == 1:
            break

        cut1 = np.quantile(X1, [(i - 1) * pp, min(i * pp, 1)])

        for ii in range(1, int(np.ceil(1 / pp2)) + 1):
            cut2 = np.quantile(X2, [(ii - 1) * pp2, min(ii * pp2, 1)])

            for iii in range(1, int(np.ceil(1 / pp3)) + 1):
                cut3 = np.quantile(X3, [(iii - 1) * pp3, min(iii * pp3, 1)])
                addon = addont

                if len(X1) > min_loc:
                    while True:
                        mask = (
                            (X1 > cut1[0] - addon) & (X1 < cut1[1] + addon) &
                            (X2 > cut2[0] - addon) & (X2 < cut2[1] + addon) &
                            (X3 >= cut3[0] - addon) & (X3 <= cut3[1] + addon)
                        )
                        IND = np.where(mask)[0]
                        if len(IND) >= min_loc:
                            break
                        addon += 10
                else:
                    IND = np.arange(len(X1))

                while len(IND) > min_loc:
                    addon -= 10
                    mask = (
                        (X1 > cut1[0] - addon) & (X1 < cut1[1] + addon) &
                        (X2 > cut2[0] - addon) & (X2 < cut2[1] + addon) &
                        (X3 >= cut3[0] - addon) & (X3 <= cut3[1] + addon)
                    )
                    IND = np.where(mask)[0]
                    if addon < 150:
                        break

                if TrueLocalizations[ksu] is not None:
                    X1t, X2t, X3t = TrueLocalizations[ksu].T
                    maskt = (
                        (X1t > cut1[0] - addon) & (X1t < cut1[1] + addon) &
                        (X2t > cut2[0] - addon) & (X2t < cut2[1] + addon) &
                        (X3t >= cut3[0] - addon) & (X3t <= cut3[1] + addon)
                    )
                    INDt = np.where(maskt)[0]

                # Plot
                fig = plt.figure(1)
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(X1[IND], X2[IND], X3[IND], c=X4[IND], s=10, cmap='jet')
                ax.set_box_aspect([1, 1, 1])
                plt.draw()
                plt.pause(0.5)
                plt.clf()

                # Save results
                LocalizationsFinal_Split.append(np.column_stack((X1[IND], X2[IND], X3[IND])))
                Photons_Split.append(Photons[ksu][IND])
                if TrueLocalizations[ksu] is not None:
                    TrueLocalizations_Split.append(np.column_stack((X1t[INDt], X2t[INDt], X3t[INDt])))

                cut1array.append(cut1)
                cut2array.append(cut2)
                cut3array.append(cut3)
                Frame_Information_Split.append(X4[IND])
                Came_from_image.append(ksu)
                Parameters_to_split.append([ppf, ppf2, ppf3])
                addonarray.append(addon)
                temp_numb_of_loc.append(len(IND))

                counter += 1

                if len(X1) < min_loc:
                    flag = 1
                    break


# Overwrite original variables with split versions
LocalizationsFinal = LocalizationsFinal_Split
Frame_Information = Frame_Information_Split
TrueLocalizations = TrueLocalizations_Split
Photons = Photons_Split

# Save to .mat file
savemat(f'Split_{Condition}.mat', {
    'LocalizationsFinal': LocalizationsFinal,
    'Frame_Information': Frame_Information,
    'addonarray': addonarray,
    'Parameters_to_split': Parameters_to_split,
    'Came_from_image': Came_from_image,
    'TrueLocalizations': TrueLocalizations,
    'cut1array': cut1array,
    'cut2array': cut2array,
    'cut3array': cut3array,
    'Resolution': Resolution,
    'Photons': Photons
})
