from . import io, processing
import logging
import numpy as np
import os
import matplotlib.pyplot as plt
import tqdm
import cv2
import pandas as pd

class SideScanSonarPipeline:

    def __init__(self, config_path):
        """
        Initialize the Side Scan Sonar Processing Pipeline with a configuration file.
        """
        # Validate the configuration path
        if not config_path:
            raise ValueError("Configuration path must be provided.")
        if not isinstance(config_path, str):
            raise TypeError("Configuration path must be a string.")
        if not config_path.endswith('.yaml'):
            raise ValueError("Configuration file must be a YAML file with .yaml extension.")
        
        # Set up logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load the configuration file
        if not io.load_yaml_config(config_path):
            raise ValueError("Failed to load configuration from {}".format(config_path))
        self.config = io.load_yaml_config(config_path)
        self.logger.info("Configuration loaded successfully from {}".format(config_path))

        # Data fields
        self.input_data = None
        self.processed_data = None
        self.mean_first_bottom_return = None

        # Non-indipendent parameters
        self.theta_s_max = self.config.get('mounting_angle') + self.config.get('beam_aperture') /2
        self.theta_s_min = self.config.get('mounting_angle') - self.config.get('beam_aperture') /2
        self.slant_range_resolution = self.config.get('slant_range') / self.config.get('bin_per_swathe')
        self.meter_to_pixel_resolution = self.slant_range_resolution * 2 


    def load_data(self):
        """
        Load the input data for the pipeline.
        """

        data_path = self.config.get('input_data_path')
        # Validate the data path
        if not data_path:
            raise ValueError("Data path must be specified in the configuration file.")
        if not isinstance(data_path, str):
            raise TypeError("Data path must be a string.")
        if not data_path.endswith('.csv'):
            raise ValueError("Input data file must be a CSV file with .csv extension.")
        self.logger.info("Loading input data from {}".format(data_path))

        self.input_data = io.load_csv(data_path)
        if self.input_data is None:
            raise ValueError("Failed to load input data from {}".format(data_path))
        self.logger.info("Input data loaded successfully from {}".format(data_path))
        self.logger.debug("Input data: {}".format(self.input_data.head()))


    def clean_data(self):
        """
        Clean the input data by removing invalid entries.
        """
        if self.input_data is None:
            raise ValueError("Input data is not loaded. Please load data before cleaning.")
        
        # Initialize the processed data as a copy of the input data
        self.processed_data = self.input_data.copy()

        # drop na values resetting the index
        initial_count = len(self.processed_data)
        self.processed_data.dropna(inplace=True)
        self.processed_data.reset_index(drop=True, inplace=True)
        self.logger.info("Dropped na values from input data. Entries Removed: {}".format(initial_count - len(self.processed_data)))

        # Remove entries with altitude less than minimum
        min_altitude = self.config.get('min_altitude')
        if min_altitude is not None:
            initial_count = len(self.processed_data)
            self.processed_data = self.processed_data[self.processed_data['altitude'] >= min_altitude]
            self.logger.info("Removed entries with altitude less than {}. Entries removed: {}".format(min_altitude, initial_count - len(self.processed_data)))
        else:
            self.logger.warning("Minimum altitude not specified in configuration. Skipping altitude filtering.")
        
    
    def compute_first_bottom_returns(self):
        """
        Add columns to the processed data for left and right first bottom returns.
        This is done based on the navigation data and geometric model of SSS.
        """
        if self.processed_data is None:
            raise ValueError("Processed data is not available. Please clean data before computing first bottom returns.")


        # Add columns initialized at 0 values
        self.processed_data['left_first_bottom_return'] = 0.0
        self.processed_data['right_first_bottom_return'] = 0.0

        # Compute the first bottom returns this can be done as column operations
        self.processed_data["right_beam_first_return"] = self.processed_data["altitude"]/(np.cos(self.processed_data["pitch"])*(np.cos(self.processed_data["roll"])*np.sin(self.theta_s_max) - np.sin(self.processed_data["roll"])*np.cos(self.theta_s_max))*self.slant_range_resolution)
        self.processed_data["left_beam_first_return"] = self.processed_data["altitude"]/(np.cos(self.processed_data["pitch"])*(np.cos(self.processed_data["roll"])*np.sin(np.pi - self.theta_s_max) - np.sin(self.processed_data["roll"])*np.cos(np.pi - self.theta_s_max))*self.slant_range_resolution)

        # Convert to integer
        self.processed_data["left_beam_first_return"] = self.processed_data["left_beam_first_return"].astype(int)
        self.processed_data["right_beam_first_return"] = self.processed_data["right_beam_first_return"].astype(int)

        self.logger.info("Computed first bottom returns for left and right beams.")

        # Compute mean first bottom return
        self.mean_first_bottom_return = np.mean([
            self.processed_data["left_beam_first_return"].mean(),
            self.processed_data["right_beam_first_return"].mean()
        ]).astype(int)

        self.logger.info("Mean first bottom return computed: {}".format(self.mean_first_bottom_return))

        # Remove outliers based on the mean first bottom return
        mean = self.mean_first_bottom_return
        std_dev = self.processed_data[["left_beam_first_return", "right_beam_first_return"]].std().mean()

        initial_count = len(self.processed_data)
        self.processed_data = self.processed_data[
            (self.processed_data["left_beam_first_return"] >= mean - 3 * std_dev) &
            (self.processed_data["left_beam_first_return"] <= mean + 3 * std_dev) &
            (self.processed_data["right_beam_first_return"] >= mean - 3 * std_dev) &
            (self.processed_data["right_beam_first_return"] <= mean + 3 * std_dev)
        ]
        self.processed_data.reset_index(drop=True, inplace=True)
        self.logger.info("Removed outliers based on mean first bottom return. Entries removed: {}".format(initial_count - len(self.processed_data)))
        self.logger.info("Standard deviation used for outlier removal: {}".format(std_dev))


    def discard_nadir_and_far_field(self):
        """
        Discard nadir and far field returns from the processed data.
        This is done based on the mean first bottom return, and the far field percentage.
        """

        # Fetch the far field percentage from the configuration
        far_field_percentage = self.config.get('far_field_percentage')
        if far_field_percentage is None:
            raise ValueError("Far field percentage must be specified in the configuration file.")
        if not (0 <= far_field_percentage <= 1):
            raise ValueError("Far field percentage must be between 0 and 1.")

        # Compute number of bins to discard
        num_bins_to_discard = int(far_field_percentage * self.config.get('bin_per_swathe'))

        # Discard nadir and far field returns
        # In processed data, there are two columns named sss_right_beam and sss_left_beam
        # These are list of returns for each bin, of length bin_per_swathe. We need to crop out
        # these vectors by resizing them from [0 -> bin_per_swathe] to [mean_first_bottom_return -> bin_per_swathe - num_bins_to_discard]
        if self.processed_data is None:
            raise ValueError("Processed data is not available. Please compute first bottom returns before discarding nadir and far field returns.")
        
        """
        self.processed_data['sss_left_beam'] = self.processed_data['sss_left_beam'].apply(
            lambda x: x[self.mean_first_bottom_return:self.config.get('bin_per_swathe') - num_bins_to_discard]
        )
        self.processed_data['sss_right_beam'] = self.processed_data['sss_right_beam'].apply(
            lambda x: x[self.mean_first_bottom_return:self.config.get('bin_per_swathe') - num_bins_to_discard]
        )
        """
        for j in range(len(self.processed_data)):
            self.processed_data.at[j, 'sss_left_beam'] = eval(self.processed_data.at[j, 'sss_left_beam'])[self.mean_first_bottom_return:self.config.get('bin_per_swathe') - num_bins_to_discard]
            self.processed_data.at[j, 'sss_right_beam'] = eval(self.processed_data.at[j, 'sss_right_beam'])[self.mean_first_bottom_return:self.config.get('bin_per_swathe') - num_bins_to_discard]


        self.logger.info("Discarded nadir and far field returns from the processed data.")
        self.logger.info("Length of left beam: {}".format(len(self.processed_data['sss_left_beam'][0])))
        #self.logger.info("Left beam: {}".format(self.processed_data['sss_left_beam'][0]))
        self.logger.info("Length of right beam: {}".format(len(self.processed_data['sss_right_beam'][0])))
        #self.logger.info("Right beam: {}".format(self.processed_data['sss_right_beam'][0]))


    def tvg_correction(self):
        """
        Apply time-varied gain (TVG) correction to the processed data.
        This is a placeholder for the actual TVG correction implementation.
        """
        if self.processed_data is None:
            raise ValueError("Processed data is not available. Please clean data before applying TVG correction.")
        

        # Compute the mean for the left and right beams
        mean_total = np.zeros(len(self.processed_data['sss_left_beam'][0]), dtype=float)
        for j in range(len(self.processed_data)):
            left_beam = np.array(self.processed_data.at[j, 'sss_left_beam'])
            right_beam = np.array(self.processed_data.at[j, 'sss_right_beam'])
            mean_total += (left_beam + right_beam) / 2
        mean_total /= len(self.processed_data)
        self.logger.info("Computed mean total beam for TVG correction.")

        # Normalize the mean total beam
        if np.any(mean_total == 0):
            raise ValueError("Mean total beam contains zero values, cannot normalize.")
        mean_total = mean_total / np.min(mean_total)

        
        # Create the folder in which to store the tvg plot
        output_folder_path = self.config.get('output_tvg_folder')
        if not output_folder_path:
            raise ValueError("Output TVG folder path must be specified in the configuration file.")
        os.makedirs(output_folder_path, exist_ok=True)

        # Compute an inverse square model for the mean total beam
        gain_model, tvg = processing.fit_inverse_square_model(mean_total)
        if gain_model is None:
            raise ValueError("Failed to fit inverse square model to the mean total beam.")
        

        # Save the plot of mean_total and gain_model compared
        output_file_path = os.path.join(output_folder_path, 'tvg_correction_plot.png')
        plt.figure()
        plt.plot(mean_total, label="Mean Total Beam")
        plt.plot(gain_model, label="Fitted Model", linestyle='--')
        plt.title("Normalized mean beam vs Fitted Gain Model")
        plt.xlabel("Bin")
        plt.ylabel("Gain")
        plt.grid()
        plt.legend()
        plt.savefig(output_file_path)
        plt.close()

        # Save the plot of tvg correction
        output_file_path = os.path.join(output_folder_path, 'tvg_correction.png')
        plt.figure()
        plt.plot(tvg)
        plt.title("Time Varying Gain Correction")
        plt.xlabel("Bin")
        plt.ylabel("Gain")
        plt.grid()
        plt.savefig(output_file_path)
        plt.close()

        self.logger.info("Mean beam normalized and tvg computed. Plots saved to {}".format(output_folder_path))

        # Apply the TVG correction to the processed data
        for j in range(len(self.processed_data)):
            left_beam = np.array(self.processed_data.at[j, 'sss_left_beam'])
            right_beam = np.array(self.processed_data.at[j, 'sss_right_beam'])
            self.processed_data.at[j, 'sss_left_beam'] = (left_beam * tvg).astype(int)
            self.processed_data.at[j, 'sss_right_beam'] = (right_beam * tvg).astype(int)
        
        self.logger.info("Applied time-varied gain correction to the processed data.")


    def assign_transects(self):
        """
        Assign transects to the processed data.
        Non-transect entries are dropped.
        """
        if self.processed_data is None:
            raise ValueError("Processed data is not available. Please clean data before assigning transects.")
        

        # Add a new column for transect assignment
        self.processed_data['transect'] = np.nan

        # Assign transects based on the navigation data
        reference_value = self.processed_data['yaw'][0]
        transect_counter = 0
        for i in tqdm.tqdm(range(len(self.processed_data)), desc="Assigning transects"):
            yaw = self.processed_data['yaw'][i]
            if abs(yaw - reference_value) > self.config.get('transect_angle_threshold'):
                transect_counter += 1
                reference_value = yaw
            self.processed_data.at[i, 'transect'] = transect_counter

        # Drop transects with less than the minimum number of instances
        min_instances = self.config.get('minimum_instances_per_transect')
        if min_instances is None:
            raise ValueError("Minimum instances per transect must be specified in the configuration file.")
        if not isinstance(min_instances, int) or min_instances <= 0:
            raise ValueError("Minimum instances per transect must be a positive integer.")
        initial_count = len(self.processed_data)
        self.processed_data = self.processed_data.groupby('transect').filter(lambda x: len(x) >= min_instances)
        self.processed_data.reset_index(drop=True, inplace=True)
        # Re-enumerate the transect column from 0 to len(df["transect"].unique())
        unique_transects = self.processed_data['transect'].unique()
        for i, transect in enumerate(unique_transects):
            self.processed_data.loc[self.processed_data['transect'] == transect, 'transect'] = i

        self.logger.info("Assigned transects to the processed data. "
                         "Dropped {} non-transect entries.".format(initial_count - len(self.processed_data)))
        self.logger.info("Number of unique transects: {}".format(len(self.processed_data['transect'].unique())))


    def waterfall_images(self):
        """
        Generate waterfall images for the processed data.
        This is a placeholder for the actual waterfall image generation implementation.
        """
        if self.processed_data is None:
            raise ValueError("Processed data is not available. Please clean data before generating waterfall images.")
        
        # Create the folder in which to store the waterfall images
        output_folder_path = self.config.get('output_waterfall_folder')
        if not output_folder_path:
            raise ValueError("Output waterfall folder path must be specified in the configuration file.")
        os.makedirs(output_folder_path, exist_ok=True)

        # Generate waterfall images for each transect
        for transect in self.processed_data['transect'].unique():
            transect_data = self.processed_data[self.processed_data['transect'] == transect]
            left_beam = np.array(transect_data['sss_left_beam'].tolist())
            right_beam = np.array(transect_data['sss_right_beam'].tolist())

            # Create a waterfall image for the left beam
            left_image = np.array(left_beam, dtype=np.uint8)
            # Flip the image horizontally and vertically to match the expected orientation
            left_image = cv2.flip(left_image, 1)
            left_image = cv2.flip(left_image, 0)
            # Normalize the image to the range [0, 255]
            left_image = cv2.normalize(left_image, None, 0, 255, cv2.NORM_MINMAX)
            # Save as grayscale image
            left_image = cv2.cvtColor(left_image, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(os.path.join(output_folder_path, 'waterfall_left_transect_{}.png'.format(int(transect))), left_image)

            # Create a waterfall image for the right beam
            right_image = np.array(right_beam, dtype=np.uint8)
            # Flip the image vertically to match the expected orientation
            right_image = cv2.flip(right_image, 0)
            # Normalize the image to the range [0, 255]
            right_image = cv2.normalize(right_image, None, 0, 255, cv2.NORM_MINMAX)
            # Save as grayscale image
            right_image = cv2.cvtColor(right_image, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(os.path.join(output_folder_path, 'waterfall_right_transect_{}.png'.format(int(transect))), right_image)

        self.logger.info("Waterfall images generated and saved to {}".format(output_folder_path))


    def bin_NED_localizer(self):
        """
        For each transect, for each side, for each beam, build a new .csv file 
        where each row corresponds to a bin. The columns are:

        - bin: the index of the bin
        - value: the intensity value of that bin
        - north: the north coordinate of the bin in NED frame
        - east: the east coordinate of the bin in NED frame
        - down: the down coordinate of the bin in NED frame
        - theta_s: the angle of the bin
        - slant_range: the slant range of the bin

        the new .csv file is saved in the output folder specified in the configuration file.
        """
        if self.processed_data is None:
            raise ValueError("Processed data is not available. Please clean data before binning NED localizer.")

        # Create the folder in which to store the bin NED localizer files
        output_folder_path = self.config.get('output_localized_measures_folder')
        os.makedirs(output_folder_path, exist_ok=True)

        # Iterate over each transect
        for transect in tqdm.tqdm(self.processed_data['transect'].unique(), desc="Localizing bins for each transect"):

            # Create the transect folder inside the output folder
            transect_folder_path = os.path.join(output_folder_path, 'transect_{}'.format(int(transect)))
            os.makedirs(transect_folder_path, exist_ok=True)

            # Gather the data for the current transect
            transect_data = self.processed_data[self.processed_data['transect'] == transect]
            # Reset the index of the transect data
            transect_data.reset_index(drop=True, inplace=True)

            # Iterate over each measure (instance of the dataframe)
            for j in tqdm.tqdm(range(len(transect_data)), desc="Processing measures for transect {}".format(int(transect))):

                # Fetch quantities used for both sides
                roll = transect_data['roll'].iloc[j]
                pitch = transect_data['pitch'].iloc[j]
                yaw = transect_data['yaw'].iloc[j]
                altitude = transect_data['altitude'].iloc[j]
                north = transect_data['north'].iloc[j]
                east = transect_data['east'].iloc[j]
                down = transect_data['down'].iloc[j]

                # Compute rotation matrix from body to NED
                R_z_yaw = np.array([[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
                R_y_pitch = np.array([[np.cos(pitch), 0, -np.sin(pitch)], [0, 1, 0], [np.sin(pitch), 0, np.cos(pitch)]])
                R_x_roll = np.array([[1, 0, 0], [0, np.cos(roll), np.sin(roll)], [0, -np.sin(roll), np.cos(roll)]])
                R_body_to_NED = np.transpose(R_z_yaw) @ np.transpose(R_y_pitch) @ np.transpose(R_x_roll)

                # Build the vector going from the origin of NED frame to origin of body frame, expressed in NED frame
                OnOb_NED = np.array([north, east, down])

                # Build the versor that points from the origin of the body frame to the seafloor, being perpendicular to the seafloor, expressed in body frame
                versor_seafloor_body = np.array([np.sin(pitch), -np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)])
                    
                # Start building transect_i_measure_j_right.csv and transect_i_measure_j_left.csv
                right_df = pd.DataFrame(columns=["bin", "value", "north", "east", "down", "theta_s"])
                left_df = pd.DataFrame(columns=["bin", "value", "north", "east", "down", "theta_s"])

                # "bin" column goes from self.mean_first_bottom_return to self.mean_first_bottom_return + len(transect_data['sss_right_beam'][0])
                right_df["bin"] = np.arange(self.mean_first_bottom_return, self.mean_first_bottom_return + len(transect_data['sss_right_beam'].iloc[j]))
                left_df["bin"] = np.arange(self.mean_first_bottom_return, self.mean_first_bottom_return + len(transect_data['sss_left_beam'].iloc[j]))

                # "value" column is the intensity value of the bin
                right_df["value"] = transect_data["sss_right_beam"].iloc[j]
                left_df["value"] = transect_data["sss_left_beam"].iloc[j]

                # "slant_range" column is computed based on "bin" column
                right_df["slant_range"] = self.slant_range_resolution * right_df["bin"]
                left_df["slant_range"] = self.slant_range_resolution * left_df["bin"]

                # Iterate over each bin in the right beam
                for k in range(len(right_df)):

                    # Compute the angle of the beam
                    temp_sin = altitude / (right_df["slant_range"][k] * np.cos(pitch))
                    if temp_sin > np.sin(self.theta_s_max - roll):
                        right_df.loc[k, "theta_s"] = None
                    elif temp_sin < np.sin(self.theta_s_min - roll):
                        right_df.loc[k, "theta_s"] = None
                    else:
                        right_df.loc[k, "theta_s"] = np.arcsin(temp_sin) + roll

                    # If the beam does not intersect the seafloor, the coordinates are set to None
                    if right_df["theta_s"][k] is None:
                        right_df.loc[k, "north"] = None
                        right_df.loc[k, "east"] = None
                        right_df.loc[k, "down"] = None
                    else:
                        # Build the versor of the beam in body frame
                        versor_beam_body = np.array([0, np.cos(right_df["theta_s"][k]), np.sin(right_df["theta_s"][k])])

                        # Compute the vector going from the origin of body frame to the intersection between the beam and the seafloor, expressed in body frame
                        dot_product = np.dot(versor_beam_body, versor_seafloor_body)
                        if dot_product == 0 or np.isnan(dot_product):
                            print(f"Warning: dot product is zero for transect {transect}, measure {j}, bin {k}. Skipping this bin.")
                            print(f"Roll: {roll}, Pitch: {pitch}, Theta_s: {right_df['theta_s'][k]}")
                            print(f"Versor beam body: {versor_beam_body}, Versor seafloor body: {versor_seafloor_body}")
                            right_df.loc[k, "north"] = None
                            right_df.loc[k, "east"] = None
                            right_df.loc[k, "down"] = None
                            continue
                        ObPs_body = versor_beam_body * altitude / dot_product

                        # Rotate the vector to NED frame
                        ObPs_NED = R_body_to_NED @ ObPs_body

                        # Compute the vector that goes from origin of NED frame to the intersection between the beam and the seafloor, expressed in NED frame
                        OnPs_NED = OnOb_NED + ObPs_NED

                        # Fill the columns
                        right_df.loc[k, "north"] = OnPs_NED[0]
                        right_df.loc[k, "east"] = OnPs_NED[1]
                        right_df.loc[k, "down"] = OnPs_NED[2]

                # Drop rows with nan values
                right_df.dropna(inplace=True)

                # Iterate over each bin in the left beam
                for k in range(len(left_df)):

                    # Compute the angle of the beam
                    temp_sin = altitude / (left_df["slant_range"][k] * np.cos(pitch))
                    if temp_sin > np.sin(np.pi - self.theta_s_max + roll):
                        left_df.loc[k, "theta_s"] = None
                    elif temp_sin < np.sin(np.pi - self.theta_s_min + roll):
                        left_df.loc[k, "theta_s"] = None
                    else:
                        left_df.loc[k, "theta_s"] = np.pi -np.arcsin(temp_sin) + roll

                    # If the beam does not intersect the seafloor, the coordinates are set to None
                    if left_df["theta_s"][k] is None:
                        left_df.loc[k, "north"] = None
                        left_df.loc[k, "east"] = None
                        left_df.loc[k, "down"] = None
                    else:
                        # Build the versor of the beam in body frame
                        versor_beam_body = np.array([0, np.cos(left_df["theta_s"][k]), np.sin(left_df["theta_s"][k])])

                        # Compute the vector going from the origin of body frame to the intersection between the beam and the seafloor, expressed in body frame
                        dot_product = np.dot(versor_beam_body, versor_seafloor_body)
                        if dot_product == 0 or np.isnan(dot_product):
                            print(f"Warning: dot product is zero for transect {transect}, measure {j}, bin {k}. Skipping this bin.")
                            print(f"Roll: {roll}, Pitch: {pitch}, Theta_s: {left_df['theta_s'][k]}")
                            print(f"Versor beam body: {versor_beam_body}, Versor seafloor body: {versor_seafloor_body}")
                            left_df.loc[k, "north"] = None
                            left_df.loc[k, "east"] = None
                            left_df.loc[k, "down"] = None
                            continue
                        ObPs_body = versor_beam_body * altitude / dot_product

                        # Rotate the vector to NED frame
                        ObPs_NED = R_body_to_NED @ ObPs_body

                        # Compute the vector that goes from origin of NED frame to the intersection between the beam and the seafloor, expressed in NED frame
                        OnPs_NED = OnOb_NED + ObPs_NED

                        # Fill the columns
                        left_df.loc[k, "north"] = OnPs_NED[0]
                        left_df.loc[k, "east"] = OnPs_NED[1]
                        left_df.loc[k, "down"] = OnPs_NED[2]

                # Drop rows with nan values
                left_df.dropna(inplace=True)

                # Save the dataframes to csv files
                right_df.to_csv(os.path.join(transect_folder_path, 'transect_{}_measure_{}_right.csv'.format(int(transect), j)), index=False)
                left_df.to_csv(os.path.join(transect_folder_path, 'transect_{}_measure_{}_left.csv'.format(int(transect), j)), index=False)
            

    def generate_mosaics(self):
        """
        Produce mosaics for each transect from the localized bins.
        """
        if self.processed_data is None:
            raise ValueError("Processed data is not available. Please clean data before producing mosaics.")
        
        # Create the folder in which to store the mosaics
        output_raw_folder_path = self.config.get('output_raw_mosaics_folder')
        if not output_raw_folder_path:
            raise ValueError("Output raw mosaic folder path must be specified in the configuration file.")
        os.makedirs(output_raw_folder_path, exist_ok=True)

        output_cleaned_folder_path = self.config.get('output_clean_mosaics_folder')
        if not output_cleaned_folder_path:
            raise ValueError("Output cleaned mosaic folder path must be specified in the configuration file.")
        os.makedirs(output_cleaned_folder_path, exist_ok=True)
        self.logger.info("Creating folders for raw and cleaned mosaics at {} and {}".format(output_raw_folder_path, output_cleaned_folder_path))
        
        output_filled_rect_folder_path = self.config.get('output_filled_rect_mosaics_folder')
        if not output_filled_rect_folder_path:
            raise ValueError("Output filled rectangle mosaic folder path must be specified in the configuration file.")
        os.makedirs(output_filled_rect_folder_path, exist_ok=True)
        self.logger.info("Creating folder for filled rectangle mosaics at {}".format(output_filled_rect_folder_path))

        # Iterate over transect folders in localized measures folder
        localized_measures_folder = self.config.get('output_localized_measures_folder')
        if not localized_measures_folder:
            raise ValueError("Output localized measures folder path must be specified in the configuration file.")
        if not os.path.exists(localized_measures_folder):
            raise FileNotFoundError("Localized measures folder does not exist: {}".format(localized_measures_folder))
        transect_folders = [f for f in os.listdir(localized_measures_folder) if os.path.isdir(os.path.join(localized_measures_folder, f))]
        if not transect_folders:
            raise ValueError("No transect folders found in the localized measures folder: {}".format(localized_measures_folder))
        self.logger.info("Found {} transect folders in the localized measures folder.".format(len(transect_folders)))

        for transect_folder in tqdm.tqdm(transect_folders, desc="Producing mosaics"):

            transect_path = os.path.join(localized_measures_folder, transect_folder)
            if not os.path.exists(transect_path):
                self.logger.warning("Transect folder does not exist: {}".format(transect_path))
                continue

            transect = transect_folder.split('_')[-1]  # Extract transect number from folder name
            
            # Initialize empty lists to hold the data for each side
            right_data = []
            left_data = []

            # Iterate over each measure file in the transect folder
            measure_files = [f for f in os.listdir(transect_path) if f.endswith('_right.csv') or f.endswith('_left.csv')]
            if not measure_files:
                self.logger.warning("No measure files found in transect folder: {}".format(transect_path))
                continue

            for measure_file in measure_files:
                measure_path = os.path.join(transect_path, measure_file)
                if not os.path.exists(measure_path):
                    self.logger.warning("Measure file does not exist: {}".format(measure_path))
                    continue
                
                # Load the measure data
                measure_data = pd.read_csv(measure_path)

                # Append the data to the respective list based on the file name
                if 'right' in measure_file:
                    right_data.append(measure_data)
                elif 'left' in measure_file:
                    left_data.append(measure_data)

            # Create mosaics for each side
            if right_data:
                right_df = pd.concat(right_data, ignore_index=True)
                

            if left_data:
                left_df = pd.concat(left_data, ignore_index=True)

            # Fetch the maximum and minimum values of the coordinates
            right_max_north = right_df["north"].max()
            right_min_north = right_df["north"].min()
            right_max_east = right_df["east"].max()
            right_min_east = right_df["east"].min()
            left_max_north = left_df["north"].max()
            left_min_north = left_df["north"].min()
            left_max_east = left_df["east"].max()
            left_min_east = left_df["east"].min()

            padding_meters = self.config.get('padding_meters', 0.0)

            # Compute the dimensions of the mosaic
            right_width = int((right_max_east - right_min_east + 2*padding_meters) / self.meter_to_pixel_resolution)
            right_height = int((right_max_north - right_min_north + 2*padding_meters) / self.meter_to_pixel_resolution)
            left_width = int((left_max_east - left_min_east + 2*padding_meters) / self.meter_to_pixel_resolution)
            left_height = int((left_max_north - left_min_north + 2*padding_meters) / self.meter_to_pixel_resolution)

            # For each row in the dataframe compute the mosaic pixel coordinates
            right_df["u_mosaic"] = ((right_df["east"] - right_min_east + padding_meters) / self.meter_to_pixel_resolution).astype(int)
            right_df["v_mosaic"] = right_height - ((right_df["north"] - right_min_north + padding_meters) / self.meter_to_pixel_resolution).astype(int)
            left_df["u_mosaic"] = ((left_df["east"] - left_min_east + padding_meters) / self.meter_to_pixel_resolution).astype(int)
            left_df["v_mosaic"] = left_height - ((left_df["north"] - left_min_north + padding_meters) / self.meter_to_pixel_resolution).astype(int)

            # Collapse the dataframe by averaging rows with the same pixel coordinates
            right_df = right_df.groupby(["u_mosaic", "v_mosaic"]).mean().reset_index()
            left_df = left_df.groupby(["u_mosaic", "v_mosaic"]).mean().reset_index()



            # Normalize the intensity values to the range [0, 255]
            right_df["value"] = cv2.normalize(right_df["value"].values, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            left_df["value"] = cv2.normalize(left_df["value"].values, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Create empty mosaics
            right_image = np.zeros((right_height, right_width), dtype=np.uint8)
            left_image = np.zeros((left_height, left_width), dtype=np.uint8)

            # Fill the mosaics with the intensity values
            for k in tqdm.tqdm(range(len(right_df)), desc=f"Building mosaic for transect {transect}, right side: "):
                right_image[right_df["v_mosaic"][k], right_df["u_mosaic"][k]] = right_df["value"][k]

            for k in tqdm.tqdm(range(len(left_df)), desc=f"Building mosaic for transect {transect}, left side: "):
                left_image[left_df["v_mosaic"][k], left_df["u_mosaic"][k]] = left_df["value"][k]
            
            # Add a alpha channel to the image and convert black pixels to transparent
            right_image = cv2.cvtColor(right_image.astype(np.uint8), cv2.COLOR_GRAY2BGRA)
            right_image[:, :, 3] = 255  # Set alpha to 255 (opaque) for all pixels
            right_black_pixels = (right_image[:, :, 0] == 0) & (right_image[:, :, 1] == 0) & (right_image[:, :, 2] == 0)
            right_image[right_black_pixels, 3] = 0  # Set alpha to 0 (transparent) for black pixels
            
            left_image = cv2.cvtColor(left_image.astype(np.uint8), cv2.COLOR_GRAY2BGRA)
            left_image[:, :, 3] = 255  # Set alpha to 255 (opaque) for all pixels
            left_black_pixels = (left_image[:, :, 0] == 0) & (left_image[:, :, 1] == 0) & (left_image[:, :, 2] == 0)
            left_image[left_black_pixels, 3] = 0  # Set alpha to 0 (transparent) for black pixels

            # Save the raw mosaics in the output raw folder
            cv2.imwrite(os.path.join(output_raw_folder_path, f"transect_{transect}_right_raw.png"), right_image)
            cv2.imwrite(os.path.join(output_raw_folder_path, f"transect_{transect}_left_raw.png"), left_image)
            

            # Create a version of the mosaics with gaps filled using cv2 inpaint
            # Extract the BGR image and the alpha channel (used as the mask)
            right_image_bgr = right_image[:, :, :3]
            right_image_mask = (right_image[:, :, 3] == 0).astype(np.uint8)

            left_image_bgr = left_image[:, :, :3]
            left_image_mask = (left_image[:, :, 3] == 0).astype(np.uint8)

            # Perform inpainting
            right_image_filled = cv2.inpaint(right_image_bgr, right_image_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            left_image_filled = cv2.inpaint(left_image_bgr, left_image_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

            # Build mask of valid pixels (non-zero alpha)
            right_mask = (right_image[:, :, 3] > 0).astype(np.uint8)
            left_mask = (left_image[:, :, 3] > 0).astype(np.uint8)

            # Get rect and rotate-crop
            rect_right, _ = processing.get_min_area_rect(right_mask)
            cropped_right, rotated_right = processing.rotate_and_crop(right_image_filled, rect_right)
            rect_left, _ = processing.get_min_area_rect(left_mask)
            cropped_left, rotated_left = processing.rotate_and_crop(left_image_filled, rect_left)

            # Draw contours on the filled images
            cv2.drawContours(right_image_filled, [cv2.boxPoints(rect_right).astype(np.int32)], 0, (0, 255, 0), 2)
            cv2.drawContours(left_image_filled, [cv2.boxPoints(rect_left).astype(np.int32)], 0, (0, 255, 0), 2)

            # Save the filled mosaics in the output filled rect folder
            cv2.imwrite(os.path.join(output_filled_rect_folder_path, f"transect_{transect}_right_rect.png"), right_image_filled)
            cv2.imwrite(os.path.join(output_filled_rect_folder_path, f"transect_{transect}_left_rect.png"), left_image_filled)

            # Save the cropped mosaics
            cv2.imwrite(os.path.join(output_cleaned_folder_path, f"transect_{transect}_right_cropped.png"), cropped_right)
            cv2.imwrite(os.path.join(output_cleaned_folder_path, f"transect_{transect}_left_cropped.png"), cropped_left)




    def run(self):
        """
        Run the side scan sonar processing pipeline.
        """

        # Load the input data csv file
        self.load_data()

        # Clean the input data
        self.clean_data()

        # Compute the first bottom returns
        self.compute_first_bottom_returns()

        # Discard nadir and far field returns
        self.discard_nadir_and_far_field()

        # Apply time-varied gain correction
        self.tvg_correction()

        # Assign transects to the processed data
        self.assign_transects()

        # Generate waterfall images for the processed data
        self.waterfall_images()

        # Localize the bins in NED frame
        if self.config.get('NED_localizer_flag'):
            self.bin_NED_localizer()
        
        # Generate mosaics images
        self.generate_mosaics()








        