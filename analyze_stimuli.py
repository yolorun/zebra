import numpy as np
import matplotlib.pyplot as plt

def read_and_analyze_stimuli(file_path, num_channels=10):
    """
    Reads a binary file with multi-channel float data, analyzes, and plots it.

    Args:
        file_path (str): The path to the binary file.
        num_channels (int): The number of channels in the data.
    """
    try:
        # Read the binary data from the file
        # The .flt format suggests floating point numbers, we'll assume 32-bit floats
        data = np.fromfile(file_path, dtype=np.float32)

        # Reshape the data according to the number of channels
        # The -1 infers the number of samples from the data length
        reshaped_data = data.reshape(-1, num_channels)

        # Print some information about the data
        print(f"Successfully read {file_path}")
        print(f"Data shape: {reshaped_data.shape}")
        print(f"Number of samples: {reshaped_data.shape[0]}")
        print(f"Number of channels: {reshaped_data.shape[1]}")

        # Plot all channels on the same figure with different colors and opacity
        plt.figure(figsize=(15, 5))
        for i in range(reshaped_data.shape[1]):
            plt.plot(reshaped_data[:, i], alpha=0.2, label=f'Channel {i+1}')
        
        plt.title('All Stimulus Channels')
        plt.xlabel('Sample Number')
        plt.ylabel('Amplitude')
        plt.grid(True)
        # plt.legend() # The legend can be crowded with 10 channels, optional

        # Save the plot to a file
        output_filename = 'stimuli_all_channels.png'
        plt.savefig(output_filename)
        print(f"Plot saved to {output_filename}")
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # The path to your stimuli file
    stimuli_file = '/home/v/v/zebra/data/stimuli_raw/stimuli_and_ephys.10chFlt'
    
    # Call the function to read and analyze the data
    read_and_analyze_stimuli(stimuli_file)
