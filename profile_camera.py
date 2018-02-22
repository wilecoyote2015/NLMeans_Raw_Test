from camera_profiler import Profiler
import os

# settings
num_cores = 4
patch_radius = 10

# import image
path_input = "/run/media/bjoern/daten/Programming/Raw_NLM_Denoise/images/Profiling_X_T20/DSCF7134.RAF"
# path_input = "/run/media/bjoern/daten/Programming/Raw_NLM_Denoise/images/Olympus_200_0.ORF"
# path_input = "/run/media/bjoern/daten/Programming/Raw_NLM_Denoise/images/Profiling_D40/DSC_1577.NEF"
dir_output = "/run/media/bjoern/daten/Programming/Raw_NLM_Denoise/images/results"
output_parameters = os.path.join(dir_output, "profile_T10_3200.csv")


profiler = Profiler(patch_radius, num_cores=num_cores)

parameters = profiler.profile_camera(path_input, dir_output)
profiler.save_parameters(parameters, output_parameters)