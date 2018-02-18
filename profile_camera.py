from camera_profiler import Profiler

# settings
num_cores = 4
patch_radius = 10

# import image
path_input = "/run/media/bjoern/daten/Programming/Raw_NLM_Denoise/images/Profiling_D40/DSC_1577.NEF"
path_output = "/run/media/bjoern/daten/Programming/Raw_NLM_Denoise/images/bla.png"


profiler = Profiler(patch_radius, num_cores=num_cores)

profiler.profile_camera(path_input)