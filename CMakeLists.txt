cmake_minimum_required(VERSION 3.20.6...3.22.6 FATAL_ERROR)

# Inherit the SDK's macOS architecture/sysroot before project() enables compilers.
find_package(Slicer COMPONENTS ConfigurePrerequisites REQUIRED)
project(SlicerDSpace)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/NeuroDMRI/SlicerDSpace")
set(EXTENSION_CATEGORY "Diffusion")
set(EXTENSION_CONTRIBUTORS "Chengjin Li, Fan Zhang (Fan's Lab, UESTC)")
set(EXTENSION_DESCRIPTION "Integrated diffusion MRI workflows for anatomical reconstruction, tractography, bundle analysis, and structural connectivity.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/NeuroDMRI/SlicerDSpace/main/Resources/Icons/SlicerDSpace.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/NeuroDMRI/SlicerDSpace/main/docs/images/slicerdds-compact-en.png")
set(EXTENSION_DEPENDS PyTorch)

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
set(MODULE_NAME SlicerDSpace)

#-----------------------------------------------------------------------------
include("${CMAKE_CURRENT_SOURCE_DIR}/scripts/extension_runtime/RuntimeResources.cmake")
slicerdsurfer_runtime_resources("${CMAKE_CURRENT_SOURCE_DIR}" MODULE_PYTHON_RESOURCES)
list(REMOVE_ITEM MODULE_PYTHON_RESOURCES "${MODULE_NAME}.py")
set(MODULE_PYTHON_SCRIPTS ${MODULE_NAME}.py)

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_CPACK})
