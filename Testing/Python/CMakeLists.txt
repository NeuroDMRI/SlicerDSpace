set(SLICERDSPACE_TEST_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# Backend unit fixtures patch Qt/process state and must not run inside the
# embedded Slicer interpreter. Native module tests are registered at the root;
# real GUI/science smoke scripts use scripts/slicer_acceptance.py separately.
set(SLICERDSPACE_UNIT_TEST_PYTHON "" CACHE FILEPATH
  "Independent Python with backend test dependencies (not an embedded Slicer app)")
if(NOT SLICERDSPACE_UNIT_TEST_PYTHON)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(SLICERDSPACE_UNIT_TEST_PYTHON "${Python3_EXECUTABLE}")
endif()
file(GLOB SLICERDSPACE_UNIT_TESTS CONFIGURE_DEPENDS
  "${SLICERDSPACE_TEST_SOURCE_DIR}/test_*.py")
foreach(TEST_SCRIPT IN LISTS SLICERDSPACE_UNIT_TESTS)
  get_filename_component(TEST_NAME "${TEST_SCRIPT}" NAME_WE)
  get_filename_component(TEST_FILENAME "${TEST_SCRIPT}" NAME)
  add_test(NAME "py_${TEST_NAME}"
    COMMAND "${SLICERDSPACE_UNIT_TEST_PYTHON}" -m unittest discover
      -s "${SLICERDSPACE_TEST_SOURCE_DIR}" -p "${TEST_FILENAME}")
  # B2: each test file runs alone (no test_00_user_data_isolation.py), so the per-user data folders of every OS point
  # into the build folder here. A value in the caller's shell for the three SlicerDSpace path variables is blanked.
  set(SLICERDSPACE_TEST_HOME "${CMAKE_CURRENT_BINARY_DIR}/test-home/${TEST_NAME}")
  set_tests_properties("py_${TEST_NAME}" PROPERTIES
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" LABELS "backend-unit"
    ENVIRONMENT "HOME=${SLICERDSPACE_TEST_HOME};USERPROFILE=${SLICERDSPACE_TEST_HOME};XDG_CONFIG_HOME=${SLICERDSPACE_TEST_HOME}/.config;XDG_DATA_HOME=${SLICERDSPACE_TEST_HOME}/.local/share;XDG_CACHE_HOME=${SLICERDSPACE_TEST_HOME}/.cache;APPDATA=${SLICERDSPACE_TEST_HOME}/AppData/Roaming;LOCALAPPDATA=${SLICERDSPACE_TEST_HOME}/AppData/Local;SLICERDSPACE_USER_CONFIG=;SLICERDSPACE_MODEL_ASSETS_DIR=;SLICERDSPACE_RUNTIME_ROOT=")
endforeach()
