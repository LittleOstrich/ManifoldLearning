project(Manifold_learning)

set(OpenCV_DIR "C:/Users/Benjamin/Desktop/LainProject/C++/opencv/opencv/build/x64/vc15/lib")

message("OpenCV_FOUND='${OpenCV_FOUND}'")
find_package(OpenCV REQUIRED )
message("OpenCV_FOUND='${OpenCV_FOUND}'")
include_directories( ${OpenCV_INCLUDE_DIRS} )

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "Version: ${OpenCV_VERSION}")
message(STATUS "Libraries: ${OpenCV_LIBS}")
message(STATUS "Include path: ${OpenCV_INCLUDE_DIRS}")
message("These are my opencv libs....: '${OpenCV_LIBS}'")

add_executable (${PROJECT_NAME}
    "main.cpp"
    # "main2.cpp"
    # "main2.h"
    "floydmarshall.h"
    "floydmarshall.cpp"
    "PCA.cpp"
    "PCA.h"
    "isomap.cpp"
    "isomap.h"
    "utils.cpp"
    "utils.h"
    )

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

