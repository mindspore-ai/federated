# This branch assumes all its dependencies are already installed
# on this system, so they can be located by find_package().

# Find Protobuf installation
# Looks for protobuf-config.cmake file installed by Protobuf's cmake installation.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,$ORIGIN:$ORIGIN/lib -Wl,--no-as-needed")

if(ENABLE_COVERAGE)
    add_compile_options(-coverage)
    add_link_options(-lgcov --coverage)
endif()

# Proto file
# Generated sources
file(GLOB_RECURSE PROTO_FILE_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./fl_arch/ccsrc/common/protos/*.proto)
ms_protobuf_generate(PROTO_SRC_LIST PROTO_HDR_LIST ${PROTO_FILE_LIST})
add_library(PROTO_SRC_LIB STATIC ${PROTO_SRC_LIST})
target_compile_options(PROTO_SRC_LIB PRIVATE "-Wno-array-bounds")

include_directories("${CMAKE_BINARY_DIR}/mindspore_federated" ${CMAKE_BINARY_DIR}) # for proto header file
include_directories("fl_arch")
include_directories("fl_arch/ccsrc")
include_directories("fl_arch/ccsrc/common")
include_directories()

set(SERVER_FLATBUFFER_OUTPUT "${CMAKE_BINARY_DIR}/schema")
set(FBS_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/schema/cipher.fbs
        ${CMAKE_CURRENT_SOURCE_DIR}/schema/fl_job.fbs
        )
ms_build_flatbuffers(FBS_FILES ${CMAKE_CURRENT_SOURCE_DIR}../schema generated_fbs_files ${SERVER_FLATBUFFER_OUTPUT})

# federated for c++ server and python interface
file(GLOB_RECURSE FEDERATED_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "fl_arch/ccsrc/armour/*.cc"
        "fl_arch/ccsrc/common/*.cc"
        "fl_arch/ccsrc/compression/*.cc"
        "fl_arch/ccsrc/scheduler/*.cc"
        "fl_arch/ccsrc/server/*.cc"
        "fl_arch/ccsrc/worker/*.cc")

add_library(federated SHARED ${FEDERATED_SRC})

include(CheckPIESupported)
check_pie_supported()
set_property(TARGET federated PROPERTY POSITION_INDEPENDENT_CODE TRUE)

target_link_libraries(federated PRIVATE PROTO_SRC_LIB)
target_link_libraries(federated PRIVATE mindspore_federated::ssl mindspore_federated::crypto)
target_link_libraries(federated PRIVATE mindspore_federated::protobuf pthread rt)
target_link_libraries(federated PRIVATE mindspore_federated::event mindspore_federated::event_pthreads)
target_link_libraries(federated PRIVATE mindspore_federated::event_core)
target_link_libraries(federated PRIVATE mindspore_federated::event_openssl)
target_link_libraries(federated PRIVATE mindspore_federated::glog)
target_link_libraries(federated PRIVATE ${SECUREC_LIBRARY})
target_link_libraries(federated PRIVATE mindspore_federated::flatbuffers)
target_link_libraries(federated PRIVATE mindspore_federated::hiredis)
target_link_libraries(federated PRIVATE mindspore_federated::hiredis_ssl)

set_target_properties(federated PROPERTIES SKIP_BUILD_RPATH TRUE)

# python
add_compile_definitions(ENABLE_PYTHON)
file(GLOB_RECURSE PY_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "fl_arch/ccsrc/python/*.cc")

find_package(Python3 3.7 COMPONENTS Interpreter Development)
if(Python3_FOUND)
    set(PYTHON_INCLUDE_DIRS "${Python3_INCLUDE_DIRS}")
    set(PYTHON_LIBRARIES "${Python3_LIBRARIES}")
else()
    find_python_package(py_inc py_lib)
    set(PYTHON_INCLUDE_DIRS "${py_inc}")
    set(PYTHON_LIBRARIES "${py_lib}")
endif()

include_directories(${PYTHON_INCLUDE_DIRS})
pybind11_add_module(_mindspore_federated NO_EXTRAS ${PY_SRC_LIST})
set_target_properties(_mindspore_federated PROPERTIES LINK_FLAGS_RELEASE -s)
target_link_libraries(_mindspore_federated PRIVATE "${PYTHON_LIBRARIES}")
target_include_directories(_mindspore_federated PRIVATE ${pybind11_INCLUDE_DIRS})
target_link_libraries(_mindspore_federated PRIVATE federated)
set_property(TARGET _mindspore_federated PROPERTY POSITION_INDEPENDENT_CODE TRUE)

target_link_options(federated PRIVATE -Wl,-init,mindspore_federated_log_init)
