message("build ut testcases...")

# virtual project for common include and library file path.
project(ut)

set(PROJECT_DIR "${PROJECT_SOURCE_DIR}/../../..")

# add include for all ut executables.
include_directories(${CMAKE_SOURCE_DIR}/mindspore_federated/fl_arch/ccsrc)
include_directories(${CMAKE_SOURCE_DIR}/mindspore_federated/fl_arch/ccsrc/armour)
include_directories(${CMAKE_SOURCE_DIR}/mindspore_federated/fl_arch/ccsrc/common)
include_directories(${CMAKE_SOURCE_DIR}/mindspore_federated/fl_arch/ccsrc/common/utils)
include_directories(${CMAKE_BINARY_DIR})
MESSAGE("check  ut_test ${CMAKE_BINARY_DIR}")

link_directories(${BUILD_PATH}/mindspore_federated/mindspore_federated/ccsrc)
link_directories(${BUILD_PATH}/mindspore_federated/common/protos)

file(GLOB_RECURSE MINDSPORE_FEDERATED_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "../../../mindspore_federated/fl_arch/ccsrc/armour/base_crypto/*.cc"
        "../../../mindspore_federated/fl_arch/ccsrc/armour/secure_protocol/psi.cc"
        "../../../mindspore_federated/fl_arch/ccsrc/armour/util/*.cc"
        "../../../mindspore_federated/fl_arch/ccsrc/common/*.cc"
        )

file(GLOB_RECURSE UT_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        ./common/*.cc
        ./psi/*.cc
        )

add_library(_ut_mindspore_federated_obj OBJECT ${MINDSPORE_FEDERATED_SRC_LIST})
add_library(_ut_ut_obj OBJECT ${UT_SRCS})
#add_dependencies(_ut_ut_obj engine-cache-server)


set(ut_objects $<TARGET_OBJECTS:_ut_ut_obj> $<TARGET_OBJECTS:_ut_mindspore_federated_obj>)

add_executable(ut_tests ${ut_objects})

add_dependencies(ut_tests
        mindspore_federated::ssl
        mindspore_federated::crypto
        mindspore_federated::protobuf
        mindspore_federated::event
        mindspore_federated::event_pthreads
        mindspore_federated::event_core
        mindspore_federated::event_openssl
        mindspore_federated::glog
        ${SECUREC_LIBRARY}
        mindspore_federated::hiredis
        mindspore_federated::hiredis_ssl
        mindspore_federated::gtest
        )

target_link_libraries(ut_tests PRIVATE
        PROTO_SRC_LIB
        mindspore_federated::ssl
        mindspore_federated::crypto
        mindspore_federated::protobuf pthread rt
        mindspore_federated::event
        mindspore_federated::event_pthreads
        mindspore_federated::event_core
        mindspore_federated::event_openssl
        mindspore_federated::glog
        ${SECUREC_LIBRARY}
        mindspore_federated::hiredis
        mindspore_federated::hiredis_ssl
        ${PYTHON_LIBRARIES} pthread util dl
        mindspore_federated::gtest
        )


