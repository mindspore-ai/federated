message("build ut testcases...")

# virtual project for common include and library file path.
project(ut)

set(PROJECT_DIR "${PROJECT_SOURCE_DIR}/../../..")

# add include for all ut executables.
include_directories(${CMAKE_SOURCE_DIR}/mindspore_federated/fl_arch/ccsrc)
include_directories(${CMAKE_SOURCE_DIR}/mindspore_federated/fl_arch/ccsrc/armour)
include_directories(${CMAKE_SOURCE_DIR}/mindspore_federated/fl_arch/ccsrc/common)
include_directories(${CMAKE_SOURCE_DIR}/mindspore_federated/fl_arch/ccsrc/common/utils)
include_directories(${CMAKE_SOURCE_DIR}/mindspore_federated/fl_arch/ccsrc/vertical)
include_directories(${CMAKE_BINARY_DIR})
MESSAGE("check  ut_test ${CMAKE_BINARY_DIR}")

link_directories(${BUILD_PATH}/mindspore_federated/mindspore_federated/ccsrc)

add_definitions(-DEnclavePATH="${CMAKE_CURRENT_SOURCE_DIR}/fl_arch/ccsrc/armour/lib/enclave.signed.so")

file(GLOB_RECURSE MINDSPORE_FEDERATED_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "../../../mindspore_federated/fl_arch/ccsrc/armour/*.cc"
        "../../../mindspore_federated/fl_arch/ccsrc/common/*.cc"
        "../../../mindspore_federated/fl_arch/ccsrc/compression/*.cc"
        "../../../mindspore_federated/fl_arch/ccsrc/scheduler/*.cc"
        "../../../mindspore_federated/fl_arch/ccsrc/server/*.cc"
        "../../../mindspore_federated/fl_arch/ccsrc/worker/*.cc"
        "../../../mindspore_federated/fl_arch/ccsrc/vertical/*.cc"
        )

file(GLOB_RECURSE UT_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        ./common/*.cc
        ./communicator/*.cc
        ./psi/*.cc
        )

if(NOT (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64" AND ENABLE_SGX))
    list(REMOVE_ITEM MINDSPORE_FEDERATED_SRC_LIST
        "../../../mindspore_federated/fl_arch/ccsrc/armour/secure_protocol/enclave_call.cc")
endif()

add_library(_ut_mindspore_federated_obj OBJECT ${MINDSPORE_FEDERATED_SRC_LIST})
add_library(_ut_ut_obj OBJECT ${UT_SRCS})

set(ut_objects $<TARGET_OBJECTS:_ut_ut_obj> $<TARGET_OBJECTS:_ut_mindspore_federated_obj>)

add_executable(ut_tests ${ut_objects})

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

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64" AND ENABLE_SGX)
    add_compile_definitions(ENABLE_TEE)
    target_link_libraries(ut_tests PRIVATE
                          ${CMAKE_SOURCE_DIR}/mindspore_federated/fl_arch/ccsrc/armour/lib/libsgx_0.so)
    target_link_libraries(ut_tests PRIVATE
                          ${CMAKE_SOURCE_DIR}/mindspore_federated/fl_arch/ccsrc/armour/lib/libcsecure_channel_static.a)
    find_library(SGX_SSL_LIB NAMES sgx_usgxssl HINTS /opt/intel/sgxssl/lib64)
    target_link_libraries(ut_tests PRIVATE ${SGX_SSL_LIB})
    target_link_libraries(ut_tests PRIVATE
                          ${CMAKE_SOURCE_DIR}/mindspore_federated/fl_arch/ccsrc/armour/lib/libsecgear.so)
endif()

