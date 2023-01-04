# include dependency
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# set package information
set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
set(CPACK_GENERATOR "External")
set(CPACK_EXTERNAL_PACKAGE_SCRIPT ${CMAKE_SOURCE_DIR}/cmake/package_script.cmake)
set(CPACK_EXTERNAL_ENABLE_STAGING true)
set(CPACK_TEMPORARY_PACKAGE_FILE_NAME ${CMAKE_SOURCE_DIR}/build/package/mindspore_federated)
set(CPACK_TEMPORARY_INSTALL_DIRECTORY ${CMAKE_SOURCE_DIR}/build/package/mindspore_federated)

set(CPACK_MS_PACKAGE_NAME "mindspore_federated")
include(CPack)

# set install path
set(INSTALL_LIB_DIR ${CMAKE_INSTALL_LIBDIR} CACHE PATH "Installation directory for libraries")
set(INSTALL_PY_DIR ".")
set(INSTALL_BASE_DIR ".")
set(INSTALL_LIB_DIR "lib")

message("INSTALL_LIB_DIR:::" ${INSTALL_LIB_DIR})

# glog
install(FILES ${glog_LIBPATH}/libmindspore_federated_glog.so.0.4.0
        DESTINATION ${INSTALL_LIB_DIR} RENAME libmindspore_federated_glog.so.0 COMPONENT mindspore_federated)


install(FILES ${hiredis_LIBPATH}/libhiredis.so.1.0.0 DESTINATION ${INSTALL_LIB_DIR} COMPONENT mindspore_federated)
install(FILES ${hiredis_LIBPATH}/libhiredis_ssl.so.1.0.0 DESTINATION ${INSTALL_LIB_DIR} COMPONENT mindspore_federated)

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64" AND ENABLE_SGX)
        install(FILES ${CMAKE_SOURCE_DIR}/mindspore_federated/fl_arch/ccsrc/armour/lib/libsgx_0.so
                DESTINATION ${INSTALL_LIB_DIR} COMPONENT mindspore_federated)
endif()

# process proto files
set(protoc_abs_path ${protobuf_ROOT}/bin/protoc)
message(find_protoc_path: ${protoc_abs_path})
set(proto_out_path ${CMAKE_SOURCE_DIR}/mindspore_federated/fl_arch/python/mindspore_federated/common)
set(proto_src_path ${CMAKE_SOURCE_DIR}/mindspore_federated/fl_arch/python/mindspore_federated/common/protos)
file(GLOB proto_list ${proto_src_path}/*.proto)
foreach(proto_path ${proto_list})
        get_filename_component(proto_file_absolute ${proto_path} ABSOLUTE)
        execute_process(
                COMMAND ${protoc_abs_path}
                ${proto_file_absolute}
                --python_out ${proto_out_path}
                -I ${proto_src_path})
endforeach()

# set python files
file(GLOB MS_PY_LIST ${CMAKE_SOURCE_DIR}/mindspore_federated/fl_arch/python/mindspore_federated/*.py)
install(
        FILES ${MS_PY_LIST}
        DESTINATION ${INSTALL_PY_DIR}
        COMPONENT mindspore_federated
)

install(
        TARGETS _mindspore_federated
        DESTINATION ${INSTALL_BASE_DIR}
        COMPONENT mindspore_federated
)
install(
        TARGETS federated
        DESTINATION ${INSTALL_LIB_DIR}
        COMPONENT mindspore_federated
)

install(
        DIRECTORY
        ${CMAKE_SOURCE_DIR}/mindspore_federated/fl_arch/python/mindspore_federated/aggregation
        ${CMAKE_SOURCE_DIR}/mindspore_federated/fl_arch/python/mindspore_federated/common
        ${CMAKE_SOURCE_DIR}/mindspore_federated/fl_arch/python/mindspore_federated/data_join
        ${CMAKE_SOURCE_DIR}/mindspore_federated/fl_arch/python/mindspore_federated/dataset
        ${CMAKE_SOURCE_DIR}/mindspore_federated/fl_arch/python/mindspore_federated/trainer
        ${CMAKE_SOURCE_DIR}/mindspore_federated/fl_arch/python/mindspore_federated/startup
        ${CMAKE_SOURCE_DIR}/mindspore_federated/fl_arch/python/mindspore_federated/privacy
        ${CMAKE_SOURCE_DIR}/mindspore_federated/fl_arch/python/mindspore_federated/compress
        DESTINATION ${INSTALL_PY_DIR}
        COMPONENT mindspore_federated
)