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

# libevent
install(FILES ${libevent_LIBPATH}/libevent-2.1.so.7.0.1
        DESTINATION ${INSTALL_LIB_DIR} RENAME libevent-2.1.so.7 COMPONENT mindspore_federated)
install(FILES ${libevent_LIBPATH}/libevent_core-2.1.so.7.0.1
        DESTINATION ${INSTALL_LIB_DIR} RENAME libevent_core-2.1.so.7 COMPONENT mindspore)
install(FILES ${libevent_LIBPATH}/libevent_openssl-2.1.so.7.0.1
        DESTINATION ${INSTALL_LIB_DIR} RENAME libevent_openssl-2.1.so.7 COMPONENT mindspore_federated)
install(FILES ${libevent_LIBPATH}/libevent_pthreads-2.1.so.7.0.1
        DESTINATION ${INSTALL_LIB_DIR} RENAME libevent_pthreads-2.1.so.7 COMPONENT mindspore_federated)

# glog
install(FILES ${glog_LIBPATH}/libmindspore_federated_glog.so.0.4.0
        DESTINATION ${INSTALL_LIB_DIR} RENAME libmindspore_federated_glog.so.0 COMPONENT mindspore)

# set python files
file(GLOB MS_PY_LIST ${CMAKE_SOURCE_DIR}/mindspore_federated/*.py)
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
        ${CMAKE_SOURCE_DIR}/mindspore_federated/fl_arch/python
        DESTINATION ${INSTALL_PY_DIR}
        COMPONENT mindspore_federated
)