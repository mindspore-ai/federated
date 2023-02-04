if(MSVC)
    set(flatbuffers_CXXFLAGS "${CMAKE_CXX_FLAGS}")
    set(flatbuffers_CFLAGS "${CMAKE_C_FLAGS}")
    set(flatbuffers_LDFLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
else()
    set(flatbuffers_CXXFLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O2 -fstack-protector-strong")
    set(flatbuffers_CFLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O2 -fstack-protector-strong")
endif()
if(ENABLE_GITEE_EULER)
    set(GIT_REPOSITORY "https://gitee.com/src-openeuler/flatbuffers.git")
    set(GIT_TAG "openEuler-22.03-LTS")
    set(SHA256 "d94ef2fb0c22198c7ffe2a6044e864bd467ca70b8cfdc52720dc94313321777b")
    set(FLATBUFFER_SRC "${TOP_DIR}/mindspore/lite/build/_deps/flatbuffers-src")
    set(FLATBUFFER_DIR "${FLATBUFFER_SRC}/flatbuffers-2.0.0")
    __download_pkg_with_git(flatbuffers ${GIT_REPOSITORY} ${GIT_TAG} ${SHA256})
    execute_process(COMMAND tar -xf ${FLATBUFFER_SRC}/v2.0.0.tar.gz WORKING_DIRECTORY ${FLATBUFFER_SRC})

    foreach(_SUBMODULE_FILE ${PKG_SUBMODULES})
        STRING(REGEX REPLACE "(.+)_(.+)" "\\1" _SUBMODEPATH ${_SUBMODULE_FILE})
        STRING(REGEX REPLACE "(.+)/(.+)" "\\2" _SUBMODENAME ${_SUBMODEPATH})
        file(GLOB ${pkg_name}_INSTALL_SUBMODULE ${_SUBMODULE_FILE}/*)
        file(COPY ${${pkg_name}_INSTALL_SUBMODULE} DESTINATION ${${pkg_name}_SOURCE_DIR}/3rdparty/${_SUBMODENAME})
    endforeach()
else()
if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/flatbuffers/repository/archive/v2.0.0.tar.gz")
    set(SHA256 "3d1eabe298ddac718de34d334aefc22486064dcd8e7a367a809d87393d59ac5a")
else()
    set(REQ_URL "https://github.com/google/flatbuffers/archive/v2.0.0.tar.gz")
    set(SHA256 "9ddb9031798f4f8754d00fca2f1a68ecf9d0f83dfac7239af1311e4fd9a565c4")
endif()
endif()
find_path(GCC_PATH gcc)
find_path(GXX_PATH g++)

set(FLATC_GXX_COMPILER ${GXX_PATH}/g++)
set(FLATC_GCC_COMPILER ${GCC_PATH}/gcc)
mindspore_add_pkg(flatbuffers
        VER 2.0.0
        LIBS flatbuffers
        EXE flatc
        URL ${REQ_URL}
        SHA256 ${SHA256}
        DIR ${FLATBUFFER_DIR}
        CMAKE_OPTION -DCMAKE_C_COMPILER=${FLATC_GCC_COMPILER} -DCMAKE_CXX_COMPILER=${FLATC_GXX_COMPILER}
        -DFLATBUFFERS_BUILD_TESTS=OFF -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_BUILD_TYPE=Release)

include_directories(${flatbuffers_INC})
add_executable(mindspore_federated::flatc ALIAS flatbuffers::flatc)
function(ms_build_flatbuffers source_schema_files
                              source_schema_dirs
                              custom_target_name
                              generated_output_dir)

    set(total_schema_dirs "")
    set(total_generated_files "")
    set(FLATC mindspore_federated::flatc)
    foreach(schema_dir ${source_schema_dirs})
        set(total_schema_dirs -I ${schema_dir} ${total_schema_dirs})
    endforeach()

    foreach(schema IN LISTS ${source_schema_files})
        get_filename_component(filename ${schema} NAME_WE)
        if(NOT ${generated_output_dir} STREQUAL "")
            set(generated_file ${generated_output_dir}/${filename}_generated.h)
            add_custom_command(
                    OUTPUT ${generated_file}
                    COMMAND ${FLATC} --gen-mutable
                    --reflect-names --gen-object-api -o ${generated_output_dir}
                    ${total_schema_dirs}
                    -c -b --reflect-types ${schema}
                    DEPENDS ${FLATC} ${schema}
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                    COMMENT "Running C++ flatbuffers compiler on ${schema}" VERBATIM)
            list(APPEND total_generated_files ${generated_file})
        endif()
    endforeach()

    add_custom_target(${custom_target_name} ALL
            DEPENDS ${total_generated_files})

    if(NOT ${generated_output_dir} STREQUAL "")
        include_directories(${generated_output_dir})
        set_property(TARGET ${custom_target_name}
                PROPERTY GENERATED_OUTPUT_DIR
                ${generated_output_dir})
    endif()
endfunction()

add_library(mindspore_federated::flatbuffers ALIAS flatbuffers::flatbuffers)
add_executable(mindspore_federated::flatc ALIAS flatbuffers::flatc)
set(CMAKE_CXX_FLAGS  ${_ms_tmp_CMAKE_CXX_FLAGS})
if(MSVC)
    set(CMAKE_STATIC_LIBRARY_PREFIX, ${_ms_tmp_CMAKE_STATIC_LIBRARY_PREFIX})
endif()
